// E8 probe: Python bindings (nanobind 3.1.0) for the out-of-tree MLX
// primitive and the hand-off / instrumentation probes.
//
// mx.array values cross the module boundary through mlx_pybridge.h (see the
// comment there for why nanobind's own type sharing cannot be used with the
// nanobind that is installed in this environment).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "e8_primitive.h"
#include "metal_counters.h"
#include "mlx_pybridge.h"

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;
using e8::pybridge::from_py;
using e8::pybridge::to_py;

namespace {

nb::object addr(const void* p) {
  return nb::int_(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)));
}

// ---------------------------------------------------------------------------
// Zero-copy hand-off probe state
// ---------------------------------------------------------------------------
struct ExtState {
  std::atomic<int> deleter_calls{0};
  void* base = nullptr;     // allocation to free (mmap / malloc)
  size_t base_len = 0;
  int kind = 0;             // 0 = mmap, 1 = malloc, 2 = foreign
  MTL::Buffer* retained = nullptr; // foreign MTLBuffer we hold a retain on
  // Optional pin: a second owner of the array's Data block.  While it is
  // held, array::is_donatable() is false, so MLX never reuses (overwrites)
  // the external memory as the output buffer of an in-place capable op.
  std::shared_ptr<mx::array::Data> pin;

  void release_memory() {
    if (kind == 0 && base) {
      munmap(base, base_len);
    } else if (kind == 1 && base) {
      std::free(base);
    }
    base = nullptr;
    if (retained) {
      retained->release();
      retained = nullptr;
    }
  }
};

std::mutex g_ext_mtx;
std::unordered_map<uint64_t, std::shared_ptr<ExtState>> g_ext;
uint64_t g_ext_next = 1;

size_t page_size() {
  return static_cast<size_t>(getpagesize());
}

size_t round_up_pages(size_t n) {
  const size_t p = page_size();
  return std::max(p, (n + p - 1) / p * p);
}

// Build an mx.array over memory that MLX did not allocate.
//   kind : "mmap"    page-aligned anonymous memory owned by this module
//          "malloc"  malloc'ed memory, deliberately offset by 64 bytes
//          "foreign" caller-provided host pointer (+ optional MTLBuffer handle
//                    that is retained for the lifetime of the array)
//   ctor : "pointer"     mx::array(void*, shape, dtype, deleter)
//          "make_buffer" allocator::make_buffer(ptr, page-rounded length) +
//                        mx::array(allocator::Buffer, shape, dtype, deleter)
//          "adopt"       allocator::Buffer{foreign MTLBuffer} +
//                        mx::array(allocator::Buffer, shape, dtype, deleter)
//   pin  : keep a second reference to the array's Data block (see ExtState);
//          must be dropped with external_unpin(token) or the memory leaks.
nb::object wrap_external(
    const std::string& kind,
    const std::vector<int>& shape_,
    const std::string& ctor,
    uint64_t host_ptr,
    uint64_t mtl_buffer,
    bool pin) {
  mx::Shape shape(shape_.begin(), shape_.end());
  size_t nelem = 1;
  for (int d : shape_) {
    nelem *= static_cast<size_t>(d);
  }
  const size_t nbytes = nelem * sizeof(float);

  auto st = std::make_shared<ExtState>();
  void* ptr = nullptr;
  if (kind == "mmap") {
    st->kind = 0;
    st->base_len = round_up_pages(nbytes);
    st->base = mmap(
        nullptr, st->base_len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (st->base == MAP_FAILED) {
      throw std::runtime_error("mmap failed");
    }
    ptr = st->base;
  } else if (kind == "malloc") {
    st->kind = 1;
    st->base_len = nbytes + 64;
    st->base = std::malloc(st->base_len);
    ptr = static_cast<char*>(st->base) + 64;
  } else if (kind == "foreign") {
    st->kind = 2;
    ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(host_ptr));
    if (mtl_buffer) {
      st->retained =
          reinterpret_cast<MTL::Buffer*>(static_cast<uintptr_t>(mtl_buffer));
      st->retained->retain();
    }
  } else {
    throw std::invalid_argument("unknown kind");
  }

  nb::dict info;
  info["host_ptr"] = addr(ptr);
  info["nbytes"] = nbytes;
  info["host_ptr_page_aligned"] =
      (reinterpret_cast<uintptr_t>(ptr) % page_size()) == 0;
  info["nbytes_page_multiple"] = (nbytes % page_size()) == 0;

  std::optional<mx::array> arr;
  if (ctor == "pointer") {
    arr.emplace(ptr, shape, mx::float32, [st](void*) {
      st->deleter_calls++;
      st->release_memory();
    });
  } else if (ctor == "make_buffer") {
    auto buf = mx::allocator::make_buffer(ptr, round_up_pages(nbytes));
    info["make_buffer_ok"] = buf.ptr() != nullptr;
    if (buf.ptr() == nullptr) {
      st->release_memory();
      return nb::make_tuple(nb::none(), info);
    }
    arr.emplace(buf, shape, mx::float32, [st](mx::allocator::Buffer b) {
      mx::allocator::release(b);
      st->deleter_calls++;
      st->release_memory();
    });
  } else if (ctor == "adopt") {
    if (!st->retained) {
      throw std::invalid_argument("ctor='adopt' needs mtl_buffer");
    }
    arr.emplace(
        mx::allocator::Buffer{st->retained},
        shape,
        mx::float32,
        [st](mx::allocator::Buffer) {
          st->deleter_calls++;
          st->release_memory();
        });
  } else {
    throw std::invalid_argument("unknown ctor");
  }

  uint64_t token;
  {
    std::lock_guard<std::mutex> lk(g_ext_mtx);
    token = g_ext_next++;
    g_ext[token] = st;
  }
  if (pin) {
    st->pin = arr->data_shared_ptr();
  }
  void* data_ptr = arr->data<void>();
  info["token"] = token;
  info["array_data_ptr"] = addr(data_ptr);
  info["zero_copy"] = data_ptr == ptr;
  info["deleter_calls_after_ctor"] = st->deleter_calls.load();
  info["mlx_mtl_buffer"] = addr(arr->buffer().ptr());
  return nb::make_tuple(nb::steal(to_py(std::move(*arr))), info);
}

} // namespace

NB_MODULE(_e8_mlx, m) {
  m.doc() = "E8 probe: MLX primitive encoding metal-sci's own metallib kernel";

  const bool ok = e8::pybridge::init();

  m.def("bridge_info", [ok]() {
    auto& st = e8::pybridge::state();
    nb::dict d;
    d["ready"] = ok;
    d["error"] = st.error;
    d["mode"] = "pinned nb_inst(v17) layout bridge (no nanobind type sharing)";
    d["mlx_runtime_version"] = st.runtime_mlx_version;
    d["mlx_header_version"] = std::to_string(MLX_VERSION_MAJOR) + "." +
        std::to_string(MLX_VERSION_MINOR) + "." +
        std::to_string(MLX_VERSION_PATCH);
    d["mlx_core_nanobind_internals_key"] = e8::pybridge::kExpectedInternalsKey;
    d["mlx_core_nanobind_internals_key_found"] = st.internals_key_found;
    d["extension_nanobind_version"] =
        std::to_string(NB_VERSION_MAJOR) + "." +
        std::to_string(NB_VERSION_MINOR) + "." +
        std::to_string(NB_VERSION_PATCH);
    return d;
  });

  m.def(
      "set_metallib",
      [](const std::string& dir, const std::string& name) {
        e8::set_metallib(dir, name);
      },
      "dir"_a,
      "name"_a = "e8_kernels");
  m.def("metallib_path", []() { return e8::metallib_path(); });

  m.def(
      "correlate1d",
      [](nb::handle x,
         nb::handle weights,
         const std::string& device,
         bool guard,
         int passes) -> nb::object {
        mx::StreamOrDevice s;
        if (device == "gpu") {
          s = mx::Device(mx::Device::gpu);
        } else if (device == "cpu") {
          s = mx::Device(mx::Device::cpu);
        } else if (device != "default") {
          throw std::invalid_argument("device must be default|gpu|cpu");
        }
        return nb::steal(to_py(e8::correlate1d_reflect(
            from_py(x.ptr()), from_py(weights.ptr()), s, guard, passes)));
      },
      "x"_a,
      "weights"_a,
      "device"_a = "default",
      "guard"_a = true,
      "passes"_a = 1,
      "Lazy 1-D correlate along the last axis, 'reflect' boundary, float32.");

  // Identity through the bridge only (no graph node): cost of the crossing.
  m.def("roundtrip", [](nb::handle x) -> nb::object {
    return nb::steal(to_py(from_py(x.ptr())));
  });

  m.def(
      "eval_stats",
      [](bool reset) {
        auto st = e8::eval_stats(reset);
        nb::dict d;
        d["gpu_calls"] = st.gpu_calls;
        d["gpu_total_ns"] = st.gpu_total_ns;
        d["cpu_calls"] = st.cpu_calls;
        nb::list recent;
        for (auto& r : st.recent) {
          nb::dict e;
          e["command_buffer"] = static_cast<uint64_t>(r.command_buffer);
          e["needs_commit_after"] = r.needs_commit_after;
          e["encode_ns"] = r.encode_ns;
          recent.append(e);
        }
        d["recent"] = recent;
        return d;
      },
      "reset"_a = false);

  m.def("install_metal_counters", []() { return e8::install_metal_counters(); });
  m.def(
      "metal_counters",
      [](bool reset) {
        auto c = e8::metal_counters(reset);
        nb::dict d;
        d["cb_commandBuffer"] = c.cb_commandBuffer;
        d["cb_commandBufferWithUnretainedReferences"] =
            c.cb_commandBufferWithUnretainedReferences;
        d["cb_commandBufferWithDescriptor"] = c.cb_commandBufferWithDescriptor;
        d["commit"] = c.commit;
        d["waitUntilCompleted"] = c.waitUntilCompleted;
        d["waitUntilScheduled"] = c.waitUntilScheduled;
        d["addCompletedHandler"] = c.addCompletedHandler;
        d["encodeSignalEvent"] = c.encodeSignalEvent;
        d["computeCommandEncoder"] = c.computeCommandEncoder;
        d["endEncoding"] = c.endEncoding;
        d["dispatchThreads"] = c.dispatchThreads;
        d["dispatchThreadgroups"] = c.dispatchThreadgroups;
        d["sharedEventWait"] = c.sharedEventWait;
        return d;
      },
      "reset"_a = false);

  // State of MLX's default GPU stream (pointer identity of the open command
  // buffer changes whenever MLX commits).
  m.def("stream_state", []() {
    auto s = mx::default_stream(mx::Device::gpu);
    auto& d = mx::metal::device(s.device);
    nb::dict out;
    out["stream_index"] = s.index;
    out["command_buffer"] = addr(d.get_command_buffer(s.index));
    out["needs_commit"] = d.command_buffer_needs_commit(s.index);
    return out;
  });

  m.def("mlx_device_info", []() {
    auto& d = mx::metal::device(mx::Device::gpu);
    nb::dict out;
    out["mtl_device"] = addr(d.mtl_device());
    out["architecture"] = d.get_architecture();
    out["page_size"] = page_size();
    return out;
  });

  // 6(b): raw MTL::Buffer* / host pointer of an evaluated mx.array.
  m.def("buffer_info", [](nb::handle h) {
    mx::array& a = from_py(h.ptr());
    if (a.status() == mx::array::Status::unscheduled) {
      throw std::invalid_argument("array has not been evaluated");
    }
    nb::dict out;
    out["is_available"] = a.is_available();
    out["nbytes"] = a.nbytes();
    out["offset"] = a.offset();
    auto* buf = static_cast<MTL::Buffer*>(a.buffer().ptr());
    out["mtl_buffer"] = addr(buf);
    if (!buf) {
      return out;
    }
    void* contents = buf->contents();
    void* data = a.data<void>();
    out["contents"] = addr(contents);
    out["data_ptr"] = addr(data);
    out["buffer_length"] = static_cast<uint64_t>(buf->length());
    out["storage_mode"] = static_cast<int>(buf->storageMode());
    out["heap_backed"] = buf->heap() != nullptr;
    out["contents_page_aligned"] =
        (reinterpret_cast<uintptr_t>(contents) % page_size()) == 0;
    out["data_page_aligned"] =
        (reinterpret_cast<uintptr_t>(data) % page_size()) == 0;
    out["same_device_as_mlx"] =
        buf->device() == mx::metal::device(mx::Device::gpu).mtl_device();
    return out;
  });

  m.def(
      "wrap_external",
      &wrap_external,
      "kind"_a,
      "shape"_a,
      "ctor"_a = "pointer",
      "host_ptr"_a = 0,
      "mtl_buffer"_a = 0,
      "pin"_a = false);

  m.def("external_unpin", [](uint64_t token) {
    std::shared_ptr<mx::array::Data> drop;
    {
      std::lock_guard<std::mutex> lk(g_ext_mtx);
      auto it = g_ext.find(token);
      if (it == g_ext.end()) {
        throw std::invalid_argument("unknown token");
      }
      drop = std::move(it->second->pin);
    }
    // `drop` is released here, outside the lock (may run the deleter).
  });

  m.def("external_deleter_calls", [](uint64_t token) {
    std::lock_guard<std::mutex> lk(g_ext_mtx);
    auto it = g_ext.find(token);
    if (it == g_ext.end()) {
      throw std::invalid_argument("unknown token");
    }
    return it->second->deleter_calls.load();
  });
}
