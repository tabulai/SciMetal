// E8 probe: hand mlx::core::array values across the Python boundary WITHOUT
// nanobind type sharing.
//
// Why this exists
// ---------------
// The documented way for an out-of-tree MLX extension to accept / return
// `mx.array` is to build with nanobind using `NB_DOMAIN mlx`, so that the
// extension shares nanobind's per-domain type registry ("internals") with
// mlx/core.cpython-*.so.  That only works when BOTH modules were built with a
// nanobind whose internals ABI matches:
//
//   mlx 0.31.1 core  : nanobind v2.10.2, key "__nb_internals_v17_system_libcpp_abi1_mlx__"
//   this venv        : nanobind 3.1.0,  internals v22 (different key, different layout)
//
// With mismatching internals the two type universes are disjoint and a
// nanobind-typed `mx::array` argument is rejected with a TypeError (the
// `_e8_nbnative` module in this probe demonstrates that).
//
// This header is the pinned workaround used by the probe: it reads the C++
// payload straight out of the nanobind-2.x instance object.  It relies on
//   (1) the `nb_inst` layout of nanobind internals v17 (checked at init by
//       looking up the internals key in the interpreter state dict and by a
//       round-trip self test), and
//   (2) sizeof/layout of mlx::core::array in MLX 0.31.1 (a single
//       std::shared_ptr<ArrayDesc>).
// A production bridge should instead build against the nanobind version that
// the pinned MLX wheel was built with and let nanobind do this.

#pragma once

#include <Python.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "mlx/array.h"
#include "mlx/version.h"

namespace e8::pybridge {

namespace mx = mlx::core;

// nanobind 2.x (internals v17) instance header: PyObject_HEAD followed by a
// 32-bit offset to the payload and a 32-bit flag word.
struct nb_inst_v17 {
  PyObject_HEAD
  int32_t offset;
  uint32_t state : 2;  // 0 = uninitialized, 1 = relinquished, 2 = ready
  uint32_t direct : 1; // payload stored at `offset` (else: pointer to it)
  uint32_t internal : 1;
  uint32_t destruct : 1;
  uint32_t cpp_delete : 1;
  uint32_t clear_keep_alive : 1;
  uint32_t intrusive : 1;
  uint32_t unused : 24;
};

inline constexpr const char* kExpectedInternalsKey =
    "__nb_internals_v17_system_libcpp_abi1_mlx__";

struct State {
  PyTypeObject* array_type = nullptr; // mlx.core.array (strong ref)
  PyObject* proto = nullptr;          // a 0-d mx.array used to mint new objects
  bool ready = false;
  bool internals_key_found = false;
  std::string runtime_mlx_version;
  std::string error;
};

inline State& state() {
  static State s;
  return s;
}

// Borrow the C++ array held by a Python `mx.array` object.
inline mx::array& from_py(PyObject* o) {
  auto& st = state();
  if (!st.ready) {
    throw std::runtime_error("e8 pybridge not initialised: " + st.error);
  }
  if (!PyObject_TypeCheck(o, st.array_type)) {
    std::string msg = "expected an mlx.core.array, got ";
    msg += Py_TYPE(o)->tp_name;
    throw std::invalid_argument(msg);
  }
  auto* inst = reinterpret_cast<nb_inst_v17*>(o);
  if (inst->state != 2) {
    throw std::invalid_argument("mlx.core.array instance is not initialised");
  }
  void* p = reinterpret_cast<void*>(
      reinterpret_cast<intptr_t>(inst) + inst->offset);
  if (!inst->direct) {
    p = *static_cast<void**>(p);
  }
  return *static_cast<mx::array*>(p);
}

// Wrap a C++ array into a brand new Python `mx.array` (new reference).
// `mx.array(proto)` yields a fresh instance sharing proto's descriptor; the
// descriptor is then swapped for ours with a plain C++ assignment, so no
// nanobind state is ever written by this module.
inline PyObject* to_py(mx::array a) {
  auto& st = state();
  if (!st.ready) {
    throw std::runtime_error("e8 pybridge not initialised: " + st.error);
  }
  // (PyObject_Vectorcall rather than PyObject_CallOneArg: the latter is not in
  // nanobind's list of CPython symbols allowed to stay undefined at link time.)
  PyObject* args[1] = {st.proto};
  PyObject* o = PyObject_Vectorcall(
      reinterpret_cast<PyObject*>(st.array_type), args, 1, nullptr);
  if (!o) {
    PyErr_Clear();
    throw std::runtime_error("could not construct an mlx.core.array");
  }
  from_py(o) = std::move(a);
  return o;
}

// One-time setup + safety checks.  Returns false (with state().error set) if
// the running MLX does not look like the one this bridge is pinned to.
inline bool init() {
  auto& st = state();
  if (st.ready) {
    return true;
  }
  st.runtime_mlx_version = mx::version();
  const std::string built = std::to_string(MLX_VERSION_MAJOR) + "." +
      std::to_string(MLX_VERSION_MINOR) + "." +
      std::to_string(MLX_VERSION_PATCH);
  if (st.runtime_mlx_version != built) {
    st.error = "libmlx runtime version " + st.runtime_mlx_version +
        " != headers this bridge was built against (" + built + ")";
    return false;
  }

  PyObject* core = PyImport_ImportModule("mlx.core");
  if (!core) {
    PyErr_Clear();
    st.error = "cannot import mlx.core";
    return false;
  }
  PyObject* type = PyObject_GetAttrString(core, "array");
  Py_DECREF(core);
  if (!type || !PyType_Check(type)) {
    PyErr_Clear();
    Py_XDECREF(type);
    st.error = "mlx.core.array is not a type";
    return false;
  }

  // nanobind stores each domain's internals in the interpreter state dict.
  PyObject* idict = PyInterpreterState_GetDict(PyInterpreterState_Get());
  PyObject* key = PyUnicode_FromString(kExpectedInternalsKey);
  st.internals_key_found = idict && key && PyDict_Contains(idict, key) == 1;
  Py_XDECREF(key);
  if (!st.internals_key_found) {
    Py_DECREF(type);
    st.error = std::string("nanobind internals key ") + kExpectedInternalsKey +
        " not found: mlx.core was built with a different nanobind";
    return false;
  }

  auto* tp = reinterpret_cast<PyTypeObject*>(type);
  if (static_cast<size_t>(tp->tp_basicsize) <
      sizeof(nb_inst_v17) + sizeof(mx::array)) {
    Py_DECREF(type);
    st.error = "mlx.core.array instance is smaller than expected";
    return false;
  }

  // Self test: a (2, 3) int32 array built by MLX's own bindings must be
  // readable through our view of the instance layout.
  PyObject* probe = PyObject_CallFunction(type, "([[iii][iii]])", 1, 2, 3, 4, 5, 6);
  PyObject* proto = PyObject_CallFunction(type, "(d)", 0.0);
  if (!probe || !proto) {
    PyErr_Clear();
    Py_XDECREF(probe);
    Py_XDECREF(proto);
    Py_DECREF(type);
    st.error = "could not construct probe arrays";
    return false;
  }
  st.array_type = tp;
  st.proto = proto;
  st.ready = true;
  bool ok = false;
  try {
    const mx::array& a = from_py(probe);
    ok = a.ndim() == 2 && a.shape(0) == 2 && a.shape(1) == 3 &&
        a.dtype() == mx::int32 && a.size() == 6;
  } catch (const std::exception&) {
    ok = false;
  }
  Py_DECREF(probe);
  if (!ok) {
    st.ready = false;
    st.error = "instance layout self test failed";
    return false;
  }
  return true;
}

} // namespace e8::pybridge
