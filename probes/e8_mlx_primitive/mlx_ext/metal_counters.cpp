// See metal_counters.h.  Plain C++ against the Objective-C runtime C API.

#include "metal_counters.h"

#include <objc/message.h>
#include <objc/runtime.h>

#include <atomic>

#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);

namespace e8 {

namespace {

namespace mx = mlx::core;

struct Atomics {
  std::atomic<uint64_t> cb_commandBuffer{0};
  std::atomic<uint64_t> cb_commandBufferWithUnretainedReferences{0};
  std::atomic<uint64_t> cb_commandBufferWithDescriptor{0};
  std::atomic<uint64_t> commit{0};
  std::atomic<uint64_t> waitUntilCompleted{0};
  std::atomic<uint64_t> waitUntilScheduled{0};
  std::atomic<uint64_t> addCompletedHandler{0};
  std::atomic<uint64_t> encodeSignalEvent{0};
  std::atomic<uint64_t> computeCommandEncoder{0};
  std::atomic<uint64_t> endEncoding{0};
  std::atomic<uint64_t> dispatchThreads{0};
  std::atomic<uint64_t> dispatchThreadgroups{0};
  std::atomic<uint64_t> sharedEventWait{0};
};

Atomics g;
bool g_installed = false;

struct Size3 {
  unsigned long width, height, depth;
};

// Original implementations.
IMP o_q_cb, o_q_cbur, o_q_cbd;
IMP o_commit, o_wait_completed, o_wait_scheduled, o_add_handler, o_signal;
IMP o_cce, o_cce_dt, o_cce_desc;
IMP o_end, o_dispatch_threads, o_dispatch_groups;
IMP o_event_wait;

// --- command queue ---------------------------------------------------------
id h_q_cb(id self, SEL cmd) {
  g.cb_commandBuffer++;
  return ((id(*)(id, SEL))o_q_cb)(self, cmd);
}
id h_q_cbur(id self, SEL cmd) {
  g.cb_commandBufferWithUnretainedReferences++;
  return ((id(*)(id, SEL))o_q_cbur)(self, cmd);
}
id h_q_cbd(id self, SEL cmd, id desc) {
  g.cb_commandBufferWithDescriptor++;
  return ((id(*)(id, SEL, id))o_q_cbd)(self, cmd, desc);
}

// --- command buffer --------------------------------------------------------
void h_commit(id self, SEL cmd) {
  g.commit++;
  ((void (*)(id, SEL))o_commit)(self, cmd);
}
void h_wait_completed(id self, SEL cmd) {
  g.waitUntilCompleted++;
  ((void (*)(id, SEL))o_wait_completed)(self, cmd);
}
void h_wait_scheduled(id self, SEL cmd) {
  g.waitUntilScheduled++;
  ((void (*)(id, SEL))o_wait_scheduled)(self, cmd);
}
void h_add_handler(id self, SEL cmd, id block) {
  g.addCompletedHandler++;
  ((void (*)(id, SEL, id))o_add_handler)(self, cmd, block);
}
void h_signal(id self, SEL cmd, id event, uint64_t value) {
  g.encodeSignalEvent++;
  ((void (*)(id, SEL, id, uint64_t))o_signal)(self, cmd, event, value);
}
id h_cce(id self, SEL cmd) {
  g.computeCommandEncoder++;
  return ((id(*)(id, SEL))o_cce)(self, cmd);
}
id h_cce_dt(id self, SEL cmd, unsigned long dispatch_type) {
  g.computeCommandEncoder++;
  return ((id(*)(id, SEL, unsigned long))o_cce_dt)(self, cmd, dispatch_type);
}
id h_cce_desc(id self, SEL cmd, id desc) {
  g.computeCommandEncoder++;
  return ((id(*)(id, SEL, id))o_cce_desc)(self, cmd, desc);
}

// --- compute encoder -------------------------------------------------------
void h_end(id self, SEL cmd) {
  g.endEncoding++;
  ((void (*)(id, SEL))o_end)(self, cmd);
}
void h_dispatch_threads(id self, SEL cmd, Size3 grid, Size3 group) {
  g.dispatchThreads++;
  ((void (*)(id, SEL, Size3, Size3))o_dispatch_threads)(self, cmd, grid, group);
}
void h_dispatch_groups(id self, SEL cmd, Size3 grid, Size3 group) {
  g.dispatchThreadgroups++;
  ((void (*)(id, SEL, Size3, Size3))o_dispatch_groups)(self, cmd, grid, group);
}

// --- shared event ----------------------------------------------------------
BOOL h_event_wait(id self, SEL cmd, uint64_t value, uint64_t ms) {
  g.sharedEventWait++;
  return ((BOOL(*)(id, SEL, uint64_t, uint64_t))o_event_wait)(
      self, cmd, value, ms);
}

// Replace -[cls sel] with `hook`, remembering the original in *orig.
// Returns false if the class does not respond to the selector.
bool hook(Class cls, const char* sel_name, IMP replacement, IMP* orig) {
  SEL sel = sel_registerName(sel_name);
  Method m = class_getInstanceMethod(cls, sel);
  if (!m) {
    return false;
  }
  *orig = method_getImplementation(m);
  if (!class_addMethod(cls, sel, replacement, method_getTypeEncoding(m))) {
    // Implemented directly on cls: swap the implementation in place.
    method_setImplementation(m, replacement);
  }
  return true;
}

} // namespace

bool metal_counters_installed() {
  return g_installed;
}

std::string install_metal_counters() {
  if (g_installed) {
    return "";
  }
  void* pool = objc_autoreleasePoolPush();
  std::string missing;
  auto& d = mx::metal::device(mx::Device::gpu);
  auto stream = mx::default_stream(mx::Device::gpu);
  MTL::CommandQueue* queue = d.get_queue(stream);

  // Discover the concrete classes with scratch objects on MLX's own queue;
  // the scratch command buffer is never committed.
  MTL::CommandBuffer* cb = queue->commandBufferWithUnretainedReferences();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
  enc->endEncoding();
  MTL::SharedEvent* ev = d.mtl_device()->newSharedEvent();

  Class q_cls = object_getClass(reinterpret_cast<id>(queue));
  Class cb_cls = object_getClass(reinterpret_cast<id>(cb));
  Class enc_cls = object_getClass(reinterpret_cast<id>(enc));
  Class ev_cls = object_getClass(reinterpret_cast<id>(ev));

  auto must = [&missing](bool ok, const char* what) {
    if (!ok) {
      missing += std::string(missing.empty() ? "" : ", ") + what;
    }
  };

  must(hook(q_cls, "commandBuffer", (IMP)h_q_cb, &o_q_cb), "commandBuffer");
  must(
      hook(
          q_cls,
          "commandBufferWithUnretainedReferences",
          (IMP)h_q_cbur,
          &o_q_cbur),
      "commandBufferWithUnretainedReferences");
  must(
      hook(q_cls, "commandBufferWithDescriptor:", (IMP)h_q_cbd, &o_q_cbd),
      "commandBufferWithDescriptor:");

  must(hook(cb_cls, "commit", (IMP)h_commit, &o_commit), "commit");
  must(
      hook(cb_cls, "waitUntilCompleted", (IMP)h_wait_completed, &o_wait_completed),
      "waitUntilCompleted");
  must(
      hook(cb_cls, "waitUntilScheduled", (IMP)h_wait_scheduled, &o_wait_scheduled),
      "waitUntilScheduled");
  must(
      hook(cb_cls, "addCompletedHandler:", (IMP)h_add_handler, &o_add_handler),
      "addCompletedHandler:");
  must(
      hook(cb_cls, "encodeSignalEvent:value:", (IMP)h_signal, &o_signal),
      "encodeSignalEvent:value:");
  must(
      hook(cb_cls, "computeCommandEncoder", (IMP)h_cce, &o_cce),
      "computeCommandEncoder");
  must(
      hook(
          cb_cls,
          "computeCommandEncoderWithDispatchType:",
          (IMP)h_cce_dt,
          &o_cce_dt),
      "computeCommandEncoderWithDispatchType:");
  must(
      hook(
          cb_cls,
          "computeCommandEncoderWithDescriptor:",
          (IMP)h_cce_desc,
          &o_cce_desc),
      "computeCommandEncoderWithDescriptor:");

  must(hook(enc_cls, "endEncoding", (IMP)h_end, &o_end), "endEncoding");
  must(
      hook(
          enc_cls,
          "dispatchThreads:threadsPerThreadgroup:",
          (IMP)h_dispatch_threads,
          &o_dispatch_threads),
      "dispatchThreads:threadsPerThreadgroup:");
  must(
      hook(
          enc_cls,
          "dispatchThreadgroups:threadsPerThreadgroup:",
          (IMP)h_dispatch_groups,
          &o_dispatch_groups),
      "dispatchThreadgroups:threadsPerThreadgroup:");

  must(
      hook(
          ev_cls,
          "waitUntilSignaledValue:timeoutMS:",
          (IMP)h_event_wait,
          &o_event_wait),
      "waitUntilSignaledValue:timeoutMS:");

  ev->release();
  objc_autoreleasePoolPop(pool);
  g_installed = true;
  return missing.empty() ? "" : "selectors not hooked: " + missing;
}

MetalCounters metal_counters(bool reset) {
  MetalCounters c;
#define E8_TAKE(field)                                   \
  c.field = reset ? g.field.exchange(0) : g.field.load();
  E8_TAKE(cb_commandBuffer)
  E8_TAKE(cb_commandBufferWithUnretainedReferences)
  E8_TAKE(cb_commandBufferWithDescriptor)
  E8_TAKE(commit)
  E8_TAKE(waitUntilCompleted)
  E8_TAKE(waitUntilScheduled)
  E8_TAKE(addCompletedHandler)
  E8_TAKE(encodeSignalEvent)
  E8_TAKE(computeCommandEncoder)
  E8_TAKE(endEncoding)
  E8_TAKE(dispatchThreads)
  E8_TAKE(dispatchThreadgroups)
  E8_TAKE(sharedEventWait)
#undef E8_TAKE
  return c;
}

} // namespace e8
