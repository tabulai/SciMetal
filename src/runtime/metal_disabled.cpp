// Backend stubs for builds without Metal (SM_ENABLE_METAL=OFF): CPU storage and
// CPU routes work; anything that needs a device reports SM_ENOTSUP.
#include "sm/detail/backend.hpp"

namespace sm::detail {
RawAllocation allocate_metal(uint64_t) {
    throw Error(SM_ENOTSUP, "Metal storage is disabled in this build");
}
void prepare_cached_metal(RawAllocation&) {
    throw Error(SM_ENOTSUP, "Metal storage is disabled in this build");
}
void discard_cached_metal(RawAllocation&) {
    throw Error(SM_ENOTSUP, "Metal storage is disabled in this build");
}
RawAllocation wrap_metal_pages(const void*, uint64_t, std::shared_ptr<void>) {
    throw Error(SM_ENOTSUP, "Metal no-copy storage is disabled in this build");
}
bool metal_event_complete(void* event, uint64_t value) noexcept {
    return !event || !value;
}
uint64_t wait_metal_event(void*, uint64_t) {
    throw Error(SM_ENOTSUP, "Metal storage is disabled in this build");
}
void verify_metal_import(const sm_buffer_view&) {
    throw Error(SM_ENOTSUP, "Metal storage is disabled in this build");
}
} // namespace sm::detail
