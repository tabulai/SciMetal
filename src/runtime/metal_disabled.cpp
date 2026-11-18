// Backend stubs for builds without Metal (MS_ENABLE_METAL=OFF): CPU storage and
// CPU routes work; anything that needs a device reports MS_ENOTSUP.
#include "ms/detail/backend.hpp"

namespace ms::detail {
RawAllocation allocate_metal(uint64_t) {
    throw Error(MS_ENOTSUP, "Metal storage is disabled in this build");
}
void prepare_cached_metal(RawAllocation&) {
    throw Error(MS_ENOTSUP, "Metal storage is disabled in this build");
}
void discard_cached_metal(RawAllocation&) {
    throw Error(MS_ENOTSUP, "Metal storage is disabled in this build");
}
RawAllocation wrap_metal_pages(const void*, uint64_t, std::shared_ptr<void>) {
    throw Error(MS_ENOTSUP, "Metal no-copy storage is disabled in this build");
}
bool metal_event_complete(void* event, uint64_t value) noexcept {
    return !event || !value;
}
uint64_t wait_metal_event(void*, uint64_t) {
    throw Error(MS_ENOTSUP, "Metal storage is disabled in this build");
}
void verify_metal_import(const ms_buffer_view&) {
    throw Error(MS_ENOTSUP, "Metal storage is disabled in this build");
}
} // namespace ms::detail
