#pragma once
// FFT planner: turns a transform request into the kernel passes that compute
// it. Metal-free, so the plans can be built and checked without a device; the
// GPU engine (gpu_engine.cpp) only encodes what a plan says.
//
// Coverage (transform length n along one axis):
//   n <= 4096, factors in {2,3,5,7,11,13}   one Stockham kernel (MLX's plan)
//   n <= 2048, one prime factor p > 13      Rader inside that kernel (p - 1 must factor as above)
//   other n <= 2048                         fused Bluestein kernel
//   4096 < n <= 2^20, power of two          MLX's two-pass four-step plan
//   larger powers of two                    our multi-pass plan (MLX's own is wrong from 2^21)
//   other n > 4096 with factors as above    our multi-pass plan
//   anything else                           unsupported: the caller declines the transform
// A transform along an axis that is not the last (tail > 1) is always a
// multi-pass plan and so needs factors in {2,3,5,7,11,13}.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sm::detail::fft {
enum class Transform : uint8_t { c2c = 0, r2c = 1, c2r = 2 };

// What the device allows one threadgroup. Shader validation doubles static
// threadgroup allocations, so validated runs plan with half the elements.
// The thread count is an upper bound only: Metal grants a pipeline fewer
// threads per threadgroup the more per-thread state its specialisation keeps
// (384 to 1024 on an M1 for these kernels), which is known only once the
// pipeline is built. The engine finds a plan that asks for too many and the
// planner replaces it (replan).
struct Limits {
    int32_t group_elements = 4096;  // float2 slots: 32 KiB of threadgroup memory
    int32_t group_threads = 1024;
};

struct Request {
    Transform transform = Transform::c2c;
    bool inverse = false;  // c2c only: r2c is always forward, c2r always inverse
    uint64_t n = 0;        // transform length (the real side's length for r2c and c2r)
    uint64_t tail = 1;     // elements between consecutive points of one transform; 1 on the last axis
    uint64_t count = 1;    // independent blocks of n * tail elements (rows, for the last axis)
};

// The function constants every FFT kernel is specialised with (indices 0..21
// of mlx/backend/metal/kernels/fft.h).
struct Constants {
    bool inverse = false;          // 0: conjugate on load, conjugate and divide by n on store
    bool power_of_two = true;      // 1
    int32_t elems_per_thread = 0;  // 2
    int32_t rader_m = 0;           // 3: n / rader_n
    int32_t stockham[9] = {};      // 4..12: steps of radix 13, 11, 8, 7, 6, 5, 4, 3, 2
    int32_t rader[9] = {};         // 13..21: the same for the Rader sub-transform
    // Identifies the values uniquely: pipelines are cached by kernel name and key.
    std::string key() const;
};

enum class Kernel : uint8_t {
    stockham,   // MLX fft: one whole transform per threadgroup slice
    rader,      // MLX rader_fft
    bluestein,  // MLX bluestein_fft
    four_step,  // MLX four_step_fft, first or last pass of a power-of-two plan
    strided,    // sm_fft_strided (fft_kernels.h)
};

// One kernel dispatch. Buffers bind at 0 (source) and 1 (target), then the
// tables, then either `values` one int per index (MLX kernels) or one
// MSFFTStridedParams.
struct Step {
    Kernel kind = Kernel::stockham;
    std::string kernel;            // host name in the metallib
    Constants constants;
    uint8_t source = 0;            // buffer slots
    uint8_t target = 1;
    int32_t width = 1;             // transforms per threadgroup (grid y)
    int32_t threads = 1;           // threads per transform (grid z)
    int32_t group_elements = 0;    // float2 of threadgroup memory the kernel instance declares
    uint64_t transforms = 1;       // transforms this pass runs per block of the request
    bool paired = false;           // a real MLX kernel packs two rows into one complex transform
    std::vector<int32_t> tables;   // indices into Plan::tables
    std::vector<int32_t> values;   // MLX kernels: int arguments; values[batch_value] is filled per call
    int32_t batch_value = -1;
    // Kernel::strided: the MSFFTStridedParams fields that do not depend on the batch.
    int32_t n = 0;
    int32_t stride = 0;
    int32_t outer = 1;
    int32_t tail = 1;
    uint32_t flags = 0;
};

// How the host fills slot 0 from the caller's rows, and reads the result slot.
enum class Load : uint8_t {
    direct,         // the caller's leading points, zero padded to the row
    // A one-sided spectrum X[0..n/2] laid out for a complex inverse whose real
    // part is the real inverse: X[0], 2 X[k] for 0 < k < n/2, X[n/2], zeros.
    half_spectrum,
};
enum class Store : uint8_t {
    direct,     // the result slot is the result
    half,       // the first n/2 + 1 points of each row (r2c through a complex plan)
    real_part,  // the real part of each point (c2r through a complex plan)
};

struct Plan {
    uint64_t id = 0;
    Request request;
    std::vector<Step> steps;
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> tables;  // Rader and Bluestein constants
    // float32 scalars each buffer slot holds per block of the request. Slot 0
    // is the input the host loads; a slot can be larger than its first use.
    std::vector<uint64_t> slots;
    // Extra float32 tail scalars required past slots[0] * count for vendored
    // kernels that read padded indices before checking bounds (Bluestein c2r).
    uint64_t input_tail_scalars = 0;
    uint8_t result_slot = 1;
    bool input_real = false;    // slot 0 holds float rows, not float2
    bool result_real = false;   // the result slot holds float rows
    uint64_t input_row = 0;     // elements per row the host loads into slot 0
    uint64_t result_row = 0;    // elements per row in the result slot
    Load load = Load::direct;
    Store store = Store::direct;
    // The fused Bluestein c2r kernel keeps imag(x[0]), which SciPy ignores:
    // the host clears it in its copy of the input.
    bool clear_dc_imag = false;
    // Four-step and strided c2r plans do not clear imag(X[0]) or imag(X[n/2])
    // inside the kernel: cleared on host for 1-D or on GPU after inner axes.
    bool clear_hermitian_imag = false;
    // An inverse built on MLX kernels comes out divided by n. A plan whose last
    // step is ours applies any factor there instead (takes_scale).
    bool divides_by_n = false;
    bool takes_scale = false;
    std::string name;           // for diagnostics, e.g. "four-step 1024x64"
};

struct Planned {
    std::shared_ptr<const Plan> plan;  // null when the size has no correct plan
    std::string reason;                // why not
};

// The plan for a request, from the cache when one was built before. The
// request's count only selects between plans tuned for small and large
// batches; a plan runs for any count (see instantiate).
Planned plan(const Request& request, const Limits& limits = {});
// Builds without consulting the cache.
Planned build_plan(const Request& request, const Limits& limits = {});
// Replaces the cached plan for (request, limits) with one whose threadgroups
// have at most `group_threads` threads, after the device refused the cached
// one: fewer transforms per threadgroup, Bluestein instead of Rader, a chain
// of strided passes instead of one Stockham kernel - or no plan at all, which
// is then what the cache answers.
Planned replan(const Request& request, const Limits& limits, int32_t group_threads);
// A multi-pass plan with the given pass lengths (their product must be n),
// for tests and tuning: every factorisation the planner can choose must
// compute the same transform.
Planned build_multipass(const Request& request, const std::vector<int32_t>& factors, const Limits& limits = {});

// A step made concrete for `count` blocks.
struct Dispatch {
    uint64_t groups = 0;          // threadgroups along x; (1, width, threads) threads each
    std::vector<int32_t> values;  // MLX kernels
    int32_t blocks = 0;           // strided: MSFFTStridedParams::blocks
};
// False when the batch exceeds the kernels' 32-bit indexing.
bool instantiate(const Plan& plan, uint64_t count, std::vector<Dispatch>& out, std::string& reason);

// Number of plans built since the process started (cache misses), for tests.
uint64_t plans_built();

// Smallest transform length >= target that the GPU planner executes efficiently
// with direct loads/stores: {2, 3, 5}-smooth up to limits.group_elements (and
// std::bit_ceil above limits.group_elements when real && last_axis so
// plan_power_of_two uses Load::direct / Store::direct).
uint64_t next_fast_len(uint64_t target, bool real = false, bool last_axis = true,
                       const Limits& limits = {});

// The pieces of MLX's planning the tests check directly.
namespace mlx {
std::vector<int32_t> prime_factors(int32_t n);
// True when n factors into the supported radices.
bool smooth(uint64_t n);
}
} // namespace sm::detail::fft
