// scipy.fft-compatible transforms behind the C ABI (ms_fft.h): request
// checking, routing, the per-axis plans of one call, and the host side of the
// GPU route (loading rows into a pooled single-precision buffer, storing the
// result in the precision SciPy returns).
#include "ms_fft.h"
#include "core/array.hpp"
#include "core/router.hpp"
#include "ms/detail/parallel.hpp"
#include "plan.hpp"
#if defined(MS_ENABLE_METAL)
#include "engine.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace ms::detail::fft {
namespace {
// Routing. MS_EXECUTION_AUTO takes the GPU when it is predicted to beat the
// caller's own FFT (pocketfft, as SciPy runs it) by a margin. Both sides are
// modelled in nanoseconds, calibrated on applegpu_g13d (M1 Ultra) with
// bench/fft.py --crossover:
//
//   host    points x host_ns_per_point(length) per transformed axis, over the
//           threads the caller would use. pocketfft costs about 1.2 ns per
//           point for short rows, 0.16 ns x log2(n) up to a few thousand
//           points and 0.25 ns x log2(n) once a transform falls out of the
//           CPU caches; double precision costs 1.8 times as much and real
//           transforms less. Its threads split a batch between them; within
//           one transform they help only for a 1-D array, by about
//           log2(n) - 15.
//   device  a fixed 0.3 ms per call (command buffer round trip, hand-off)
//           plus 0.35 ns per point, or 1 ns when the input is narrowed and
//           the result widened.
//
// The router compares one number with one threshold, so the per-point device
// cost is taken off the host estimate: the work is the time the GPU route
// would save before its fixed cost, and the family threshold is that fixed
// cost. A long transform is expensive per point on the host and so crosses
// over with far fewer points (2^17 single precision, 2^16 double) than a
// batch of short rows (some 250 000 to 500 000 points).
constexpr double device_fixed_ns = 300e3;
constexpr double device_ns_per_point = 0.35;
constexpr double device_ns_per_point_double = 1.0;
constexpr double host_double_factor = 1.8;
constexpr double host_r2c_factor = 0.55;  // rfft against fft of the same length
constexpr double host_c2r_factor = 0.75;  // irfft, and fft of a real input
// The GPU must be predicted this much faster before a call leaves SciPy: at
// break-even nothing is gained for the single-precision arithmetic.
constexpr double required_advantage = 1.25;
const router::Family family{"fft", static_cast<uint64_t>(required_advantage * device_fixed_ns)};

double host_ns_per_point(uint64_t length) {
    const double log2n = std::log2(static_cast<double>(std::max<uint64_t>(length, 2)));
    return length < (uint64_t{1} << 15) ? std::max(1.2, 0.16 * log2n) : 0.25 * log2n;
}

struct Call {
    Transform transform = Transform::c2c;
    bool inverse = false;            // the transform's direction: MS_FFT_INVERSE and MS_FFT_C2R
    array::Shape in_shape;
    uint32_t in_dtype = 0;
    array::Shape out_shape;
    uint32_t out_dtype = 0;
    bool last_axis = false;          // the last axis is transformed
    std::vector<uint32_t> inner;     // the other transformed axes, ascending
    uint64_t rows = 0;               // product of every extent but the last
    uint64_t have = 0;               // points per row in the input
    uint64_t need = 0;               // points per row the transform reads (zero padded beyond `have`)
    uint64_t n = 0;                  // transform length along the last axis; `have` when it is not transformed
    double scale = 1.0;              // SciPy's normalisation for the whole call
    uint64_t work = 0;               // router work (see above)
};

Call parse(uint32_t dtype, const array::Shape& shape, const ms_fft_options* options) {
    if (!options) throw Error(MS_EINVAL, "null FFT options");
    if (options->struct_size != sizeof(ms_fft_options) || options->abi_version != MS_FFT_ABI_VERSION)
        throw Error(MS_EINVAL, "FFT options size or ABI version mismatch");
    if (options->reserved[0] || options->reserved[1]) throw Error(MS_EINVAL, "FFT options reserved fields must be zero");
    if (options->norm > MS_FFT_NORM_FORWARD) throw Error(MS_EINVAL, "unknown FFT normalisation");
    if (options->axis_count < 1 || options->axis_count > shape.ndim)
        throw Error(MS_EINVAL, "an FFT transforms between one and all of the input's axes");

    Call call;
    switch (options->kind) {
    case MS_FFT_FORWARD: call.transform = Transform::c2c; break;
    case MS_FFT_INVERSE: call.transform = Transform::c2c; call.inverse = true; break;
    case MS_FFT_R2C: call.transform = Transform::r2c; break;
    case MS_FFT_C2R: call.transform = Transform::c2r; call.inverse = true; break;
    default: throw Error(MS_EINVAL, "unknown FFT kind");
    }
    const uint32_t last = shape.ndim - 1;
    uint32_t seen = 0;
    for (uint32_t i = 0; i < options->axis_count; ++i) {
        const uint32_t axis = options->axes[i];
        if (axis >= shape.ndim) throw Error(MS_EINVAL, "FFT axis exceeds the input's rank");
        if (seen & (1u << axis)) throw Error(MS_EINVAL, "FFT axes must be distinct");
        seen |= 1u << axis;
    }
    call.last_axis = (seen & (1u << last)) != 0;
    for (uint32_t axis = 0; axis < last; ++axis) {
        if (seen & (1u << axis)) call.inner.push_back(axis);
    }
    const bool real = call.transform != Transform::c2c;
    if (real && !call.last_axis) throw Error(MS_EINVAL, "a real FFT transforms the input's last axis");
    if (call.transform == Transform::r2c && array::is_complex(dtype))
        throw Error(MS_EINVAL, "a real-to-complex FFT needs a real input");
    if (options->n && !call.last_axis)
        throw Error(MS_EINVAL, "an FFT length applies to the last axis, which is not transformed");

    // Empty inputs and lengths are SciPy's to reject.
    if (shape.elements == 0) throw Error(MS_ENOTSUP, "empty FFT input");
    call.in_shape = shape;
    call.in_dtype = dtype;
    call.have = shape.dims[last];
    call.rows = shape.elements / call.have;
    call.n = options->n ? options->n : (call.transform == Transform::c2r ? 2 * (call.have - 1) : call.have);
    if (call.n == 0) throw Error(MS_ENOTSUP, "empty FFT length");
    if (call.n > INT32_MAX) throw Error(MS_ENOTSUP, "the FFT length exceeds the kernels' 32-bit indexing");
    call.need = call.transform == Transform::c2r ? call.n / 2 + 1 : call.n;

    call.out_shape = shape;
    const uint64_t out_row = call.transform == Transform::r2c ? call.n / 2 + 1 : call.n;
    call.out_shape.dims[last] = out_row;
    if (ms_checked_mul(call.rows, out_row, &call.out_shape.elements) != MS_OK)
        throw Error(MS_ENOTSUP, "the FFT result exceeds the kernels' 32-bit indexing");
    const bool is_double = array::is_double(dtype);
    call.out_dtype = call.transform == Transform::c2r ? (is_double ? MS_FLOAT64 : MS_FLOAT32)
                                                      : (is_double ? MS_COMPLEX128 : MS_COMPLEX64);

    // Normalisation and router work over the transformed axes.
    const double points = static_cast<double>(call.rows) * static_cast<double>(call.n);
    const double threads = static_cast<double>(std::max<uint32_t>(options->host_threads, 1));
    double total = 1.0, host_ns = 0.0;
    const auto add_axis = [&](uint64_t length, double factor) {
        total *= static_cast<double>(length);
        const double within = shape.ndim == 1 ? std::log2(static_cast<double>(length)) - 15.0 : 1.0;
        const double parallel = std::max(1.0, std::min(threads, std::max(points / static_cast<double>(length), within)));
        host_ns += points * host_ns_per_point(length) * factor / parallel;
    };
    const double precision_factor = is_double ? host_double_factor : 1.0;
    if (call.last_axis) {
        const double kind_factor = call.transform == Transform::r2c ? host_r2c_factor
                                   : call.transform == Transform::c2r || !array::is_complex(dtype) ? host_c2r_factor : 1.0;
        add_axis(call.n, precision_factor * kind_factor);
    }
    for (const uint32_t axis : call.inner) add_axis(shape.dims[axis], precision_factor);
    switch (options->norm) {
    case MS_FFT_NORM_ORTHO: call.scale = 1.0 / std::sqrt(total); break;
    case MS_FFT_NORM_FORWARD: call.scale = call.inverse ? 1.0 : 1.0 / total; break;
    default: call.scale = call.inverse ? 1.0 / total : 1.0; break;
    }
    const double saved = host_ns - required_advantage * points * (is_double ? device_ns_per_point_double : device_ns_per_point);
    call.work = saved <= 0.0 ? 0 : saved >= 1.8e19 ? UINT64_MAX : static_cast<uint64_t>(saved);
    return call;
}

array::Shape checked_shape(uint32_t dtype, uint32_t ndim, const uint64_t* shape) {
    (void)array::dtype_size(dtype);
    if (ndim < 1 || ndim > MS_MAX_NDIM || !shape) throw Error(MS_EINVAL, "array rank must be 1 to 8");
    array::Shape result;
    result.ndim = ndim;
    result.elements = 1;
    for (uint32_t axis = 0; axis < ndim; ++axis) {
        result.dims[axis] = shape[axis];
        if (ms_checked_mul(result.elements, shape[axis], &result.elements) != MS_OK)
            throw Error(MS_EOVERFLOW, "array element count overflows");
    }
    return result;
}

#if defined(MS_ENABLE_METAL)
// The plans of one call, in execution order, and what the host does around them.
struct Staged {
    std::vector<Stage> stages;
    bool device_real = false;     // slot 0 of the first stage holds float rows
    uint64_t device_row = 0;      // points per row the host loads
    Load load = Load::direct;
    bool clear_dc_imag = false;
    uint64_t result_row = 0;      // points per row in the last stage's result
    Store store = Store::direct;
    double applied = 1.0;         // factor the kernels apply by themselves
    std::string description;
};

Staged stage(const Call& call, const Limits& limits, bool with_description = true) {
    Staged staged;
    std::shared_ptr<const Plan> last;
    if (call.last_axis) {
        Request request;
        request.transform = call.transform;
        request.inverse = call.inverse;
        request.n = call.n;
        request.count = call.rows;
        Planned planned = plan(request, limits);
        if (!planned.plan) throw Error(MS_ENOTSUP, "no GPU plan for an FFT of length " + std::to_string(call.n) + ": " + planned.reason);
        last = std::move(planned.plan);
        if (last->clear_dc_imag && !call.inner.empty())
            throw Error(MS_ENOTSUP, "the Bluestein inverse real FFT is not available together with other axes");
    }
    // Row length of the device array while the inner axes are transformed:
    // after the last axis for a forward transform, before it for an inverse.
    const uint64_t before = last ? last->input_row : call.have;
    const uint64_t after = last ? last->result_row : call.have;
    const uint64_t inner_row = call.inverse ? before : after;

    std::vector<Stage> inner;
    std::vector<std::string> inner_names;
    for (auto axis = call.inner.rbegin(); axis != call.inner.rend(); ++axis) {
        Request request;
        request.transform = Transform::c2c;
        request.inverse = call.inverse;
        request.n = call.in_shape.dims[*axis];
        request.count = 1;
        request.tail = inner_row;
        for (uint32_t other = 0; other < *axis; ++other) request.count *= call.in_shape.dims[other];
        for (uint32_t other = *axis + 1; other + 1 < call.in_shape.ndim; ++other) {
            if (ms_checked_mul(request.tail, call.in_shape.dims[other], &request.tail) != MS_OK)
                throw Error(MS_ENOTSUP, "the FFT exceeds the kernels' 32-bit indexing");
        }
        if (request.n == 1) continue;  // a one-point transform is the identity
        Planned planned = plan(request, limits);
        if (!planned.plan)
            throw Error(MS_ENOTSUP, "no GPU plan for an FFT of length " + std::to_string(request.n) + " along axis " +
                                        std::to_string(*axis) + ": " + planned.reason);
        if (with_description) inner_names.push_back("axis " + std::to_string(*axis) + ": " + planned.plan->name);
        inner.push_back({std::move(planned.plan), request.count});
    }

    // Forward transforms take the last axis first (a real input becomes complex
    // there); inverses take it last.
    if (last && !call.inverse) staged.stages.push_back({last, call.rows});
    staged.stages.insert(staged.stages.end(), inner.begin(), inner.end());
    if (last && call.inverse) staged.stages.push_back({last, call.rows});
    if (staged.stages.empty()) throw Error(MS_ENOTSUP, "a one-point FFT is a copy");
    if (with_description) {
        std::vector<std::string> names;
        const std::string last_name = last ? "axis " + std::to_string(call.in_shape.ndim - 1) + ": " + last->name : "";
        if (last && !call.inverse) names.push_back(last_name);
        names.insert(names.end(), inner_names.begin(), inner_names.end());
        if (last && call.inverse) names.push_back(last_name);
        for (const std::string& name : names) staged.description += (staged.description.empty() ? "" : "; ") + name;
    }

    staged.device_real = last && last->input_real;
    staged.device_row = before;
    staged.result_row = after;
    if (last) {
        staged.load = last->load;
        staged.store = last->store;
        staged.clear_dc_imag = last->clear_dc_imag;
    }
    for (const Stage& each : staged.stages) {
        if (each.plan->divides_by_n) staged.applied /= static_cast<double>(each.plan->request.n);
    }
    return staged;
}

// Runs f(lane, row, begin, end) over the points [begin, end) of every row,
// on the performance cores: whole rows per lane for a batch, and the points
// of each row split between lanes when the rows are few and long.
constexpr uint64_t lane_points = uint64_t{1} << 17;
template<class Function> void for_row_spans(uint64_t rows, uint64_t row_points, Function&& f) {
    if (rows == 0 || row_points == 0) return;
    if (rows >= parallel::hardware_lanes() || row_points <= lane_points) {
        const uint64_t rows_per_lane = std::max<uint64_t>(1, lane_points / row_points);
        parallel::parallel_for(rows, rows_per_lane, [&](unsigned lane, uint64_t begin, uint64_t end) {
            for (uint64_t row = begin; row < end; ++row) f(lane, row, uint64_t{0}, row_points);
        });
    } else {
        for (uint64_t row = 0; row < rows; ++row)
            parallel::parallel_for(row_points, lane_points, [&](unsigned lane, uint64_t begin, uint64_t end) { f(lane, row, begin, end); });
    }
}

// The largest magnitude among `count` scalars, as its bit pattern: for
// non-negative IEEE doubles the integer order is the numeric order, with
// infinity and NaN above every finite value, and integer maxima vectorise.
uint64_t magnitude_bits(const double* values, uint64_t count) {
    uint64_t peak = 0;
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t bits;
        std::memcpy(&bits, values + i, sizeof(bits));
        peak = std::max(peak, bits & 0x7FFFFFFFFFFFFFFFull);
    }
    return peak;
}
constexpr uint64_t load_block = 2048;  // points

// Fills the device rows from the caller's: the first `used` points of each
// row converted to single precision (a real point becomes a complex one when
// the device rows are complex), zeros beyond. Load::half_spectrum doubles the
// interior points of a one-sided spectrum of the n-point transform. Returns
// the largest magnitude loaded (bit pattern, double-precision sources only).
template<class Source>
uint64_t load_rows(const Source* __restrict__ source, bool source_complex, float* __restrict__ device,
                   bool device_complex, uint64_t rows, uint64_t have, uint64_t used, uint64_t device_row,
                   Load load, uint64_t n, bool clear_dc_imag) {
    const uint64_t in_step = source_complex ? 2 : 1, out_step = device_complex ? 2 : 1;
    uint64_t peaks[parallel::max_lanes] = {};
    for_row_spans(rows, device_row, [&](unsigned lane, uint64_t row, uint64_t begin, uint64_t end) {
        const Source* __restrict__ in = source + row * have * in_step;
        float* __restrict__ out = device + row * device_row * out_step;
        const uint64_t stop = std::min(end, used);  // points [begin, stop) come from the caller
        if constexpr (std::is_same_v<Source, float>) {
            (void)lane;
            if (begin < stop) {
                if (source_complex == device_complex) {
                    std::memcpy(out + begin * out_step, in + begin * in_step,
                                (stop - begin) * out_step * sizeof(float));
                } else {
                    for (uint64_t i = begin; i < stop; ++i) {
                        out[2 * i] = in[i];
                        out[2 * i + 1] = 0.0f;
                    }
                }
            }
        } else {
            // In blocks that stay in the first-level cache between the magnitude
            // scan and the conversion.
            for (uint64_t from = begin; from < stop; from += load_block) {
                const uint64_t to = std::min(stop, from + load_block);
                peaks[lane] = std::max(peaks[lane], magnitude_bits(in + from * in_step, (to - from) * in_step));
                if (source_complex == device_complex) {
                    for (uint64_t i = from * out_step; i < to * out_step; ++i) out[i] = static_cast<float>(in[i]);
                } else {
                    for (uint64_t i = from; i < to; ++i) {
                        out[2 * i] = static_cast<float>(in[i]);
                        out[2 * i + 1] = 0.0f;
                    }
                }
            }
        }
        if (begin < stop) {
            if (load == Load::half_spectrum) {
                // x = Re(ifft(y)) with y = X[0], 2 X[k] for 0 < k < n/2, X[n/2], 0, ...
                for (uint64_t k = std::max<uint64_t>(begin, 1); k < stop && 2 * k < n; ++k) {
                    out[2 * k] *= 2.0f;
                    out[2 * k + 1] *= 2.0f;
                }
            }
            if (clear_dc_imag && begin == 0) out[1] = 0.0f;
        }
        const uint64_t pad = std::max(begin, used);
        if (pad < end) std::memset(out + pad * out_step, 0, (end - pad) * out_step * sizeof(float));
    });
    return *std::max_element(std::begin(peaks), std::end(peaks));
}

// Writes the caller's rows from the device result: the leading points of each
// device row (their real parts for Store::real_part), scaled, in the caller's
// precision.
template<class Target>
void store_rows(const float* __restrict__ device, uint64_t device_row, Target* __restrict__ target,
                uint64_t target_row, uint64_t rows, bool complex, Store store, double scale) {
    const uint64_t in_step = store == Store::real_part || complex ? 2 : 1;
    const uint64_t out_step = complex ? 2 : 1;
    for_row_spans(rows, target_row, [&](unsigned, uint64_t row, uint64_t begin, uint64_t end) {
        const float* __restrict__ in = device + row * device_row * in_step;
        Target* __restrict__ out = target + row * target_row * out_step;
        if (store == Store::real_part) {
            if (scale == 1.0) {
                for (uint64_t i = begin; i < end; ++i) out[i] = static_cast<Target>(in[2 * i]);
            } else {
                for (uint64_t i = begin; i < end; ++i)
                    out[i] = static_cast<Target>(static_cast<double>(in[2 * i]) * scale);
            }
        } else if (scale == 1.0) {
            if constexpr (std::is_same_v<Target, float>) {
                if (begin < end) {
                    std::memcpy(out + begin * out_step, in + begin * out_step,
                                (end - begin) * out_step * sizeof(float));
                }
            } else {
                for (uint64_t i = begin * out_step; i < end * out_step; ++i) out[i] = static_cast<double>(in[i]);
            }
        } else {
            for (uint64_t i = begin * out_step; i < end * out_step; ++i)
                out[i] = static_cast<Target>(static_cast<double>(in[i]) * scale);
        }
    });
}

// Magnitudes a double-precision input may have for single precision to carry
// its transform: sums over up to 2^31 points must stay finite (float32 ends at
// 2^128), and what underflows (below 2^-149) must stay negligible against the
// largest value. Anything else, infinities and NaNs included, is SciPy's.
constexpr double narrow_ceiling = 0x1p96;
constexpr double narrow_floor = 0x1p-100;

BufferPtr load(ms_context* context, const ms_array& input, const Call& call, const Staged& staged,
               Profile* profile = nullptr) {
    using Clock = std::chrono::steady_clock;
    const auto elapsed_ns = [](Clock::time_point start, Clock::time_point end) -> uint64_t {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    };
    const Stage& first = staged.stages.front();
    const uint64_t bytes = first.plan->slots[0] * first.count * sizeof(float);
    Clock::time_point t0;
    if (profile) t0 = Clock::now();
    BufferPtr buffer = array::allocate(context, bytes);
    Clock::time_point t1;
    if (profile) {
        t1 = Clock::now();
        profile->alloc_in_ns = elapsed_ns(t0, t1);
        profile->narrowed = array::is_double(call.in_dtype);
    }
    auto* device = reinterpret_cast<float*>(mutable_data(buffer.get()));
    const bool source_complex = array::is_complex(call.in_dtype);
    const bool device_complex = !staged.device_real;
    const uint64_t used = std::min(call.have, call.need);
    if (call.rows * staged.device_row * (device_complex ? 2 : 1) * sizeof(float) > bytes)
        throw Error(MS_EBACKEND, "FFT input slot is smaller than the rows it holds");
    // Slack beyond the rows (a real input slot sized for the spectrum that
    // replaces it) is written by the last pass before anything reads it.
    if (!array::is_double(call.in_dtype) && source_complex == device_complex && call.have == staged.device_row &&
        used == staged.device_row && staged.load == Load::direct && !staged.clear_dc_imag) {
        array::convert(input.data, call.in_dtype, device, call.in_dtype, call.rows * call.have);
    } else if (array::is_double(call.in_dtype)) {
        const uint64_t peak_bits = load_rows(static_cast<const double*>(input.data), source_complex, device, device_complex,
                                             call.rows, call.have, used, staged.device_row, staged.load, call.n,
                                             staged.clear_dc_imag);
        double peak;
        std::memcpy(&peak, &peak_bits, sizeof(peak));
        if (!(peak <= narrow_ceiling) || (peak != 0.0 && peak < narrow_floor)) {
            if (profile) profile->copy_in_ns = elapsed_ns(t1, Clock::now());
            throw Error(MS_ENOTSUP, "the input's magnitudes do not fit single precision; SciPy's FFT is used");
        }
    } else {
        load_rows(static_cast<const float*>(input.data), source_complex, device, device_complex, call.rows, call.have,
                  used, staged.device_row, staged.load, call.n, staged.clear_dc_imag);
    }
    if (profile) profile->copy_in_ns = elapsed_ns(t1, Clock::now());
    return buffer;
}

// The plans of a call that this device's pipelines can run: stages the call
// and replaces any plan whose threadgroups its pipeline does not hold, until
// all fit. MS_ENOTSUP when a length is left without a plan. Builds the
// pipelines, so it is also what preparing a transform ahead of time means.
Staged fit(const Call& call, const Limits& limits, Profile* profile = nullptr, bool with_description = true) {
    using Clock = std::chrono::steady_clock;
    const auto elapsed_ns = [](Clock::time_point start, Clock::time_point end) -> uint64_t {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    };
    Clock::time_point t0;
    if (profile) t0 = Clock::now();
    Staged staged = stage(call, limits, with_description);
    if (profile) profile->stage_ns += elapsed_ns(t0, Clock::now());
    int32_t ceiling = limits.group_threads;
    for (Refusal refusal;;) {
        Clock::time_point tr;
        if (profile) tr = Clock::now();
        const bool is_refused = refused(staged.stages, refusal);
        if (profile) profile->refused_ns += elapsed_ns(tr, Clock::now());
        if (!is_refused) break;
        // Each replacement was built within the ceiling and refused below it,
        // so the ceiling falls every round.
        Clock::time_point tp;
        if (profile) tp = Clock::now();
        ceiling = std::min(ceiling, refusal.granted);
        const Stage& stage_refused = staged.stages[refusal.stage];
        Request request = stage_refused.plan->request;
        request.count = stage_refused.count;
        replan(request, limits, ceiling);
        if (profile) {
            profile->replan_ns += elapsed_ns(tp, Clock::now());
            ++profile->replan_iters;
        }
        Clock::time_point ts;
        if (profile) ts = Clock::now();
        staged = stage(call, limits, with_description);
        if (profile) profile->stage_ns += elapsed_ns(ts, Clock::now());
    }
    return staged;
}

BufferPtr transform(ms_context* context, const ms_array& input, const Call& call, const Limits& limits,
                    Profile* profile = nullptr) {
    using Clock = std::chrono::steady_clock;
    const auto elapsed_ns = [](Clock::time_point start, Clock::time_point end) -> uint64_t {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    };
    const Staged staged = fit(call, limits, profile, false);
    const uint64_t out_row = call.out_shape.dims[call.out_shape.ndim - 1];
    const bool out_complex = array::is_complex(call.out_dtype);
    const uint64_t out_bytes = call.out_shape.elements * array::dtype_size(call.out_dtype);
    const double remaining = call.scale / staged.applied;

    // A single-precision result that needs no cropping is the device buffer
    // itself, scaled on the device. Everything else is written by the host in
    // one pass over the performance cores that crops, scales and widens.
    // (Widening on the device in a last pass was measured and is not faster:
    // 2.1 ms against 1.9 ms for 2^24 complex points on an M1 Ultra.)
    const bool publish_device = !array::is_double(call.out_dtype) && staged.store == Store::direct;
    BufferPtr result = run(context, staged.stages, load(context, input, call, staged, profile),
                           publish_device ? static_cast<float>(remaining) : 1.0f, profile);
    if (publish_device) {
        if (buffer_size(result.get(), context) != out_bytes)
            throw Error(MS_EBACKEND, "FFT result buffer does not match the result's size");
        return result;
    }
    Clock::time_point tw0;
    if (profile) tw0 = Clock::now();
    const auto* source = reinterpret_cast<const float*>(read(result.get(), context).data);
    BufferPtr output = array::allocate(context, out_bytes);
    uint8_t* target = mutable_data(output.get());
    Clock::time_point tw1;
    if (profile) {
        tw1 = Clock::now();
        profile->widen_alloc_ns = elapsed_ns(tw0, tw1);
    }
    if (array::is_double(call.out_dtype)) {
        store_rows(source, staged.result_row, reinterpret_cast<double*>(target), out_row, call.rows, out_complex,
                   staged.store, remaining);
    } else {
        store_rows(source, staged.result_row, reinterpret_cast<float*>(target), out_row, call.rows, out_complex,
                   staged.store, remaining);
    }
    if (profile) profile->widen_store_ns = elapsed_ns(tw1, Clock::now());
    return output;
}

// Bytes the run holds at once: every device buffer plus a separate host result.
uint64_t footprint(const Call& call, const Staged& staged) {
    std::vector<uint64_t> idle;
    uint64_t total = staged.stages.front().plan->slots[0] * staged.stages.front().count * sizeof(float);
    for (const Stage& each : staged.stages) {
        std::vector<uint64_t> held;
        for (size_t slot = 1; slot < each.plan->slots.size(); ++slot) {
            const uint64_t bytes = each.plan->slots[slot] * each.count * sizeof(float);
            const auto reuse = std::find(idle.begin(), idle.end(), bytes);
            if (reuse != idle.end()) idle.erase(reuse);
            else total += bytes;
            held.push_back(bytes);
        }
        // Everything but the carried buffer is free for the next stage; sizes
        // are all the model needs.
        idle.insert(idle.end(), held.begin(), held.end());
    }
    if (array::is_double(call.out_dtype) || staged.store != Store::direct)
        total += call.out_shape.elements * array::dtype_size(call.out_dtype);
    return total;
}
#endif

// Routes a checked call: the decision, or MS_ENOTSUP when it stays on the CPU.
router::Decision route(ms_context* context, const Call& call, const ms_route_options& options) {
    const router::Decision decision = router::decide(context, family, options, call.in_dtype, call.work);
    if (decision.engine != MS_EXECUTION_GPU) {
        if (options.execution == MS_EXECUTION_CPU)
            throw Error(MS_ENOTSUP, "there is no native CPU FFT; SciPy's is used");
        if (array::is_double(call.in_dtype) && options.precision == MS_PRECISION_STRICT)
            throw Error(MS_ENOTSUP, "strict precision keeps double-precision FFTs on the CPU, where SciPy's is used");
        throw Error(MS_ENOTSUP, "the FFT is below the GPU crossover; SciPy's is used");
    }
    return decision;
}
} // namespace
} // namespace ms::detail::fft

using ms::detail::Error;
namespace array = ms::detail::array;
namespace fft = ms::detail::fft;

#if defined(MS_ENABLE_METAL)
struct ProfileGuard {
    using Clock = std::chrono::steady_clock;
    bool active = false;
    fft::Profile data;
    Clock::time_point entry;
    const ms_array* input = nullptr;
    const ms_fft_options* options = nullptr;
    ms_status status = MS_OK;

    ProfileGuard(const ms_array* in, const ms_fft_options* opts) noexcept
        : active(fft::profile_enabled()), input(in), options(opts) {
        if (active) entry = Clock::now();
    }
    ~ProfileGuard() noexcept {
        if (!active) return;
        data.total_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - entry).count());

        char shape_buf[128] = "null";
        const char* dtype_str = "?";
        if (input) {
            switch (input->dtype) {
            case MS_FLOAT32: dtype_str = "f32"; break;
            case MS_FLOAT64: dtype_str = "f64"; break;
            case MS_COMPLEX64: dtype_str = "c64"; break;
            case MS_COMPLEX128: dtype_str = "c128"; break;
            default: dtype_str = "?"; break;
            }
            if (input->ndim >= 1 && input->ndim <= MS_MAX_NDIM) {
                size_t pos = 0;
                for (uint32_t i = 0; i < input->ndim && pos + 1 < sizeof(shape_buf); ++i) {
                    const int w = std::snprintf(shape_buf + pos, sizeof(shape_buf) - pos, "%s%llu",
                                                i == 0 ? "" : "x", static_cast<unsigned long long>(input->shape[i]));
                    if (w <= 0) break;
                    pos += static_cast<size_t>(w) < sizeof(shape_buf) - pos ? static_cast<size_t>(w)
                                                                            : sizeof(shape_buf) - pos - 1;
                }
            } else {
                std::snprintf(shape_buf, sizeof(shape_buf), "ndim=%u", input->ndim);
            }
        }
        const char* kind_str = "?";
        if (options) {
            switch (options->kind) {
            case MS_FFT_FORWARD: kind_str = "fwd"; break;
            case MS_FFT_INVERSE: kind_str = "inv"; break;
            case MS_FFT_R2C: kind_str = "r2c"; break;
            case MS_FFT_C2R: kind_str = "c2r"; break;
            default: kind_str = "?"; break;
            }
        }
        const char* status_str = "?";
        switch (status) {
        case MS_OK: status_str = "OK"; break;
        case MS_EINVAL: status_str = "EINVAL"; break;
        case MS_ENOTSUP: status_str = "ENOTSUP"; break;
        case MS_EOVERFLOW: status_str = "EOVERFLOW"; break;
        case MS_EOOM: status_str = "EOOM"; break;
        case MS_ETIMEOUT: status_str = "ETIMEOUT"; break;
        case MS_EBACKEND: status_str = "EBACKEND"; break;
        case MS_ESTATE: status_str = "ESTATE"; break;
        }

        char buf[768];
        const int len = std::snprintf(
            buf, sizeof(buf),
            "ms_fft_profile shape=%s dtype=%s kind=%s status=%s narrowed=%d replan_iters=%u "
            "total_ns=%llu parse_route_ns=%llu stage_ns=%llu refused_ns=%llu replan_ns=%llu "
            "alloc_in_ns=%llu copy_in_ns=%llu alloc_scratch_ns=%llu encode_ns=%llu gpu_ns=%llu "
            "wait_overhead_ns=%llu widen_alloc_ns=%llu widen_store_ns=%llu publish_ns=%llu\n",
            shape_buf, dtype_str, kind_str, status_str, data.narrowed ? 1 : 0, data.replan_iters,
            static_cast<unsigned long long>(data.total_ns),
            static_cast<unsigned long long>(data.parse_route_ns),
            static_cast<unsigned long long>(data.stage_ns),
            static_cast<unsigned long long>(data.refused_ns),
            static_cast<unsigned long long>(data.replan_ns),
            static_cast<unsigned long long>(data.alloc_in_ns),
            static_cast<unsigned long long>(data.copy_in_ns),
            static_cast<unsigned long long>(data.alloc_scratch_ns),
            static_cast<unsigned long long>(data.encode_ns),
            static_cast<unsigned long long>(data.gpu_ns),
            static_cast<unsigned long long>(data.wait_overhead_ns),
            static_cast<unsigned long long>(data.widen_alloc_ns),
            static_cast<unsigned long long>(data.widen_store_ns),
            static_cast<unsigned long long>(data.publish_ns));
        if (len > 0) {
            const size_t n = static_cast<size_t>(len) < sizeof(buf) ? static_cast<size_t>(len) : sizeof(buf) - 1;
            buf[n - 1] = '\n';
            std::fwrite(buf, 1, n, stderr);
        }
    }
};
#endif

ms_status ms_fft_transform(ms_context* context, const ms_array* input, const ms_fft_options* options,
                           const ms_route_options* route_options, ms_array_result* out, ms_route_info* info) {
#if defined(MS_ENABLE_METAL)
    ProfileGuard profile(input, options);
    fft::Profile* const prof = profile.active ? &profile.data : nullptr;
    profile.status = ms::detail::api(context, [&] {
        using Clock = std::chrono::steady_clock;
        Clock::time_point t0;
        if (prof) t0 = Clock::now();
        if (!out) throw Error(MS_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(route_options);
        const array::Shape shape = array::checked_shape(input);
        const fft::Call call = fft::parse(input->dtype, shape, options);
        const ms::detail::router::Decision decision = fft::route(context, call, *route_options);
        if (prof) {
            prof->parse_route_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
        }
        const fft::Limits limits = fft::device_limits();
        ms::detail::GpuOperationScope operation(context, true);
        ms::detail::BufferPtr result;
        try {
            result = fft::transform(context, *input, call, limits, prof);
        } catch (const Error& error) {
            // A drop-in must not fail where SciPy would not: without room for
            // the buffers the call is declined instead.
            if (error.status == MS_EOOM)
                throw Error(MS_ENOTSUP, std::string("the context's memory budget cannot hold the FFT's buffers: ") + error.what());
            throw;
        }
        Clock::time_point tp;
        if (prof) tp = Clock::now();
        operation.complete();
        array::publish(out, std::move(result), call.out_dtype, call.out_shape);
        array::report(info, decision.engine, decision.compute_dtype, "");
        if (prof) {
            prof->publish_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - tp).count());
        }
    });
    return profile.status;
#else
    return ms::detail::api(context, [&] {
        if (!out) throw Error(MS_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(route_options);
        const array::Shape shape = array::checked_shape(input);
        const fft::Call call = fft::parse(input->dtype, shape, options);
        const ms::detail::router::Decision decision = fft::route(context, call, *route_options);
        (void)decision;
        (void)info;
        throw Error(MS_ENOTSUP, "this build has no GPU engine and there is no native CPU FFT");
    });
#endif
}

ms_status ms_fft_plan(ms_context* context, uint32_t dtype, uint32_t ndim, const uint64_t* shape,
                      const ms_fft_options* options, const ms_route_options* route_options, uint32_t prepare,
                      ms_fft_plan_info* out) {
    return ms::detail::api(context, [&] {
        if (!out || out->struct_size != sizeof(ms_fft_plan_info) || out->abi_version != MS_FFT_ABI_VERSION)
            throw Error(MS_EINVAL, "FFT plan info size or ABI version mismatch");
        array::check_options(route_options);
        const array::Shape checked = fft::checked_shape(dtype, ndim, shape);
        ms_fft_plan_info result{};
        result.struct_size = sizeof(result);
        result.abi_version = MS_FFT_ABI_VERSION;
        std::string description;
        try {
            const fft::Call call = fft::parse(dtype, checked, options);
            const ms::detail::router::Decision decision = fft::route(context, call, *route_options);
#if defined(MS_ENABLE_METAL)
            // Preparing builds the pipelines and so finds the plans that fit them;
            // otherwise the plans are the planner's for this device's limits.
            const fft::Staged staged = prepare ? fft::fit(call, fft::device_limits()) : fft::stage(call, fft::device_limits());
            std::vector<fft::Dispatch> dispatches;
            for (const fft::Stage& each : staged.stages) {
                std::string reason;
                if (!fft::instantiate(*each.plan, each.count, dispatches, reason)) throw Error(MS_ENOTSUP, reason);
                result.passes += static_cast<uint32_t>(each.plan->steps.size());
            }
            if (prepare) fft::prepare(staged.stages);
            result.status = MS_OK;
            result.engine = decision.engine;
            result.device_bytes = fft::footprint(call, staged);
            description = staged.description;
#else
            (void)decision;
            (void)prepare;
            throw Error(MS_ENOTSUP, "this build has no GPU engine and there is no native CPU FFT");
#endif
        } catch (const Error& error) {
            if (error.status != MS_ENOTSUP) throw;
            result.status = MS_ENOTSUP;
            result.passes = 0;
            description = error.what();
        }
        std::snprintf(result.description, sizeof(result.description), "%s", description.c_str());
        *out = result;
    });
}

uint64_t ms_fft_auto_min_points(void) {
    // An upper bound on the work of any call with p points: every one of up to
    // MS_MAX_NDIM axes as costly per point as the longest possible, in double
    // precision.
    static const uint64_t points = [] {
        for (uint64_t p = 2;; ++p) {
            const double axes = std::min<double>(MS_MAX_NDIM, std::floor(std::log2(static_cast<double>(p))));
            const double saved = static_cast<double>(p) * (axes * fft::host_ns_per_point(p) * fft::host_double_factor -
                                                           fft::required_advantage * fft::device_ns_per_point_double);
            if (saved >= static_cast<double>(fft::family.gpu_min_work)) return p;
        }
    }();
    return points;
}
