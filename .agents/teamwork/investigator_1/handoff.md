# A0 Stage Requirements Investigation Report

## Observation
Investigated `metal-sci` codebase for A0 stage tasks.
- `fft/_basic.py`: Contains `__public__`, `performance_cores()`, and logic for FFT fall-through using `_THREADED_MIN_POINTS` and `_THREADED_FALLBACK`.
- `_native.py`: Native core bindings, suitable for `performance_cores()`.
- `_dispatch.py`: Already handles `__module__`/`__qualname__` via `accelerated`.
- `tests/python/test_fft_oracle.py`: Contains a `pytest_configure` hook (lines 34-45) for `scipy.fft` namespace capabilities.

## Logic Chain
1.  **FFT route**: The private `_THREADED_MIN_POINTS` should be replaced by calls to `_native.decline_small` and `metal_sci.set_small_inputs` to centralize small input handling.
2.  **`NotAccelerated`**: To allow keyword overrides for fall-through (like `workers=`), the handling logic in `fft/_basic.py` should be moved to `_dispatch.accelerated`.
3.  **Shared `performance_cores()`**: Moving this to `_native.py` removes duplication as `ndimage` or other modules may eventually need it.
4.  **Redundant plugin hook**: The `pytest_configure` hook in `test_fft_oracle.py` appears intended as a temporary measure until `metal_sci_oracle` handles it.
5.  **FFT wrappers**: Since `_dispatch.accelerated` already sets `__module__`/`__qualname__`, the local `_public` helper and its calls in `fft/_basic.py` are redundant and can be removed.
6.  **Threaded FFT fall-through**: The current implementation of `_THREADED_FALLBACK` impacts strict precision compliance in some cases. Defaulting to `0` or removing it seems necessary to maintain strict results.

## Caveats
- Design decision for threaded FFT fall-through requires verification of performance impact if disabled.

## Conclusion
- The required A0 tasks are well-scoped.
- The `_dispatch` and `_native` modules are the appropriate locations for shared functionality and generalized dispatch logic.
- The `tests` modification is a cleanup task.

## Verification Method
1.  Apply code changes.
2.  Run tests: `python -m pytest tests/python/`.
3.  Ensure `tests/python/test_fft_oracle.py` passes without the redundant hook.
4.  Verify that FFT fall-through behaves as expected with precision policies.
