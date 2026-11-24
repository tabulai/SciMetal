## 2026-11-18T20:29:04Z
You are investigating the A0 stage requirements in the `metal-sci` project.
Read `/Users/tunguz/.gemini/antigravity/scratch/teamwork/d468ea9b-8fcd-495c-aeef-1dc927ad5fad/.agents/teamwork/sentinel_1/ORIGINAL_REQUEST.md` first.
Then investigate the codebase (specifically `fft/`, `_dispatch.py`, `_native.py`, `tests/python/test_fft_oracle.py`) to map features, constraints, dependencies, and file/line findings for the requested A0 tasks:
- FFT route: `_native.decline_small`, `metal_sci.set_small_inputs`.
- `NotAccelerated`: carry keyword overrides for fall-through.
- Shared `performance_cores()` in `_native.py`.
- Remove redundant plugin hook from `tests/python/test_fft_oracle.py`.
- Remove local `__module__`/`__qualname__` from FFT wrappers (`_public` in `fft/_basic.py`).
- Decide threaded FFT fall-through default behavior (considering precision policy).
Report findings, including proposed fix strategies, in `handoff.md`.
