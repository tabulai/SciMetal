# Vendored metal-cpp provenance

This directory is an unmodified copy of Apple's Apache-2.0-licensed
[`apple/metal-cpp`](https://github.com/apple/metal-cpp) headers, plus this provenance
file.

- Upstream commit: `27c4382b7151d55a51692cdcb27aaa98752240de`
- Upstream commit date: 2026-06-08
- Reported `METALCPP_VERSION`: 381.0.0
- License: `LICENSE.txt` in this directory

metal-arrow needs these headers only when it builds Metal code: `MA_ENABLE_METAL=ON`
(the `metal*` presets and the wheel) or `MA_S0_METAL_PROBE=ON`. They are committed so that
a clean checkout or sdist builds without the sibling `../metal-graph` checkout that
earlier builds read them from; the copy is byte-identical to metal-graph's
`third_party/metal-cpp`, apart from this file. Configuration checks only that
`Metal/Metal.hpp` exists, so an incomplete tree fails at compile time, not at
configure time. Updating the dependency should replace the full tree from a reviewed
upstream commit and update this file in the same change.
