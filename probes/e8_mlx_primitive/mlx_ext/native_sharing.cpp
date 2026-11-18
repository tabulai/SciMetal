// E8 probe: the DOCUMENTED way to move mx.array across modules -- a nanobind
// module in NB_DOMAIN "mlx" whose signatures use mlx::core::array directly and
// rely on nanobind finding MLX's own binding of that type.
//
// Built with the nanobind available in this environment (3.1.0) to record
// what happens against an MLX core that was built with nanobind 2.10.2.

#include <nanobind/nanobind.h>

#include "mlx/array.h"
#include "mlx/ops.h"

namespace nb = nanobind;
namespace mx = mlx::core;

NB_MODULE(_e8_nbnative, m) {
  // mx.array -> C++
  m.def("ndim", [](const mx::array& a) { return static_cast<int>(a.ndim()); });
  // C++ -> mx.array
  m.def("make", []() { return mx::array(1.0f); });
  // both directions
  m.def("twice", [](const mx::array& a) {
    return mx::multiply(a, mx::array(2.0f));
  });
}
