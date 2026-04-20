#include "algorithms.h"

#include <cmath>
#include <vector>

namespace bench::cpu_scalar {
namespace {
void nbody_step(const f32* px, const f32* py, const f32* mass, f32* fx, f32* fy, usize n) {
  const f32 eps2 = 1e-4f;
  for (usize i = 0; i < n; ++i) {
    f32 ax = 0.0f;
    f32 ay = 0.0f;
    for (usize j = 0; j < n; ++j) {
      if (j == i) {
        continue;
      }
      const f32 dx = px[j] - px[i];
      const f32 dy = py[j] - py[i];
      const f32 dist2 = dx * dx + dy * dy + eps2;
      const f32 inv_dist = 1.0f / std::sqrt(dist2);
      const f32 inv_dist3 = inv_dist * inv_dist * inv_dist;
      const f32 f = mass[j] * inv_dist3;
      ax += f * dx;
      ay += f * dy;
    }
    fx[i] = ax;
    fy[i] = ay;
  }
}
} // namespace

BenchResult NBodyAlgorithm::run(const BenchOptions& options) {
  const usize n = options.n;
  auto px = bench::make_random(n, options.seed, -100.0f, 100.0f);
  auto py = bench::make_random(n, options.seed + 1u, -100.0f, 100.0f);
  auto mass = bench::make_random(n, options.seed + 2u, 0.1f, 2.0f);
  std::vector<f32> fx(n, 0.0f), fy(n, 0.0f);

  const f64 flops = 20.0 * static_cast<f64>(n) * static_cast<f64>(n - 1);
  BenchmarkSpec spec{};
  spec.flops = static_cast<u64>(flops);
  spec.bytes_moved = static_cast<u64>(5 * n * sizeof(f32));

  auto run = [&]() {
    nbody_step(px.data(), py.data(), mass.data(), fx.data(), fy.data(), n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum_xy(fx.data(), fy.data(), n);
  return result;
}
} // namespace bench::cpu_scalar
