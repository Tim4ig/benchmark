#include "algorithms.h"

#include <vector>

namespace bench::cpu_scalar {
namespace {
void vec_add(const f32* x, const f32* y, f32* out, f32 a, usize n) {
  for (usize i = 0; i < n; ++i) {
    out[i] = a * x[i] + y[i];
  }
}
} // namespace

BenchResult VecAddAlgorithm::run(const BenchOptions& options) {
  const usize n = options.n;
  const f32 alpha = 0.7f;
  auto x = bench::make_random(n, options.seed);
  auto y = bench::make_random(n, options.seed + 1u);
  std::vector<f32> out(n, 0.0f);

  BenchmarkSpec spec{};
  spec.flops = static_cast<u64>(2 * n);
  spec.bytes_moved = static_cast<u64>(3 * n * sizeof(f32));

  auto run = [&]() {
    vec_add(x.data(), y.data(), out.data(), alpha, n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(out.data(), n);
  return result;
}
} // namespace bench::cpu_scalar
