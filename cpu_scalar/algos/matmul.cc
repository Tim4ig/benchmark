#include "algorithms.h"

#include <vector>

namespace bench::cpu_scalar {
namespace {
void matmul(const f32* a, const f32* b, f32* c, usize n) {
  const usize nn = n * n;
  for (usize idx = 0; idx < nn; ++idx) {
    c[idx] = 0.0f;
  }
  for (usize i = 0; i < n; ++i) {
    const usize row = i * n;
    for (usize k = 0; k < n; ++k) {
      const f32 aik = a[row + k];
      const usize brow = k * n;
      for (usize j = 0; j < n; ++j) {
        c[row + j] += aik * b[brow + j];
      }
    }
  }
}
} // namespace

BenchResult MatmulAlgorithm::run(const BenchOptions& options) {
  const usize n = options.n;
  const usize nn = n * n;
  auto a = bench::make_random(nn, options.seed);
  auto b = bench::make_random(nn, options.seed + 1u);
  std::vector<f32> c(nn, 0.0f);

  const f64 flops = 2.0 * static_cast<f64>(n) * static_cast<f64>(n) * static_cast<f64>(n);
  BenchmarkSpec spec{};
  spec.flops = static_cast<u64>(flops);
  spec.bytes_moved = static_cast<u64>(3 * nn * sizeof(f32));

  auto run = [&]() {
    matmul(a.data(), b.data(), c.data(), n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(c.data(), nn);
  return result;
}
} // namespace bench::cpu_scalar
