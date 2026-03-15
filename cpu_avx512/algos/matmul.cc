#include "../algorithms_opt.hpp"
#include "algorithms.h"

#include <vector>

namespace bench::cpu_avx512 {
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
    bench::matmul_cpu_opt(a.data(), b.data(), c.data(), n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(c.data(), nn);
  return result;
}
} // namespace bench::cpu_avx512
