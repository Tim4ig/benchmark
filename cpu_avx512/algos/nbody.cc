#include "../algorithms_opt.hpp"
#include "algorithms.h"

#include <vector>

namespace bench::cpu_avx512 {
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
    bench::nbody_step_cpu_opt(px.data(), py.data(), mass.data(), fx.data(), fy.data(), n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(fx.data(), n) + bench::checksum(fy.data(), n);
  return result;
}
} // namespace bench::cpu_avx512
