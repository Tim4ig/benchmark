#include "../algorithms_opt.hpp"
#include "algorithms.h"

#include <vector>

namespace bench::cpu_avx512 {
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
    bench::vecadd_cpu_opt(x.data(), y.data(), out.data(), alpha, n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(out.data(), n);
  return result;
}
} // namespace bench::cpu_avx512
