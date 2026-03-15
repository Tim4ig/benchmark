#include "../algorithms_opt.hpp"
#include "algorithms.h"

#include <vector>

namespace bench::cpu_avx512 {
BenchResult BlackScholesAlgorithm::run(const BenchOptions& options) {
  const auto n = options.n;
  auto s = make_random(n, options.seed, 80.0f, 120.0f);
  std::vector<f32> out(n, 0.0f);

  BenchmarkSpec spec{};
  spec.flops = static_cast<u64>(50.0 * static_cast<f64>(n));
  spec.bytes_moved = static_cast<u64>(2 * n * sizeof(f32));

  auto run = [&]() {
    blackscholes_cpu_opt(s.data(), out.data(), n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = checksum(out.data(), n);
  return result;
}
} // namespace bench::cpu_avx512
