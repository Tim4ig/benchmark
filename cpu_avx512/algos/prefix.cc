#include "../algorithms_opt.hpp"
#include "algorithms.h"

#include <vector>

namespace bench::cpu_avx512 {
BenchResult PrefixAlgorithm::run(const BenchOptions& options) {
  const usize n = options.n;
  auto data = bench::make_random(n, options.seed);
  std::vector<f32> out(n, 0.0f);

  BenchmarkSpec spec{};
  spec.flops = static_cast<u64>(n);
  spec.bytes_moved = static_cast<u64>(2 * n * sizeof(f32));

  auto run = [&]() {
    bench::prefix_sum_inclusive_cpu_opt(data.data(), out.data(), n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(out.data(), n);
  return result;
}
} // namespace bench::cpu_avx512
