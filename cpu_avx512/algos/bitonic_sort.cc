#include "../algorithms_opt.hpp"
#include "algorithms.h"

#include <vector>

namespace bench::cpu_avx512 {
BenchResult BitonicSortAlgorithm::run(const BenchOptions& options) {
  usize n2 = 1;
  while (n2 < options.n) {
    n2 <<= 1;
  }

  std::vector<f32> data = bench::make_random(n2, options.seed);
  const std::vector<f32> orig = data;

  usize log2n = 0;
  for (usize t = n2; t > 1; t >>= 1) {
    ++log2n;
  }
  const auto comparisons = (static_cast<u64>(n2) >> 1) * log2n * (log2n + 1) / 2;

  BenchmarkSpec spec{};
  spec.flops = comparisons;
  spec.bytes_moved = static_cast<u64>(2 * n2 * sizeof(f32));

  auto run = [&]() {
    data = orig;
    bench::bitonic_sort_cpu_opt(data.data(), n2);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(data.data(), n2);
  return result;
}
} // namespace bench::cpu_avx512
