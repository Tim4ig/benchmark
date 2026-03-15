#include "../algorithms_opt.hpp"
#include "algorithms.h"

#include <vector>

namespace bench::cpu_avx512 {
BenchResult HistogramAlgorithm::run(const BenchOptions& options) {
  const usize n = options.n;
  const usize bins = options.bins;
  auto data = bench::make_random_u32(n, options.seed, static_cast<u32>(bins));
  std::vector<u32> hist(bins, 0u);

  BenchmarkSpec spec{};
  spec.flops = static_cast<u64>(n);
  spec.bytes_moved = static_cast<u64>(n * sizeof(u32) + bins * sizeof(u32));

  auto run = [&]() {
    bench::histogram_u32_cpu_opt(data.data(), n, hist.data(), bins);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum_u32(hist.data(), bins);
  return result;
}
} // namespace bench::cpu_avx512
