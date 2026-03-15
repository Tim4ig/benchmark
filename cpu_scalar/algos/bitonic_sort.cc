#include "algorithms.h"

#include <algorithm>
#include <vector>

namespace bench::cpu_scalar {
namespace {
void bitonic_sort(f32* data, usize n) {
  for (usize k = 2; k <= n; k <<= 1) {
    for (usize j = k >> 1; j > 0; j >>= 1) {
      for (usize i = 0; i < n; ++i) {
        const usize l = i ^ j;
        if (l > i) {
          const bool asc = (i & k) == 0;
          if (asc ? (data[i] > data[l]) : (data[i] < data[l])) {
            std::swap(data[i], data[l]);
          }
        }
      }
    }
  }
}
} // namespace

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
  const u64 comparisons = (static_cast<u64>(n2) >> 1) * log2n * (log2n + 1) / 2;

  BenchmarkSpec spec{};
  spec.flops = comparisons;
  spec.bytes_moved = static_cast<u64>(2 * n2 * sizeof(f32));

  auto run = [&]() {
    data = orig;
    bitonic_sort(data.data(), n2);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(data.data(), n2);
  return result;
}
} // namespace bench::cpu_scalar
