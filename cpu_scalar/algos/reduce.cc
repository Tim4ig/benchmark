#include "algorithms.h"

namespace bench::cpu_scalar {
namespace {
f32 reduce_sum(const f32* x, usize n) {
  f32 sum = 0.0f;
  for (usize i = 0; i < n; ++i) {
    sum += x[i];
  }
  return sum;
}
} // namespace

BenchResult ReduceAlgorithm::run(const BenchOptions& options) {
  const usize n = options.n;
  auto data = bench::make_random(n, options.seed);
  f32 sum = 0.0f;

  BenchmarkSpec spec{};
  spec.flops = static_cast<u64>(n);
  spec.bytes_moved = static_cast<u64>(n * sizeof(f32));

  auto run = [&]() {
    sum = reduce_sum(data.data(), n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = sum;
  return result;
}
} // namespace bench::cpu_scalar
