#include "algorithms.h"

#include <vector>

namespace bench::cpu_scalar {
namespace {
void prefix_sum_inclusive(const f32* in, f32* out, usize n) {
  if (n == 0) {
    return;
  }
  out[0] = in[0];
  for (usize i = 1; i < n; ++i) {
    out[i] = out[i - 1] + in[i];
  }
}
} // namespace

BenchResult PrefixAlgorithm::run(const BenchOptions& options) {
  const usize n = options.n;
  auto data = bench::make_random(n, options.seed);
  std::vector<f32> out(n, 0.0f);

  BenchmarkSpec spec{};
  spec.flops = static_cast<u64>(n);
  spec.bytes_moved = static_cast<u64>(2 * n * sizeof(f32));

  auto run = [&]() {
    prefix_sum_inclusive(data.data(), out.data(), n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(out.data(), n);
  return result;
}
} // namespace bench::cpu_scalar
