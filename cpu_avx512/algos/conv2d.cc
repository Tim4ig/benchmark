#include "../algorithms_opt.hpp"
#include "algorithms.h"

#include <vector>

namespace bench::cpu_avx512 {
BenchResult Conv2dAlgorithm::run(const BenchOptions& options) {
  const usize n = options.n;
  const usize ksize = options.ksize;
  const usize size = n * n;
  auto input = bench::make_random(size, options.seed);
  auto kernel = bench::make_kernel(ksize, options.seed + 1u);
  std::vector<f32> out(size, 0.0f);

  const f64 flops = 2.0 * static_cast<f64>(ksize) * static_cast<f64>(ksize) * static_cast<f64>(size);
  BenchmarkSpec spec{};
  spec.flops = static_cast<u64>(flops);
  spec.bytes_moved = static_cast<u64>((2 * size + ksize * ksize) * sizeof(f32));

  auto run = [&]() {
    bench::conv2d_cpu_opt(input.data(), kernel.data(), out.data(), n, n, ksize);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(out.data(), size);
  return result;
}
} // namespace bench::cpu_avx512
