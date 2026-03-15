#include "algorithms.h"

#include <vector>

namespace bench::cpu_scalar {
namespace {
void conv2d(const f32* input, const f32* kernel, f32* output, usize width, usize height, usize ksize) {
  if (width == 0 || height == 0 || ksize == 0) {
    return;
  }
  const usize radius = ksize / 2;
  for (usize y = 0; y < height; ++y) {
    for (usize x = 0; x < width; ++x) {
      f32 acc = 0.0f;
      for (usize ky = 0; ky < ksize; ++ky) {
        const usize in_y = y + ky;
        if (in_y < radius || in_y >= height + radius) {
          continue;
        }
        const usize src_y = in_y - radius;
        for (usize kx = 0; kx < ksize; ++kx) {
          const usize in_x = x + kx;
          if (in_x < radius || in_x >= width + radius) {
            continue;
          }
          const usize src_x = in_x - radius;
          acc += input[src_y * width + src_x] * kernel[ky * ksize + kx];
        }
      }
      output[y * width + x] = acc;
    }
  }
}
} // namespace

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
    conv2d(input.data(), kernel.data(), out.data(), n, n, ksize);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(out.data(), size);
  return result;
}
} // namespace bench::cpu_scalar
