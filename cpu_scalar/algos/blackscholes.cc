#include "algorithms.h"

#include <cmath>
#include <vector>

namespace bench::cpu_scalar {
namespace {
static inline float norm_cdf(float x) {
  return 0.5f * std::erfc(-x * 0.7071067811865476f);
}

void black_scholes(const f32* s, f32* out, usize n) {
  const f32 strike_price = 100.0f;
  const f32 maturity = 1.0f;
  const f32 r = 0.05f;
  const f32 sigma = 0.2f;
  const f32 sigma_sqrt_t = sigma * std::sqrt(maturity);
  const f32 discount = strike_price * std::exp(-r * maturity);
  for (usize i = 0; i < n; ++i) {
    const f32 si = s[i];
    const f32 log_sk = std::log(si / strike_price);
    const f32 d1 = log_sk / sigma_sqrt_t + (r / sigma + 0.5f * sigma) * std::sqrt(maturity);
    const f32 d2 = d1 - sigma_sqrt_t;
    out[i] = si * norm_cdf(d1) - discount * norm_cdf(d2);
  }
}
} // namespace

BenchResult BlackScholesAlgorithm::run(const BenchOptions& options) {
  const usize n = options.n;
  auto s = bench::make_random(n, options.seed, 80.0f, 120.0f);
  std::vector<f32> out(n, 0.0f);

  BenchmarkSpec spec{};
  spec.flops = static_cast<u64>(50.0 * static_cast<f64>(n));
  spec.bytes_moved = static_cast<u64>(2 * n * sizeof(f32));

  auto run = [&]() {
    black_scholes(s.data(), out.data(), n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(out.data(), n);
  return result;
}
} // namespace bench::cpu_scalar
