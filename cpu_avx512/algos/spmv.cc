#include "../algorithms_opt.hpp"
#include "algorithms.h"

#include <vector>

namespace bench::cpu_avx512 {
BenchResult SpmvAlgorithm::run(const BenchOptions& options) {
  const usize n = options.n;
  const usize nnz = options.nnz_per_row;
  const auto mat = bench::make_csr_matrix(n, n, nnz, options.seed);
  auto x = bench::make_random(n, options.seed + 1u);
  std::vector<f32> y(n, 0.0f);

  const f64 flops = 2.0 * static_cast<f64>(mat.values.size());
  BenchmarkSpec spec{};
  spec.flops = static_cast<u64>(flops);
  spec.bytes_moved =
      static_cast<u64>(mat.values.size() * sizeof(f32) + mat.col_idx.size() * sizeof(u32) +
                       mat.row_ptr.size() * sizeof(u32) + x.size() * sizeof(f32) + y.size() * sizeof(f32));

  auto run = [&]() {
    bench::spmv_csr_cpu_opt(mat.row_ptr.data(), mat.col_idx.data(), mat.values.data(), x.data(), y.data(), n);
  };

  BenchResult result = run_benchmark(options, run, spec);
  result.checksum = bench::checksum(y.data(), n);
  return result;
}
} // namespace bench::cpu_avx512
