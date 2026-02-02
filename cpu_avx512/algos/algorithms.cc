#include "algorithms.h"

#include <vector>

#include "../algorithms_opt.hpp"

namespace bench
{
  namespace cpu_avx512
  {
    BenchResult VecAddAlgorithm::Run(const BenchOptions &options)
    {
      const usize n = options.n;
      const f32 alpha = 0.7f;
      auto x = bench::make_random(n, options.seed);
      auto y = bench::make_random(n, options.seed + 1u);
      std::vector<f32> out(n, 0.0f);

      BenchmarkSpec spec{};
      spec.flops = static_cast<u64>(2.0 * n);
      spec.bytes_moved = static_cast<u64>(3 * n * sizeof(f32));

      auto run = [&]() { bench::vecadd_cpu_opt(x.data(), y.data(), out.data(), alpha, n); };

      BenchResult result = RunBenchmark(options, run, spec);
      result.checksum = bench::checksum(out.data(), n);
      return result;
    }

    BenchResult ReduceAlgorithm::Run(const BenchOptions &options)
    {
      const usize n = options.n;
      auto data = bench::make_random(n, options.seed);
      f32 sum = 0.0f;

      BenchmarkSpec spec{};
      spec.flops = static_cast<u64>(n);
      spec.bytes_moved = static_cast<u64>(n * sizeof(f32));

      auto run = [&]() { sum = bench::reduce_sum_cpu_opt(data.data(), n); };

      BenchResult result = RunBenchmark(options, run, spec);
      result.checksum = sum;
      return result;
    }

    BenchResult PrefixAlgorithm::Run(const BenchOptions &options)
    {
      const usize n = options.n;
      auto data = bench::make_random(n, options.seed);
      std::vector<f32> out(n, 0.0f);

      BenchmarkSpec spec{};
      spec.flops = static_cast<u64>(n);
      spec.bytes_moved = static_cast<u64>(2 * n * sizeof(f32));

      auto run = [&]() { bench::prefix_sum_inclusive_cpu_opt(data.data(), out.data(), n); };

      BenchResult result = RunBenchmark(options, run, spec);
      result.checksum = bench::checksum(out.data(), n);
      return result;
    }

    BenchResult HistogramAlgorithm::Run(const BenchOptions &options)
    {
      const usize n = options.n;
      const usize bins = options.bins;
      auto data = bench::make_random_u32(n, options.seed, static_cast<u32>(bins));
      std::vector<u32> hist(bins, 0u);

      BenchmarkSpec spec{};
      spec.flops = static_cast<u64>(n);
      spec.bytes_moved = static_cast<u64>(n * sizeof(u32) + bins * sizeof(u32));

      auto run = [&]() { bench::histogram_u32_cpu_opt(data.data(), n, hist.data(), bins); };

      BenchResult result = RunBenchmark(options, run, spec);
      result.checksum = bench::checksum_u32(hist.data(), bins);
      return result;
    }

    BenchResult Conv2dAlgorithm::Run(const BenchOptions &options)
    {
      const usize n = options.n;
      const usize ksize = options.ksize;
      const usize size = n * n;
      auto input = bench::make_random(size, options.seed);
      auto kernel = bench::make_kernel(ksize, options.seed + 1u);
      std::vector<f32> out(size, 0.0f);

      const f64 flops =
          2.0 * static_cast<f64>(ksize) * static_cast<f64>(ksize) * static_cast<f64>(size);
      BenchmarkSpec spec{};
      spec.flops = static_cast<u64>(flops);
      spec.bytes_moved = static_cast<u64>((size + ksize * ksize) * sizeof(f32));

      auto run = [&]() { bench::conv2d_cpu_opt(input.data(), kernel.data(), out.data(), n, n, ksize); };

      BenchResult result = RunBenchmark(options, run, spec);
      result.checksum = bench::checksum(out.data(), size);
      return result;
    }

    BenchResult SpmvAlgorithm::Run(const BenchOptions &options)
    {
      const usize n = options.n;
      const usize nnz = options.nnz_per_row;
      const auto mat = bench::make_csr_matrix(n, n, nnz, options.seed);
      auto x = bench::make_random(n, options.seed + 1u);
      std::vector<f32> y(n, 0.0f);

      const f64 flops = 2.0 * static_cast<f64>(mat.values.size());
      BenchmarkSpec spec{};
      spec.flops = static_cast<u64>(flops);
      spec.bytes_moved = static_cast<u64>(mat.values.size() * sizeof(f32) +
                                          mat.col_idx.size() * sizeof(usize) +
                                          mat.row_ptr.size() * sizeof(usize) + x.size() * sizeof(f32) +
                                          y.size() * sizeof(f32));

      auto run = [&]()
      {
        bench::spmv_csr_cpu_opt(mat.row_ptr.data(), mat.col_idx.data(), mat.values.data(), x.data(),
                                y.data(), n);
      };

      BenchResult result = RunBenchmark(options, run, spec);
      result.checksum = bench::checksum(y.data(), n);
      return result;
    }

    BenchResult MatmulAlgorithm::Run(const BenchOptions &options)
    {
      const usize n = options.n;
      const usize nn = n * n;
      auto a = bench::make_random(nn, options.seed);
      auto b = bench::make_random(nn, options.seed + 1u);
      std::vector<f32> c(nn, 0.0f);

      const f64 flops = 2.0 * static_cast<f64>(n) * static_cast<f64>(n) * static_cast<f64>(n);
      BenchmarkSpec spec{};
      spec.flops = static_cast<u64>(flops);
      spec.bytes_moved = static_cast<u64>(3 * nn * sizeof(f32));

      auto run = [&]() { bench::matmul_cpu_opt(a.data(), b.data(), c.data(), n); };

      BenchResult result = RunBenchmark(options, run, spec);
      result.checksum = bench::checksum(c.data(), nn);
      return result;
    }

    std::unique_ptr<Algorithm> CreateAlgorithm(BenchAlgo algo)
    {
      switch (algo)
      {
        case BENCH_ALGO_VECADD:
          return std::make_unique<VecAddAlgorithm>();
        case BENCH_ALGO_REDUCE:
          return std::make_unique<ReduceAlgorithm>();
        case BENCH_ALGO_PREFIX:
          return std::make_unique<PrefixAlgorithm>();
        case BENCH_ALGO_HIST:
          return std::make_unique<HistogramAlgorithm>();
        case BENCH_ALGO_CONV2D:
          return std::make_unique<Conv2dAlgorithm>();
        case BENCH_ALGO_SPMV:
          return std::make_unique<SpmvAlgorithm>();
        case BENCH_ALGO_MATMUL:
          return std::make_unique<MatmulAlgorithm>();
        default:
          return nullptr;
      }
    }
  } // namespace cpu_avx512
} // namespace bench
