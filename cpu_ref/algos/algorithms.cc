#include "algorithms.h"

#include <vector>

namespace bench
{
  namespace cpu_ref
  {
    namespace
    {
      void VecAdd(const f32 *x, const f32 *y, f32 *out, f32 a, usize n)
      {
        for (usize i = 0; i < n; ++i)
        {
          out[i] = a * x[i] + y[i];
        }
      }

      void Matmul(const f32 *a, const f32 *b, f32 *c, usize n)
      {
        const usize nn = n * n;
        for (usize idx = 0; idx < nn; ++idx)
        {
          c[idx] = 0.0f;
        }
        for (usize i = 0; i < n; ++i)
        {
          const usize row = i * n;
          for (usize k = 0; k < n; ++k)
          {
            const f32 aik = a[row + k];
            const usize brow = k * n;
            for (usize j = 0; j < n; ++j)
            {
              c[row + j] += aik * b[brow + j];
            }
          }
        }
      }

      f32 ReduceSum(const f32 *x, usize n)
      {
        f32 sum = 0.0f;
        for (usize i = 0; i < n; ++i)
        {
          sum += x[i];
        }
        return sum;
      }

      void PrefixSumInclusive(const f32 *in, f32 *out, usize n)
      {
        if (n == 0)
        {
          return;
        }
        out[0] = in[0];
        for (usize i = 1; i < n; ++i)
        {
          out[i] = out[i - 1] + in[i];
        }
      }

      void HistogramU32(const u32 *data, usize n, u32 *bins, usize nbins)
      {
        for (usize i = 0; i < nbins; ++i)
        {
          bins[i] = 0;
        }
        if (nbins == 0)
        {
          return;
        }
        for (usize i = 0; i < n; ++i)
        {
          const usize idx = static_cast<usize>(data[i]) % nbins;
          ++bins[idx];
        }
      }

      void Conv2d(const f32 *input, const f32 *kernel, f32 *output, usize width, usize height,
                  usize ksize)
      {
        if (width == 0 || height == 0 || ksize == 0)
        {
          return;
        }
        const usize radius = ksize / 2;
        for (usize y = 0; y < height; ++y)
        {
          for (usize x = 0; x < width; ++x)
          {
            f32 acc = 0.0f;
            for (usize ky = 0; ky < ksize; ++ky)
            {
              const usize in_y = y + ky;
              if (in_y < radius || in_y >= height + radius)
              {
                continue;
              }
              const usize src_y = in_y - radius;
              for (usize kx = 0; kx < ksize; ++kx)
              {
                const usize in_x = x + kx;
                if (in_x < radius || in_x >= width + radius)
                {
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

      void SpmvCsr(const usize *row_ptr, const usize *col_idx, const f32 *values, const f32 *x, f32 *y,
                   usize rows)
      {
        for (usize row = 0; row < rows; ++row)
        {
          f32 sum = 0.0f;
          const usize start = row_ptr[row];
          const usize end = row_ptr[row + 1];
          for (usize idx = start; idx < end; ++idx)
          {
            sum += values[idx] * x[col_idx[idx]];
          }
          y[row] = sum;
        }
      }
    } // namespace

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

      auto run = [&]() { VecAdd(x.data(), y.data(), out.data(), alpha, n); };

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

      auto run = [&]() { sum = ReduceSum(data.data(), n); };

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

      auto run = [&]() { PrefixSumInclusive(data.data(), out.data(), n); };

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

      auto run = [&]() { HistogramU32(data.data(), n, hist.data(), bins); };

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

      auto run = [&]() { Conv2d(input.data(), kernel.data(), out.data(), n, n, ksize); };

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
        SpmvCsr(mat.row_ptr.data(), mat.col_idx.data(), mat.values.data(), x.data(), y.data(), n);
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

      auto run = [&]() { Matmul(a.data(), b.data(), c.data(), n); };

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
  } // namespace cpu_ref
} // namespace bench