#include "cpu_ref_runner.h"

#include "algos/algorithms.h"

namespace bench
{
  namespace cpu_ref
  {
    namespace
    {
      void FillDefaults(BenchOptions &options)
      {
        if (options.repeats == 0)
        {
          options.repeats = 5;
        }
        if (options.seed == 0)
        {
          options.seed = 1234;
        }
        if (options.n == 0)
        {
          switch (options.algo)
          {
            case BENCH_ALGO_CONV2D:
              options.n = 1024;
              break;
            case BENCH_ALGO_MATMUL:
              options.n = 512;
              break;
            case BENCH_ALGO_SPMV:
              options.n = 1u << 18;
              break;
            default:
              options.n = 1u << 20;
              break;
          }
        }
        if (options.bins == 0)
        {
          options.bins = 256;
        }
        if (options.nnz_per_row == 0)
        {
          options.nnz_per_row = 16;
        }
        if (options.ksize == 0)
        {
          options.ksize = 3;
        }
        if (options.ksize % 2 == 0)
        {
          options.ksize += 1;
        }
      }

      BenchResult MakeErrorResult(i32 status)
      {
        BenchResult result{};
        result.status = status;
        return result;
      }
    } // namespace

    BenchResult CpuRefRunner::Run(const BenchOptions &options) const
    {
      BenchOptions resolved = options;
      FillDefaults(resolved);

      auto algorithm = CreateAlgorithm(resolved.algo);
      if (!algorithm)
      {
        return MakeErrorResult(-2);
      }

      return algorithm->Run(resolved);
    }
  } // namespace cpu_ref
} // namespace bench