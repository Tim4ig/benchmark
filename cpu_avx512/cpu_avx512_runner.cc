#include "cpu_avx512_runner.h"

#include "algos/algorithms.h"
#include "../bench_settings.h"

namespace bench
{
  namespace cpu_avx512
  {
    namespace
    {
      BenchResult MakeErrorResult(i32 status)
      {
        BenchResult result{};
        result.status = status;
        return result;
      }
    } // namespace

    BenchResult CpuAvx512Runner::Run(const BenchOptions &options) const
    {
      BenchOptions resolved = bench::ResolveOptions(options);

      auto algorithm = CreateAlgorithm(resolved.algo);
      if (!algorithm) { return MakeErrorResult(-2); }

      return algorithm->Run(resolved);
    }
  } // namespace cpu_avx512
} // namespace bench