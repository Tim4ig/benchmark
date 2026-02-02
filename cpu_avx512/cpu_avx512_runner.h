#ifndef CPU_AVX512_CPU_AVX512_RUNNER_H_
#define CPU_AVX512_CPU_AVX512_RUNNER_H_

#include "../common_abi/bench_abi.h"

namespace bench
{
  namespace cpu_avx512
  {
    class CpuAvx512Runner final
    {
    public:
      BenchResult Run(const BenchOptions &options) const;
    };
  } // namespace cpu_avx512
} // namespace bench

#endif  // CPU_AVX512_CPU_AVX512_RUNNER_H_