#ifndef CPU_REF_CPU_REF_RUNNER_H_
#define CPU_REF_CPU_REF_RUNNER_H_

#include "../common_abi/bench_abi.h"

namespace bench
{
  namespace cpu_ref
  {
    class CpuRefRunner final
    {
    public:
      BenchResult Run(const BenchOptions &options) const;
    };
  } // namespace cpu_ref
} // namespace bench

#endif  // CPU_REF_CPU_REF_RUNNER_H_