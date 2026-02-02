#include "../common_abi/bench_abi.h"
#include "../common_abi/types.h"
#include "cpu_ref_runner.h"

namespace
{
  constexpr BenchEntry kEntries[] = {
    {"vecadd", BENCH_ALGO_VECADD}, {"reduce", BENCH_ALGO_REDUCE}, {"prefix", BENCH_ALGO_PREFIX},
    {"hist", BENCH_ALGO_HIST}, {"conv2d", BENCH_ALGO_CONV2D}, {"spmv", BENCH_ALGO_SPMV},
    {"matmul", BENCH_ALGO_MATMUL},
  };

  const char *GetName() { return "cpu_ref"; }

  u32 GetEntries(const BenchEntry **out_entries)
  {
    if (out_entries)
    {
      *out_entries = kEntries;
    }
    return static_cast<u32>(sizeof(kEntries) / sizeof(kEntries[0]));
  }

  i32 RunBench(const BenchOptions *options, BenchResult *out_result)
  {
    if (!options || !out_result)
    {
      return -1;
    }

    const bench::cpu_ref::CpuRefRunner runner;
    BenchResult result = runner.Run(*options);
    *out_result = result;
    return result.status;
  }
} // namespace

extern "C" const BenchApi *bench_get_api()
{
  static BenchApi api{GetName, GetEntries, RunBench};
  return &api;
}