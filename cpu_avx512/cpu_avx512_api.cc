#include "../common_abi/bench_abi.h"
#include "../common_abi/types.h"
#include "cpu_avx512_runner.h"

namespace {
constexpr BenchEntry kEntries[] = {
    {"vecadd", BenchAlgo::kBenchAlgoVecadd},
    {"reduce", BenchAlgo::kBenchAlgoReduce},
    {"prefix", BenchAlgo::kBenchAlgoPrefix},
    {"hist", BenchAlgo::kBenchAlgoHist},
    {"conv2d", BenchAlgo::kBenchAlgoConV2D},
    {"spmv", BenchAlgo::kBenchAlgoSpmv},
    {"matmul", BenchAlgo::kBenchAlgoMatmul},
    {"blackscholes", BenchAlgo::kBenchAlgoBlackscholes},
    {"bsort", BenchAlgo::kBenchAlgoBsort},
    {"nbody", BenchAlgo::kBenchAlgoNbody},
};

const char* get_name() {
  return "cpu_avx512";
}

u32 get_entries(const BenchEntry** out_entries) {
  if (out_entries != nullptr) {
    *out_entries = kEntries;
  }
  return static_cast<u32>(sizeof(kEntries) / sizeof(kEntries[0]));
}

i32 run_bench(const BenchOptions* options, BenchResult* out_result) {
  if (options == nullptr || out_result == nullptr) {
    return -1;
  }

  const bench::cpu_avx512::CpuAvx512Runner runner;
  BenchResult result = runner.run(*options);
  *out_result = result;
  return result.status;
}
} // namespace

extern "C" const BenchApi* bench_get_api() {
  static BenchApi api{get_name, get_entries, run_bench};
  return &api;
}
