#include "../common_abi/bench_abi.h"
#include "../common_abi/types.h"
#include "cpu_auto_runner.h"

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
  return "cpu_auto";
}

u32 get_entries(const BenchEntry** out) {
  if (out != nullptr) {
    *out = kEntries;
  }
  return sizeof(kEntries) / sizeof(kEntries[0]);
}

i32 run_bench(const BenchOptions* options, BenchResult* out_result) {
  if (options == nullptr || out_result == nullptr) {
    return -1;
  }
  constexpr bench::cpu_auto::CpuAutoRunner kRunner;
  const BenchResult result = kRunner.run(*options);
  *out_result = result;
  return result.status;
}
} // namespace

extern "C" const BenchApi* bench_get_api() {
  static BenchApi api{get_name, get_entries, run_bench};
  return &api;
}
