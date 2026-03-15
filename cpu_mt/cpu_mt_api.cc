#include "../common_abi/bench_abi.h"
#include "../common_abi/types.h"
#include "cpu_mt_runner.h"

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
  return "cpu_mt";
}

u32 get_entries(const BenchEntry** out) {
  if (out != nullptr) {
    *out = kEntries;
  }
  return static_cast<u32>(sizeof(kEntries) / sizeof(kEntries[0]));
}

i32 run_bench(const BenchOptions* opts, BenchResult* out) {
  if (opts == nullptr || out == nullptr) {
    return -1;
  }
  *out = bench::cpu_mt::cpu_mt_runner_run(*opts);
  return out->status;
}
} // namespace

extern "C" const BenchApi* bench_get_api() {
  static BenchApi api{get_name, get_entries, run_bench};
  return &api;
}
