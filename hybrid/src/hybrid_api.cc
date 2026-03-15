#include "../../common_abi/bench_abi.h"
#include "../../common_abi/types.h"
#include "hybrid_runner.h"

namespace {
constexpr BenchEntry kEntries[] = {
    {"matmul", BenchAlgo::kBenchAlgoMatmul},
    {"nbody", BenchAlgo::kBenchAlgoNbody},
};

const char* get_name() {
  return "hybrid";
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
  static bench::hybrid::HybridRunner runner;
  BenchResult result = runner.run(*opts);
  *out = result;
  return result.status;
}
} // namespace

extern "C" const BenchApi* bench_get_api() {
  static BenchApi api{get_name, get_entries, run_bench};
  return &api;
}
