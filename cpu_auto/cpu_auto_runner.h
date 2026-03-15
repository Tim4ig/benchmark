#pragma once
#include "../common_abi/bench_abi.h"

namespace bench::cpu_auto {
class CpuAutoRunner final {
 public:
  static BenchResult run(const BenchOptions& options);
};
} // namespace bench::cpu_auto
