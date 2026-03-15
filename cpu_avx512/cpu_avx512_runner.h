#pragma once

#include "../common_abi/bench_abi.h"

namespace bench::cpu_avx512 {
class CpuAvx512Runner final {
 public:
  BenchResult run(const BenchOptions& options) const;
};
} // namespace bench::cpu_avx512

