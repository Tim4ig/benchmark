#pragma once

#include "../common_abi/bench_abi.h"

namespace bench::cpu_ref {
class CpuRefRunner final {
 public:
  BenchResult run(const BenchOptions& options) const;
};
} // namespace bench::cpu_ref

