#pragma once
#include "../common_abi/bench_abi.h"

namespace bench::cpu_mt {
BenchResult cpu_mt_runner_run(const BenchOptions& options);
} // namespace bench::cpu_mt
