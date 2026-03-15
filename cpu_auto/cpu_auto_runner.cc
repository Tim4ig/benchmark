#include "cpu_auto_runner.h"

#include "../bench_settings.h"
#include "algos/algorithms.h"

namespace bench::cpu_auto {
BenchResult CpuAutoRunner::run(const BenchOptions& options) {
  const BenchOptions resolved = resolve_options(options);
  const auto algorithm = create_algorithm(resolved.algo);
  if (!algorithm) {
    BenchResult r{};
    r.status = -2;
    return r;
  }
  return algorithm->run(resolved);
}
} // namespace bench::cpu_auto
