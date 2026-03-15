#include "cpu_ref_runner.h"

#include "../bench_settings.h"
#include "algos/algorithms.h"

namespace bench::cpu_ref {
namespace {
BenchResult make_error_result(i32 status) {
  BenchResult result{};
  result.status = status;
  return result;
}
} // namespace

BenchResult CpuRefRunner::run(const BenchOptions& options) const {
  BenchOptions resolved = bench::resolve_options(options);

  auto algorithm = create_algorithm(resolved.algo);
  if (!algorithm) {
    return make_error_result(-2);
  }

  return algorithm->run(resolved);
}
} // namespace bench::cpu_ref
