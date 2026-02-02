#ifndef CPU_REF_ALGOS_ALGORITHM_H_
#define CPU_REF_ALGOS_ALGORITHM_H_

#include <utility>

#include "../../common_abi/bench_abi.h"
#include "../../common_abi/bench_utils.hpp"
#include "../../common_abi/types.h"

namespace bench
{
  namespace cpu_ref
  {
    struct BenchmarkSpec
    {
      u64 flops = 0;
      u64 bytes_moved = 0;
    };

    template<typename Fn>
    BenchResult RunBenchmark(const BenchOptions &options, Fn &&fn, const BenchmarkSpec &spec)
    {
      const auto timing = bench::measure_ms(static_cast<i32>(options.repeats), std::forward<Fn>(fn));
      BenchResult result{};
      result.status = 0;
      result.total_time_ms = timing.total_ms;
      result.calc_time_ms = timing.avg_ms;
      result.mem_time_ms = 0.0;
      result.flops = spec.flops;
      result.bytes_moved = spec.bytes_moved;
      result.gflops =
          spec.flops > 0 ? static_cast<f64>(spec.flops) / (timing.avg_ms * 1.0e6) : 0.0;
      result.gbytes =
          spec.bytes_moved > 0 ? static_cast<f64>(spec.bytes_moved) / (timing.avg_ms * 1.0e6) : 0.0;
      result.checksum = 0.0;
      return result;
    }

    class Algorithm
    {
    public:
      virtual ~Algorithm() = default;
      virtual BenchResult Run(const BenchOptions &options) = 0;
    };
  } // namespace cpu_ref
} // namespace bench

#endif  // CPU_REF_ALGOS_ALGORITHM_H_