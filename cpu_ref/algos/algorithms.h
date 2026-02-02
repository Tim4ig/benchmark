#ifndef CPU_REF_ALGOS_ALGORITHMS_H_
#define CPU_REF_ALGOS_ALGORITHMS_H_

#include <memory>

#include "algorithm.h"

namespace bench
{
  namespace cpu_ref
  {
    class VecAddAlgorithm final : public Algorithm
    {
    public:
      BenchResult Run(const BenchOptions &options) override;
    };

    class ReduceAlgorithm final : public Algorithm
    {
    public:
      BenchResult Run(const BenchOptions &options) override;
    };

    class PrefixAlgorithm final : public Algorithm
    {
    public:
      BenchResult Run(const BenchOptions &options) override;
    };

    class HistogramAlgorithm final : public Algorithm
    {
    public:
      BenchResult Run(const BenchOptions &options) override;
    };

    class Conv2dAlgorithm final : public Algorithm
    {
    public:
      BenchResult Run(const BenchOptions &options) override;
    };

    class SpmvAlgorithm final : public Algorithm
    {
    public:
      BenchResult Run(const BenchOptions &options) override;
    };

    class MatmulAlgorithm final : public Algorithm
    {
    public:
      BenchResult Run(const BenchOptions &options) override;
    };

    std::unique_ptr<Algorithm> CreateAlgorithm(BenchAlgo algo);
  } // namespace cpu_ref
} // namespace bench

#endif  // CPU_REF_ALGOS_ALGORITHMS_H_