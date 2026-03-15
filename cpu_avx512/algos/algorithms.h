#pragma once

#include "algorithm.h"

#include <memory>

namespace bench::cpu_avx512 {
class VecAddAlgorithm final : public Algorithm {
 public:
  BenchResult run(const BenchOptions& options) override;
};

class ReduceAlgorithm final : public Algorithm {
 public:
  BenchResult run(const BenchOptions& options) override;
};

class PrefixAlgorithm final : public Algorithm {
 public:
  BenchResult run(const BenchOptions& options) override;
};

class HistogramAlgorithm final : public Algorithm {
 public:
  BenchResult run(const BenchOptions& options) override;
};

class Conv2dAlgorithm final : public Algorithm {
 public:
  BenchResult run(const BenchOptions& options) override;
};

class SpmvAlgorithm final : public Algorithm {
 public:
  BenchResult run(const BenchOptions& options) override;
};

class MatmulAlgorithm final : public Algorithm {
 public:
  BenchResult run(const BenchOptions& options) override;
};

class BlackScholesAlgorithm final : public Algorithm {
 public:
  BenchResult run(const BenchOptions& options) override;
};

class BitonicSortAlgorithm final : public Algorithm {
 public:
  BenchResult run(const BenchOptions& options) override;
};

class NBodyAlgorithm final : public Algorithm {
 public:
  BenchResult run(const BenchOptions& options) override;
};

std::unique_ptr<Algorithm> create_algorithm(BenchAlgo algo);
} // namespace bench::cpu_avx512

