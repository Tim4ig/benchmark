#pragma once

#include "../../cpu_scalar/algos/algorithms.h"
#include "algorithm.h"

namespace bench::cpu_ref {
using bench::cpu_scalar::BitonicSortAlgorithm;
using bench::cpu_scalar::BlackScholesAlgorithm;
using bench::cpu_scalar::Conv2dAlgorithm;
using bench::cpu_scalar::HistogramAlgorithm;
using bench::cpu_scalar::MatmulAlgorithm;
using bench::cpu_scalar::NBodyAlgorithm;
using bench::cpu_scalar::PrefixAlgorithm;
using bench::cpu_scalar::ReduceAlgorithm;
using bench::cpu_scalar::SpmvAlgorithm;
using bench::cpu_scalar::VecAddAlgorithm;

inline std::unique_ptr<Algorithm> create_algorithm(const BenchAlgo algo) {
  return bench::cpu_scalar::create_algorithm(algo);
}
} // namespace bench::cpu_ref
