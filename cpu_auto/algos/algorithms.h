#pragma once

#include "../../cpu_scalar/algos/algorithms.h"
#include "algorithm.h"

namespace bench::cpu_auto {
using cpu_scalar::BitonicSortAlgorithm;
using cpu_scalar::BlackScholesAlgorithm;
using cpu_scalar::Conv2dAlgorithm;
using cpu_scalar::HistogramAlgorithm;
using cpu_scalar::MatmulAlgorithm;
using cpu_scalar::NBodyAlgorithm;
using cpu_scalar::PrefixAlgorithm;
using cpu_scalar::ReduceAlgorithm;
using cpu_scalar::SpmvAlgorithm;
using cpu_scalar::VecAddAlgorithm;

inline std::unique_ptr<Algorithm> create_algorithm(const BenchAlgo algo) {
  return cpu_scalar::create_algorithm(algo);
}
} // namespace bench::cpu_auto
