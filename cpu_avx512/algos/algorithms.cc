#include "algorithms.h"

namespace bench::cpu_avx512 {
std::unique_ptr<Algorithm> create_algorithm(BenchAlgo algo) {
  switch (algo) {
    case BenchAlgo::kBenchAlgoVecadd:
      return std::make_unique<VecAddAlgorithm>();
    case BenchAlgo::kBenchAlgoReduce:
      return std::make_unique<ReduceAlgorithm>();
    case BenchAlgo::kBenchAlgoPrefix:
      return std::make_unique<PrefixAlgorithm>();
    case BenchAlgo::kBenchAlgoHist:
      return std::make_unique<HistogramAlgorithm>();
    case BenchAlgo::kBenchAlgoConV2D:
      return std::make_unique<Conv2dAlgorithm>();
    case BenchAlgo::kBenchAlgoSpmv:
      return std::make_unique<SpmvAlgorithm>();
    case BenchAlgo::kBenchAlgoMatmul:
      return std::make_unique<MatmulAlgorithm>();
    case BenchAlgo::kBenchAlgoBlackscholes:
      return std::make_unique<BlackScholesAlgorithm>();
    case BenchAlgo::kBenchAlgoBsort:
      return std::make_unique<BitonicSortAlgorithm>();
    case BenchAlgo::kBenchAlgoNbody:
      return std::make_unique<NBodyAlgorithm>();
    default:
      return nullptr;
  }
}
} // namespace bench::cpu_avx512
