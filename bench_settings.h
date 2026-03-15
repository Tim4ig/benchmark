#pragma once

#include "common_abi/bench_abi.h"

namespace bench {
struct AlgorithmSettings {
  BenchAlgo algo;
  usize n;
  usize bins;
  usize nnz_per_row;
  usize ksize;
};

inline constexpr u32 kDefaultRepeats = 20;
inline constexpr u32 kDefaultSeed = 1234;
inline constexpr usize kDefaultN = 1U << 20;
inline constexpr usize kDefaultBins = 256;
inline constexpr usize kDefaultNnzPerRow = 16;
inline constexpr usize kDefaultKsize = 3;

inline constexpr AlgorithmSettings kAlgorithmSettings[] = {
    {BenchAlgo::kBenchAlgoVecadd, kDefaultN, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BenchAlgo::kBenchAlgoReduce, kDefaultN, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BenchAlgo::kBenchAlgoPrefix, kDefaultN, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BenchAlgo::kBenchAlgoHist, kDefaultN, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BenchAlgo::kBenchAlgoConV2D, 1024, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BenchAlgo::kBenchAlgoSpmv, 1U << 18, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BenchAlgo::kBenchAlgoMatmul, 512, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BenchAlgo::kBenchAlgoBlackscholes, kDefaultN, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BenchAlgo::kBenchAlgoBsort, kDefaultN, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BenchAlgo::kBenchAlgoNbody, 4096, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
};

inline const AlgorithmSettings& get_algorithm_settings(BenchAlgo algo) {
  for (const auto& settings : kAlgorithmSettings) {
    if (settings.algo == algo) {
      return settings;
    }
  }
  return kAlgorithmSettings[0];
}

inline void apply_algorithm_defaults(BenchOptions& options) {
  if (options.repeats == 0) {
    options.repeats = kDefaultRepeats;
  }
  if (options.seed == 0) {
    options.seed = kDefaultSeed;
  }

  const AlgorithmSettings& settings = get_algorithm_settings(options.algo);
  if (options.n == 0) {
    options.n = settings.n;
  }
  if (options.bins == 0) {
    options.bins = settings.bins;
  }
  if (options.nnz_per_row == 0) {
    options.nnz_per_row = settings.nnz_per_row;
  }
  if (options.ksize == 0) {
    options.ksize = settings.ksize;
  }
  if (options.ksize % 2 == 0) {
    options.ksize += 1;
  }
}

inline BenchOptions resolve_options(const BenchOptions& options) {
  BenchOptions resolved = options;
  apply_algorithm_defaults(resolved);
  return resolved;
}

inline BenchOptions default_options() {
  BenchOptions options{};
  options.algo = BenchAlgo::kBenchAlgoVecadd;
  options.repeats = kDefaultRepeats;
  options.seed = kDefaultSeed;
  options.n = 0;
  options.bins = kDefaultBins;
  options.nnz_per_row = kDefaultNnzPerRow;
  options.ksize = kDefaultKsize;
  return options;
}
} // namespace bench

