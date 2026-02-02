#ifndef BENCH_SETTINGS_H_
#define BENCH_SETTINGS_H_

#include "common_abi/bench_abi.h"

namespace bench
{
  struct AlgorithmSettings
  {
    BenchAlgo algo;
    usize n;
    usize bins;
    usize nnz_per_row;
    usize ksize;
  };

  inline constexpr u32 kDefaultRepeats = 5;
  inline constexpr u32 kDefaultSeed = 1234;
  inline constexpr usize kDefaultN = 1u << 20;
  inline constexpr usize kDefaultBins = 256;
  inline constexpr usize kDefaultNnzPerRow = 16;
  inline constexpr usize kDefaultKsize = 3;

  inline constexpr AlgorithmSettings kAlgorithmSettings[] = {
    {BENCH_ALGO_VECADD, kDefaultN, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BENCH_ALGO_REDUCE, kDefaultN, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BENCH_ALGO_PREFIX, kDefaultN, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BENCH_ALGO_HIST, kDefaultN, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BENCH_ALGO_CONV2D, 1024, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BENCH_ALGO_SPMV, 1u << 18, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
    {BENCH_ALGO_MATMUL, 512, kDefaultBins, kDefaultNnzPerRow, kDefaultKsize},
  };

  inline const AlgorithmSettings &GetAlgorithmSettings(BenchAlgo algo)
  {
    for (const auto &settings: kAlgorithmSettings) { if (settings.algo == algo) { return settings; } }
    return kAlgorithmSettings[0];
  }

  inline void ApplyAlgorithmDefaults(BenchOptions &options)
  {
    if (options.repeats == 0) { options.repeats = kDefaultRepeats; }
    if (options.seed == 0) { options.seed = kDefaultSeed; }

    const AlgorithmSettings &settings = GetAlgorithmSettings(options.algo);
    if (options.n == 0) { options.n = settings.n; }
    if (options.bins == 0) { options.bins = settings.bins; }
    if (options.nnz_per_row == 0) { options.nnz_per_row = settings.nnz_per_row; }
    if (options.ksize == 0) { options.ksize = settings.ksize; }
    if (options.ksize % 2 == 0) { options.ksize += 1; }
  }

  inline BenchOptions ResolveOptions(const BenchOptions &options)
  {
    BenchOptions resolved = options;
    ApplyAlgorithmDefaults(resolved);
    return resolved;
  }

  inline BenchOptions DefaultOptions()
  {
    BenchOptions options{};
    options.algo = BENCH_ALGO_VECADD;
    options.repeats = kDefaultRepeats;
    options.seed = kDefaultSeed;
    options.n = 0;
    options.bins = kDefaultBins;
    options.nnz_per_row = kDefaultNnzPerRow;
    options.ksize = kDefaultKsize;
    return options;
  }
} // namespace bench

#endif  // BENCH_SETTINGS_H_