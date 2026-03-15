#pragma once

#include "types.h"

enum class BenchAlgo : u8 {
  kBenchAlgoVecadd = 0,
  kBenchAlgoReduce = 1,
  kBenchAlgoPrefix = 2,
  kBenchAlgoHist = 3,
  kBenchAlgoConV2D = 4,
  kBenchAlgoSpmv = 5,
  kBenchAlgoMatmul = 6,
  kBenchAlgoBlackscholes = 7,
  kBenchAlgoBsort = 8,
  kBenchAlgoNbody = 9,
  kBenchAlgoCount = 10
};

struct BenchOptions {
  BenchAlgo algo;
  u32 repeats;
  u32 seed;
  usize n;
  usize bins;
  usize nnz_per_row;
  usize ksize;
};

struct BenchResult {
  f64 total_time_ms;
  f64 calc_time_ms;
  f64 mem_time_ms;
  f64 gflops;
  f64 gbytes;
  f64 checksum;
  u64 bytes_moved;
  u64 flops;
  i32 status;
};

struct BenchEntry {
  const char* name;
  BenchAlgo algo;
};

using bench_get_name_fn = const char* (*)();
using bench_get_entries_fn = u32 (*)(const BenchEntry** out_entries);
using bench_run_fn = i32 (*)(const BenchOptions* opts, BenchResult* out_result);

struct BenchApi {
  bench_get_name_fn get_name;
  bench_get_entries_fn get_entries;
  bench_run_fn run;
};

extern "C" const BenchApi* bench_get_api();
using bench_get_api_fn = decltype(&bench_get_api);
