#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#include "../bench_settings.h"
#include "../common_abi/bench_abi.h"

namespace {

struct Library {
  std::string path;
  void* handle = nullptr;
  const BenchApi* api = nullptr;
};

void PrintResult(const char* backend, const char* algo, const BenchOptions& opt,
                 const BenchResult& result) {
  std::printf(
      "backend=%s algo=%s n=%zu repeats=%u bins=%zu nnz=%zu ksize=%zu total_ms=%.3f calc_ms=%.3f "
      "mem_ms=%.3f gflops=%.3f gbytes=%.3f checksum=%.6f status=%d\n",
      backend, algo, opt.n, opt.repeats, opt.bins, opt.nnz_per_row, opt.ksize,
      result.total_time_ms, result.calc_time_ms, result.mem_time_ms, result.gflops, result.gbytes,
      result.checksum, result.status);
}

bool LoadLibrary(Library& lib) {
  dlerror();
  lib.handle = dlopen(lib.path.c_str(), RTLD_NOW);
  if (!lib.handle) {
    const char* err = dlerror();
    std::fprintf(stderr, "dlopen failed for %s: %s\n", lib.path.c_str(),
                 err ? err : "unknown error");
    return false;
  }
  auto get_api = reinterpret_cast<bench_get_api_fn>(dlsym(lib.handle, "bench_get_api"));
  if (!get_api) {
    std::fprintf(stderr, "Failed to find bench_get_api in %s\n", lib.path.c_str());
    return false;
  }
  lib.api = get_api();
  return lib.api != nullptr;
}

void CloseLibrary(Library& lib) {
  if (!lib.handle) {
    return;
  }
  dlclose(lib.handle);
  lib.handle = nullptr;
  lib.api = nullptr;
}

const char* AlgoName(BenchAlgo algo) {
  switch (algo) {
    case BENCH_ALGO_VECADD:
      return "vecadd";
    case BENCH_ALGO_REDUCE:
      return "reduce";
    case BENCH_ALGO_PREFIX:
      return "prefix";
    case BENCH_ALGO_HIST:
      return "hist";
    case BENCH_ALGO_CONV2D:
      return "conv2d";
    case BENCH_ALGO_SPMV:
      return "spmv";
    case BENCH_ALGO_MATMUL:
      return "matmul";
    default:
      return "unknown";
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> libs;
  for (int i = 1; i < argc; ++i) {
    libs.emplace_back(argv[i]);
  }

  if (libs.empty()) {
    std::fprintf(stderr,
                 "Usage: bench_orchestrator <libcpu_ref.so> <libcpu_avx512.so> <libvkbench.so>\n");
    return 1;
  }

  for (const auto& path : libs) {
    Library lib;
    lib.path = path;
    if (!LoadLibrary(lib)) {
      std::fprintf(stderr, "Failed to load %s\n", path.c_str());
      continue;
    }

    const char* backend = lib.api->get_name();
    const BenchEntry* entries = nullptr;
    u32 count = lib.api->get_entries(&entries);

    for (u32 i = 0; i < count; ++i) {
      const BenchEntry& entry = entries[i];
      BenchOptions opt = bench::DefaultOptions();
      opt.algo = entry.algo;
      BenchResult result{};
      const int rc = lib.api->run(&opt, &result);
      if (rc != 0) {
        result.status = rc;
      }
      PrintResult(backend, entry.name ? entry.name : AlgoName(entry.algo), opt, result);
    }

    CloseLibrary(lib);
  }

  return 0;
}
