#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../common_abi/bench_abi.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace
{
  struct Library
  {
    std::string path;
#if defined(_WIN32)
    HMODULE handle = nullptr;
#else
    void *handle = nullptr;
#endif
    const BenchApi *api = nullptr;
  };

  static void print_result(const char *backend, const char *algo, const BenchOptions &opt,
                           const BenchResult &r)
  {
    std::printf(
      "backend=%s algo=%s n=%zu repeats=%u bins=%zu nnz=%zu ksize=%zu total_ms=%.3f calc_ms=%.3f "
      "mem_ms=%.3f gflops=%.3f gbytes=%.3f checksum=%.6f status=%d\n",
      backend, algo, opt.n, opt.repeats, opt.bins, opt.nnz_per_row, opt.ksize, r.total_time_ms,
      r.calc_time_ms, r.mem_time_ms, r.gflops, r.gbytes, r.checksum, r.status);
  }

  static bool load_library(Library &lib)
  {
#if defined(_WIN32)
    lib.handle = LoadLibraryA(lib.path.c_str());
    if (!lib.handle)
    {
      std::fprintf(stderr, "LoadLibrary failed for %s (error=%lu)\n", lib.path.c_str(),
                   GetLastError());
      return false;
    }
    auto get_api = reinterpret_cast<bench_get_api_fn>(GetProcAddress(lib.handle, "bench_get_api"));
#else
    dlerror();
    lib.handle = dlopen(lib.path.c_str(), RTLD_NOW);
    if (!lib.handle)
    {
      const char *err = dlerror();
      std::fprintf(stderr, "dlopen failed for %s: %s\n", lib.path.c_str(),
                   err ? err : "unknown error");
      return false;
    }
    auto get_api = reinterpret_cast<bench_get_api_fn>(dlsym(lib.handle, "bench_get_api"));
#endif
    if (!get_api)
    {
      std::fprintf(stderr, "Failed to find bench_get_api in %s\n", lib.path.c_str());
      return false;
    }
    lib.api = get_api();
    return lib.api != nullptr;
  }

  static void close_library(Library &lib)
  {
    if (!lib.handle)
    {
      return;
    }
#if defined(_WIN32)
    FreeLibrary(lib.handle);
#else
    dlclose(lib.handle);
#endif
    lib.handle = nullptr;
    lib.api = nullptr;
  }

  static BenchOptions default_options()
  {
    BenchOptions opt{};
    opt.algo = BENCH_ALGO_VECADD;
    opt.repeats = 5;
    opt.seed = 1234;
    opt.n = 0;
    opt.bins = 256;
    opt.nnz_per_row = 16;
    opt.ksize = 3;
    return opt;
  }

  static const char *algo_name(BenchAlgo algo)
  {
    switch (algo)
    {
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
} // namespace

int main(int argc, char **argv)
{
  std::vector<std::string> libs;
  for (int i = 1; i < argc; ++i)
  {
    libs.emplace_back(argv[i]);
  }

  if (libs.empty())
  {
    std::fprintf(stderr,
                 "Usage: bench_orchestrator <libcpu_ref.so> <libcpu_avx512.so> <libvkbench.so>\n");
    return 1;
  }

  for (const auto &path: libs)
  {
    Library lib;
    lib.path = path;
    if (!load_library(lib))
    {
      std::fprintf(stderr, "Failed to load %s\n", path.c_str());
      continue;
    }

    const char *backend = lib.api->get_name();
    const BenchEntry *entries = nullptr;
    u32 count = lib.api->get_entries(&entries);

    for (u32 i = 0; i < count; ++i)
    {
      const BenchEntry &entry = entries[i];
      BenchOptions opt = default_options();
      opt.algo = entry.algo;
      BenchResult result{};
      const int rc = lib.api->run(&opt, &result);
      if (rc != 0)
      {
        result.status = rc;
      }
      print_result(backend, entry.name ? entry.name : algo_name(entry.algo), opt, result);
    }

    close_library(lib);
  }

  return 0;
}
