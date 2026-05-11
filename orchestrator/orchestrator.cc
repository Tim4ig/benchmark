#include "../bench_settings.h"
#include "../common_abi/bench_abi.h"
#include "build_info.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Power measurement via sysfs
// ---------------------------------------------------------------------------

static std::string find_hwmon_power_path(const char* driver_name, const char* sensor) {
  for (int i = 0; i < 32; ++i) {
    char name_path[128];
    std::snprintf(name_path, sizeof(name_path), "/sys/class/hwmon/hwmon%d/name", i);
    std::ifstream nf(name_path);
    if (!nf.is_open())
      continue;
    std::string name;
    nf >> name;
    if (name == driver_name) {
      char sensor_path[128];
      std::snprintf(sensor_path, sizeof(sensor_path), "/sys/class/hwmon/hwmon%d/%s", i, sensor);
      // Verify the file exists
      std::ifstream sf(sensor_path);
      if (sf.is_open())
        return sensor_path;
    }
  }
  return {};
}

// Resolved once at startup
static const std::string gpu_power_path = find_hwmon_power_path("amdgpu", "power1_average");
static const char* const rapl_path      = "/sys/class/powercap/intel-rapl:0/energy_uj";

static int64_t read_rapl_uj() {
  std::ifstream f(rapl_path);
  if (!f.is_open())
    return -1;
  int64_t v = -1;
  f >> v;
  return f ? v : -1;
}

static double read_gpu_power_uw() {
  if (gpu_power_path.empty())
    return -1.0;
  std::ifstream f(gpu_power_path);
  if (!f.is_open())
    return -1.0;
  double v = -1.0;
  f >> v;
  return f ? v : -1.0;
}

struct PowerSample {
  double watts_cpu = -1.0; // from RAPL (package energy delta / time)
  double watts_gpu = -1.0; // from amdgpu power1_average (averaged over benchmark duration)
};

// Run work() while sampling GPU power in a background thread and reading RAPL
// energy before/after. Returns averaged power over the benchmark duration.
static PowerSample measure_power(const std::function<void()>& work) {
  std::atomic<bool> stop{false};
  std::vector<double> gpu_samples;
  gpu_samples.reserve(256);

  // Background thread samples GPU power every 10 ms.
  // First sample is taken immediately so short benchmarks get at least one reading.
  std::thread sampler([&]() {
    while (true) {
      const double p = read_gpu_power_uw();
      if (p >= 0.0)
        gpu_samples.push_back(p);
      if (stop.load(std::memory_order_relaxed))
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });

  const int64_t rapl_before = read_rapl_uj();
  const auto t_before       = std::chrono::steady_clock::now();

  work();

  const auto t_after   = std::chrono::steady_clock::now();
  const int64_t rapl_after = read_rapl_uj();
  stop.store(true, std::memory_order_relaxed);
  sampler.join();

  PowerSample ps;

  // CPU: RAPL energy delta / elapsed time
  if (rapl_before >= 0 && rapl_after > rapl_before) {
    const double delta_uj = static_cast<double>(rapl_after - rapl_before);
    const double delta_s  = std::chrono::duration<double>(t_after - t_before).count();
    if (delta_s > 0.0)
      ps.watts_cpu = (delta_uj * 1e-6) / delta_s;
  }

  // GPU: average of µW samples -> W
  if (!gpu_samples.empty()) {
    double sum = 0.0;
    for (const double s : gpu_samples)
      sum += s;
    ps.watts_gpu = (sum / static_cast<double>(gpu_samples.size())) * 1e-6;
  }

  return ps;
}

// ---------------------------------------------------------------------------
// Library loading
// ---------------------------------------------------------------------------

struct Library {
  std::string path;
  void* handle = nullptr;
  const BenchApi* api = nullptr;
};

void print_result(const char*        backend,
                  const char*        algo,
                  const BenchOptions& opt,
                  const BenchResult&  result,
                  const PowerSample&  pwr) {
  std::printf("backend=%s algo=%s n=%zu repeats=%u bins=%zu nnz=%zu ksize=%zu "
              "total_ms=%.3f calc_ms=%.3f mem_ms=%.3f gflops=%.4f gbytes=%.4f "
              "checksum=%.6f watts_cpu=%.2f watts_gpu=%.2f status=%d\n",
              backend,
              algo,
              opt.n,
              opt.repeats,
              opt.bins,
              opt.nnz_per_row,
              opt.ksize,
              result.total_time_ms,
              result.calc_time_ms,
              result.mem_time_ms,
              result.gflops,
              result.gbytes,
              result.checksum,
              pwr.watts_cpu,
              pwr.watts_gpu,
              result.status);
}

void print_csv(std::FILE*          fd,
               const char*         backend,
               const char*         algo,
               const BenchOptions& opt,
               const BenchResult&  result,
               const PowerSample&  pwr) {
  std::fprintf(fd,
               "%s,%s,%zu,%u,%zu,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%llu,%llu,%.2f,%.2f,%d\n",
               backend,
               algo,
               opt.n,
               opt.repeats,
               opt.bins,
               opt.nnz_per_row,
               opt.ksize,
               result.total_time_ms,
               result.calc_time_ms,
               result.mem_time_ms,
               result.gflops,
               result.gbytes,
               result.checksum,
               static_cast<unsigned long long>(result.flops),
               static_cast<unsigned long long>(result.bytes_moved),
               pwr.watts_cpu,
               pwr.watts_gpu,
               result.status);
}

bool load_library(Library& lib) {
  dlerror();
  lib.handle = dlopen(lib.path.c_str(), RTLD_NOW);
  if (lib.handle == nullptr) {
    const char* err = dlerror();
    std::fprintf(stderr, "dlopen failed for %s: %s\n", lib.path.c_str(), err != nullptr ? err : "unknown error");
    return false;
  }
  const auto get_api = reinterpret_cast<bench_get_api_fn>(dlsym(lib.handle, "bench_get_api"));
  if (get_api == nullptr) {
    std::fprintf(stderr, "Failed to find bench_get_api in %s\n", lib.path.c_str());
    return false;
  }
  lib.api = get_api();
  return lib.api != nullptr;
}

void close_library(Library& lib) {
  if (lib.handle == nullptr)
    return;
  dlclose(lib.handle);
  lib.handle = nullptr;
  lib.api    = nullptr;
}

const char* algo_name(BenchAlgo algo) {
  switch (algo) {
    case BenchAlgo::kBenchAlgoVecadd:       return "vecadd";
    case BenchAlgo::kBenchAlgoReduce:       return "reduce";
    case BenchAlgo::kBenchAlgoPrefix:       return "prefix";
    case BenchAlgo::kBenchAlgoHist:         return "hist";
    case BenchAlgo::kBenchAlgoConV2D:       return "conv2d";
    case BenchAlgo::kBenchAlgoSpmv:         return "spmv";
    case BenchAlgo::kBenchAlgoMatmul:       return "matmul";
    case BenchAlgo::kBenchAlgoBlackscholes: return "blackscholes";
    case BenchAlgo::kBenchAlgoBsort:        return "bsort";
    case BenchAlgo::kBenchAlgoNbody:        return "nbody";
    default:                                return "unknown";
  }
}

void print_usage() {
  std::fprintf(stderr,
               "Usage: bench_orchestrator [--csv <file.csv>] [--repeats N] <lib1.so> [lib2.so ...]\n"
               "\n"
               "Options:\n"
               "  --csv <file>    Write results to CSV file in addition to stdout\n"
               "  --repeats N     Override number of benchmark repetitions (default: %u)\n"
               "\n"
               "Power measurement:\n"
               "  CPU watts: RAPL via %s (needs read permission; run:\n"
               "             sudo chmod o+r /sys/class/powercap/intel-rapl*/energy_uj)\n"
               "  GPU watts: amdgpu hwmon power1_average (%s)\n"
               "\n"
               "Example:\n"
               "  bench_orchestrator --csv results.csv --repeats 20 \\\n"
               "                     cpu_ref/libcpu_ref.so vkbench/libvkbench.so\n",
               bench::kDefaultRepeats,
               rapl_path,
               gpu_power_path.empty() ? "(not found)" : gpu_power_path.c_str());
}

} // namespace

int main(const int argc, char** argv) {
  std::string csv_path;
  std::vector<std::string> libs;
  u32 repeats_override = 0;

  for (auto i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
      csv_path = argv[++i];
    } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
      repeats_override = static_cast<u32>(std::atoi(argv[++i]));
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage();
      return 0;
    } else {
      libs.emplace_back(argv[i]);
    }
  }

  if (libs.empty()) {
    print_usage();
    return 1;
  }

  // Report power measurement availability
  std::fprintf(stderr, "[power] RAPL CPU: %s\n",
               read_rapl_uj() >= 0 ? "available" : "unavailable (needs: sudo chmod o+r /sys/class/powercap/intel-rapl*/energy_uj)");
  std::fprintf(stderr, "[power] GPU hwmon: %s\n",
               gpu_power_path.empty() ? "not found" : gpu_power_path.c_str());

  // Open CSV file if requested
  std::FILE* csv_file = nullptr;
  if (!csv_path.empty()) {
    csv_file = std::fopen(csv_path.c_str(), "w");
    if (csv_file == nullptr) {
      std::fprintf(stderr, "Cannot open CSV file: %s\n", csv_path.c_str());
      return 1;
    }
    std::fprintf(csv_file, "# git_hash=%s\n", BUILD_GIT_HASH);
    std::fprintf(csv_file, "# build_ts=%s\n", BUILD_TIMESTAMP);
    std::fprintf(csv_file, "# compiler=%s\n", BUILD_CXX_COMPILER);
    std::fprintf(csv_file,
                 "backend,algo,n,repeats,bins,nnz_per_row,ksize,"
                 "total_ms,calc_ms,mem_ms,gflops,gbytes,checksum,flops,bytes_moved,"
                 "watts_cpu,watts_gpu,status\n");
  }
  std::fprintf(stderr,
               "[build] git_hash=%s build_ts=%s compiler=%s\n",
               BUILD_GIT_HASH,
               BUILD_TIMESTAMP,
               BUILD_CXX_COMPILER);

  for (const auto& path : libs) {
    Library lib;
    lib.path = path;
    if (!load_library(lib)) {
      std::fprintf(stderr, "Failed to load %s\n", path.c_str());
      continue;
    }

    const char* backend       = lib.api->get_name();
    const BenchEntry* entries = nullptr;
    const auto count          = lib.api->get_entries(&entries);

    for (u32 i = 0; i < count; ++i) {
      const auto& [name, algo] = entries[i];
      BenchOptions opt = bench::default_options();
      opt.algo         = algo;
      bench::apply_algorithm_defaults(opt);

      BenchResult result{};
      PowerSample pwr;

      if (repeats_override > 0) {
        // Manual mode: short fixed warmup then measured run.
        opt.repeats = std::max(3u, repeats_override / 4u);
        BenchResult warmup_result{};
        lib.api->run(&opt, &warmup_result);

        opt.repeats = repeats_override;
        pwr = measure_power([&]() {
          if (const auto rc = lib.api->run(&opt, &result); rc != 0)
            result.status = rc;
        });
      } else {
        // Time-based mode: calibrate → warmup ~4.5s → re-estimate on warmed state
        // → measure ~5.0s.
        constexpr double kWarmupMs  = 4500.0;
        constexpr double kMeasureMs = 5000.0;
        constexpr u32    kMinReps   = 5;
        constexpr u32    kMaxReps   = 200000;
        constexpr u32    kCalibrationReps = 3;

        // Step 1: calibration — a short cold estimate to size the warm-up run.
        opt.repeats = kCalibrationReps;
        BenchResult cal{};
        lib.api->run(&opt, &cal);
        const double iter_ms = (cal.status == 0 && cal.calc_time_ms > 0.0)
                                 ? cal.calc_time_ms
                                 : 50.0; // safe fallback

        const u32 warmup_reps = std::clamp(static_cast<u32>(std::llround(kWarmupMs / iter_ms)), kMinReps, kMaxReps);

        // Step 2: warm-up (heats caches, CPU boost, GPU steady state — not measured).
        opt.repeats = warmup_reps;
        BenchResult warmup{};
        lib.api->run(&opt, &warmup);

        const double warm_iter_ms = (warmup.status == 0 && warmup.calc_time_ms > 0.0)
                                      ? warmup.calc_time_ms
                                      : iter_ms;
        const u32 measure_reps =
            std::clamp(static_cast<u32>(std::llround(kMeasureMs / warm_iter_ms)), kMinReps, kMaxReps);

        std::fprintf(stderr, "[timing] %s/%s  cold_iter=%.3fms  warmup=%u  warm_iter=%.3fms  measure=%u\n",
                     backend, algo_name(algo), iter_ms, warmup_reps, warm_iter_ms, measure_reps);

        // Step 3: measured run.
        opt.repeats = measure_reps;
        pwr = measure_power([&]() {
          if (const auto rc = lib.api->run(&opt, &result); rc != 0)
            result.status = rc;
        });
      }

      const char* aname = name != nullptr ? name : algo_name(algo);
      print_result(backend, aname, opt, result, pwr);
      if (csv_file != nullptr)
        print_csv(csv_file, backend, aname, opt, result, pwr);
    }

    close_library(lib);
  }

  if (csv_file != nullptr) {
    std::fclose(csv_file);
    std::fprintf(stdout, "[orchestrator] CSV written to: %s\n", csv_path.c_str());
  }

  return 0;
}
