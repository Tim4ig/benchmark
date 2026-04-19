#include "hybrid_runner.h"

#include "../../bench_settings.h"
#include "../../common_abi/bench_utils.hpp"
#include "../../vkbench/src/vk_common.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <immintrin.h>
#include <string>
#include <thread>
#include <vector>

namespace bench::hybrid {
namespace {
constexpr f32 kSoftening2 = 1.0e-4f;
constexpr double kDefaultMatmulCpuRatio = 0.55;
constexpr double kDefaultNbodyCpuRatio = 0.63;
const usize kThreadCount = std::max(1u, std::thread::hardware_concurrency());

struct Timing {
  f64 total_ms = 0.0;
  f64 avg_ms = 0.0;
};

template <typename Fn> f64 time_ms_fn(Fn&& fn) {
  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  fn();
  const auto end = clock::now();
  return std::chrono::duration_cast<std::chrono::duration<f64, std::milli>>(end - start).count();
}

struct NBodyPush {
  u32 source_n;
  u32 target_n;
  f32 softening2;
};

double clamp_ratio(const double value) {
  return std::clamp(value, 0.0, 1.0);
}

double parse_cpu_ratio_override() {
  const char* raw = std::getenv("HYBRID_CPU_RATIO");
  if (raw == nullptr) {
    return -1.0;
  }
  char* end = nullptr;
  const double value = std::strtod(raw, &end);
  if (end == raw) {
    return -1.0;
  }
  return clamp_ratio(value);
}

double cpu_ratio_for(const BenchAlgo algo) {
  const double override_ratio = parse_cpu_ratio_override();
  if (override_ratio >= 0.0) {
    return override_ratio;
  }
  switch (algo) {
    case BenchAlgo::kBenchAlgoMatmul:
      return kDefaultMatmulCpuRatio;
    case BenchAlgo::kBenchAlgoNbody:
      return kDefaultNbodyCpuRatio;
    default:
      return 0.50;
  }
}

std::string self_dir() {
  Dl_info info{};
  if (::dladdr(reinterpret_cast<const void*>(&self_dir), &info) && info.dli_fname != nullptr) {
    const std::string path(info.dli_fname);
    const auto pos = path.find_last_of('/');
    if (pos != std::string::npos) {
      return path.substr(0, pos);
    }
  }
  return ".";
}

VkDescriptorBufferInfo buffer_info(const VkBufferHandle& buf) {
  VkDescriptorBufferInfo info{};
  info.buffer = buf.buffer;
  info.offset = 0;
  info.range = buf.size;
  return info;
}

std::vector<VkDescriptorBufferInfo> make_bindings(const VkBufferHandle& dummy,
                                                  const std::vector<VkBufferHandle>& bufs) {
  std::vector<VkDescriptorBufferInfo> infos(6, buffer_info(dummy));
  for (usize i = 0; i < bufs.size() && i < infos.size(); ++i) {
    infos[i] = buffer_info(bufs[i]);
  }
  return infos;
}

template <typename Fn> void parallel_for_range(const usize begin, const usize end, Fn&& fn) {
  const usize count = end - begin;
  if (count == 0) {
    return;
  }
  const usize nt = std::min(count, kThreadCount);
  const usize chunk = (count + nt - 1) / nt;
  std::vector<std::thread> threads;
  threads.reserve(nt);
  for (usize t = 0; t < nt; ++t) {
    const usize chunk_begin = begin + t * chunk;
    const usize chunk_end = std::min(chunk_begin + chunk, end);
    if (chunk_begin < chunk_end) {
      threads.emplace_back(fn, chunk_begin, chunk_end);
    }
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

void nbody_range(
    const f32* px, const f32* py, const f32* mass, f32* fx, f32* fy, const usize source_n, const usize i0, const usize i1) {
  const __m512 veps2 = _mm512_set1_ps(kSoftening2);
  const __m512 half = _mm512_set1_ps(0.5f);
  const __m512 c15 = _mm512_set1_ps(1.5f);

  for (usize i = i0; i < i1; ++i) {
    const __m512 vxi = _mm512_set1_ps(px[i]);
    const __m512 vyi = _mm512_set1_ps(py[i]);
    __m512 vax = _mm512_setzero_ps();
    __m512 vay = _mm512_setzero_ps();

    usize j = 0;
    for (; j + 15 < source_n; j += 16) {
      const __m512 vmj_raw = _mm512_loadu_ps(mass + j);
      const __mmask16 self_mask = (i >= j && i < j + 16) ? static_cast<__mmask16>(~(1u << (i - j))) : 0xFFFFu;
      const __m512 vmj = _mm512_maskz_mov_ps(self_mask, vmj_raw);
      const __m512 dx = _mm512_sub_ps(_mm512_loadu_ps(px + j), vxi);
      const __m512 dy = _mm512_sub_ps(_mm512_loadu_ps(py + j), vyi);
      __m512 d2 = _mm512_fmadd_ps(dx, dx, _mm512_fmadd_ps(dy, dy, veps2));
      __m512 inv = _mm512_rsqrt14_ps(d2);
      inv = _mm512_mul_ps(inv, _mm512_fnmadd_ps(_mm512_mul_ps(d2, inv), _mm512_mul_ps(inv, half), c15));
      const __m512 inv3 = _mm512_mul_ps(inv, _mm512_mul_ps(inv, inv));
      const __m512 f = _mm512_mul_ps(vmj, inv3);
      vax = _mm512_fmadd_ps(f, dx, vax);
      vay = _mm512_fmadd_ps(f, dy, vay);
    }

    alignas(64) f32 tmp_ax[16];
    alignas(64) f32 tmp_ay[16];
    _mm512_storeu_ps(tmp_ax, vax);
    _mm512_storeu_ps(tmp_ay, vay);
    f32 ax = 0.0f;
    f32 ay = 0.0f;
    for (int k = 0; k < 16; ++k) {
      ax += tmp_ax[k];
      ay += tmp_ay[k];
    }

    for (; j < source_n; ++j) {
      if (j == i) {
        continue;
      }
      const f32 dx = px[j] - px[i];
      const f32 dy = py[j] - py[i];
      const f32 d2 = dx * dx + dy * dy + kSoftening2;
      const f32 inv = 1.0f / std::sqrt(d2);
      const f32 f = mass[j] * inv * inv * inv;
      ax += f * dx;
      ay += f * dy;
    }

    fx[i] = ax;
    fy[i] = ay;
  }
}

void dispatch_nbody(VkContext& ctx,
                    const VkPipelineResources& resources,
                    const VkPipelineBundle& pipe,
                    const VkDescriptorSet set,
                    const NBodyPush& push,
                    const u32 groups) {
  VK_CHECK(vkResetCommandBuffer(ctx.command_buffer, 0));

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(ctx.command_buffer, &begin_info));

  vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
  vkCmdBindDescriptorSets(
      ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0, 1, &set, 0, nullptr);
  vkCmdPushConstants(
      ctx.command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  vkCmdDispatch(ctx.command_buffer, groups, 1, 1);

  VK_CHECK(vkEndCommandBuffer(ctx.command_buffer));

  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &ctx.command_buffer;
  VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
  VK_CHECK(vkQueueWaitIdle(ctx.queue));
}

void matmul_rows(const f32* a, const f32* b, f32* c, const usize n, const usize row_begin, const usize row_end) {
  for (usize row = row_begin; row < row_end; ++row) {
    const usize row_offset = row * n;
    f32* c_row = c + row_offset;
    std::fill(c_row, c_row + n, 0.0f);
    for (usize k = 0; k < n; ++k) {
      const f32 aik = a[row_offset + k];
      const f32* b_row = b + k * n;
      const __m512 va = _mm512_set1_ps(aik);
      usize col = 0;
      for (; col + 15 < n; col += 16) {
        _mm512_storeu_ps(
            c_row + col, _mm512_fmadd_ps(va, _mm512_loadu_ps(b_row + col), _mm512_loadu_ps(c_row + col)));
      }
      for (; col < n; ++col) {
        c_row[col] += aik * b_row[col];
      }
    }
  }
}

void dispatch_matmul(VkContext& ctx,
                     const VkPipelineResources& resources,
                     const VkPipelineBundle& pipe,
                     const VkDescriptorSet set,
                     const u32 n,
                     const u32 group_x,
                     const u32 group_y) {
  VK_CHECK(vkResetCommandBuffer(ctx.command_buffer, 0));

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(ctx.command_buffer, &begin_info));

  vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
  vkCmdBindDescriptorSets(
      ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0, 1, &set, 0, nullptr);
  vkCmdPushConstants(ctx.command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n), &n);
  vkCmdDispatch(ctx.command_buffer, group_x, group_y, 1);

  VK_CHECK(vkEndCommandBuffer(ctx.command_buffer));

  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &ctx.command_buffer;
  VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
  VK_CHECK(vkQueueWaitIdle(ctx.queue));
}
} // namespace

HybridRunner::HybridRunner() {
  shader_dir_ = self_dir() + "/../vkbench/shaders";
  const bool enable_validation = std::getenv("VKBENCH_VALIDATION") != nullptr;
  if (!vk_init(ctx_, enable_validation)) {
    return;
  }

  resources_ = vk_create_pipeline_resources(ctx_);
  dummy_ = vk_create_buffer(ctx_,
                            4,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (dummy_.buffer == VK_NULL_HANDLE) {
    vk_destroy_pipeline_resources(ctx_, resources_);
    vk_destroy(ctx_);
    return;
  }
  u32 zero = 0;
  vk_write_buffer(ctx_, dummy_, &zero, sizeof(zero));

  matmul_pipeline_ = vk_create_compute_pipeline(ctx_, resources_, shader_dir_ + "/matmul.spv");
  nbody_pipeline_ = vk_create_compute_pipeline(ctx_, resources_, shader_dir_ + "/nbody.spv");
  if (matmul_pipeline_.pipeline == VK_NULL_HANDLE || nbody_pipeline_.pipeline == VK_NULL_HANDLE) {
    vk_destroy_pipeline(ctx_, matmul_pipeline_);
    vk_destroy_buffer(ctx_, dummy_);
    vk_destroy_pipeline_resources(ctx_, resources_);
    vk_destroy(ctx_);
    return;
  }

  ready_ = true;
}

HybridRunner::~HybridRunner() {
  if (!ready_) {
    return;
  }
  vk_destroy_pipeline(ctx_, matmul_pipeline_);
  vk_destroy_pipeline(ctx_, nbody_pipeline_);
  vk_destroy_buffer(ctx_, dummy_);
  vk_destroy_pipeline_resources(ctx_, resources_);
  vk_destroy(ctx_);
}

BenchResult HybridRunner::run(const BenchOptions& options) {
  const BenchOptions opts = bench::resolve_options(options);
  if (!ready_) {
    BenchResult result{};
    result.status = -3;
    return result;
  }
  switch (opts.algo) {
    case BenchAlgo::kBenchAlgoMatmul:
      return run_matmul(opts);
    case BenchAlgo::kBenchAlgoNbody:
      return run_nbody(opts);
    default: {
      BenchResult result{};
      result.status = -2;
      return result;
    }
  }
}

BenchResult HybridRunner::run_matmul(const BenchOptions& options) {
  const usize n = options.n;
  const usize size = n * n;

  auto a = bench::make_random(size, options.seed);
  auto b = bench::make_random(size, options.seed + 1u);
  std::vector<f32> c(size, 0.0f);
  std::vector<f32> gpu_c(size, 0.0f);

  auto ba = vk_create_buffer(ctx_,
                             size * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bb = vk_create_buffer(ctx_,
                             size * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bc = vk_create_buffer(ctx_,
                             size * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  if (ba.buffer == VK_NULL_HANDLE || bb.buffer == VK_NULL_HANDLE || bc.buffer == VK_NULL_HANDLE) {
    vk_destroy_buffer(ctx_, ba);
    vk_destroy_buffer(ctx_, bb);
    vk_destroy_buffer(ctx_, bc);
    BenchResult result{};
    result.status = -4;
    return result;
  }

  const f64 mem_h2d_ms = time_ms_fn([&] {
    vk_write_buffer(ctx_, ba, a.data(), ba.size);
    vk_write_buffer(ctx_, bb, b.data(), bb.size);
  });

  VkDescriptorSet set = vk_allocate_descriptor_set(ctx_, resources_);
  const auto infos = make_bindings(dummy_, {ba, bb, bc});
  vk_update_descriptor_set(ctx_, set, infos);

  const u32 group_x = static_cast<u32>((n + 15) / 16);
  const u32 push_n = static_cast<u32>(n);

  // Adaptive ratio: start with heuristic default, refine each repeat.
  // Override env var bypasses adaptation.
  const bool has_override = parse_cpu_ratio_override() >= 0.0;
  double ratio = cpu_ratio_for(BenchAlgo::kBenchAlgoMatmul);

  using clock = std::chrono::steady_clock;
  f64 total_ms = 0.0;
  usize last_gpu_rows = 0;

  for (u32 repeat = 0; repeat < options.repeats; ++repeat) {
    usize cpu_rows = static_cast<usize>(std::round(static_cast<double>(n) * ratio));
    if (n > 1) {
      cpu_rows = std::min(std::max(cpu_rows, usize{1}), n - 1);
    }
    const usize gpu_rows = n - cpu_rows;
    last_gpu_rows = gpu_rows;
    const u32 group_y = static_cast<u32>((gpu_rows + 15) / 16);

    std::atomic<f64> cpu_time_ms{0.0};
    const auto rep_start = clock::now();

    std::thread cpu_thread([&, gpu_rows_cap = gpu_rows] {
      const auto t0 = clock::now();
      parallel_for_range(gpu_rows_cap, n, [&](const usize begin, const usize end) {
        matmul_rows(a.data(), b.data(), c.data(), n, begin, end);
      });
      cpu_time_ms.store(
          std::chrono::duration_cast<std::chrono::duration<f64, std::milli>>(clock::now() - t0).count());
    });

    const auto gpu_t0 = clock::now();
    dispatch_matmul(ctx_, resources_, matmul_pipeline_, set, push_n, group_x, group_y);
    const f64 gpu_time_ms =
        std::chrono::duration_cast<std::chrono::duration<f64, std::milli>>(clock::now() - gpu_t0).count();

    cpu_thread.join();
    total_ms +=
        std::chrono::duration_cast<std::chrono::duration<f64, std::milli>>(clock::now() - rep_start).count();

    // Adaptive update: equalize CPU and GPU throughput rates.
    // Skip update when timings are below 1 ms — clock noise dominates at sub-ms
    // granularity and causes the ratio to oscillate rather than converge.
    if (!has_override) {
      const f64 t_cpu = cpu_time_ms.load();
      constexpr f64 kMinAdaptiveMs = 1.0;
      if (t_cpu > kMinAdaptiveMs && gpu_time_ms > kMinAdaptiveMs) {
        const double r_cpu = static_cast<double>(cpu_rows) / t_cpu;
        const double r_gpu = static_cast<double>(gpu_rows) / gpu_time_ms;
        const double target = r_cpu / (r_cpu + r_gpu);
        ratio = clamp_ratio(0.9 * ratio + 0.1 * target);
      }
    }
  }

  const f64 mem_d2h_ms = time_ms_fn([&] { vk_read_buffer(ctx_, bc, gpu_c.data(), bc.size); });
  if (last_gpu_rows > 0) {
    std::copy_n(gpu_c.data(), last_gpu_rows * n, c.data());
  }

  VK_CHECK(vkFreeDescriptorSets(ctx_.device, resources_.descriptor_pool, 1, &set));
  vk_destroy_buffer(ctx_, ba);
  vk_destroy_buffer(ctx_, bb);
  vk_destroy_buffer(ctx_, bc);

  BenchResult result{};
  result.status = 0;
  result.total_time_ms = mem_h2d_ms + total_ms + mem_d2h_ms;
  result.calc_time_ms = options.repeats > 0 ? total_ms / static_cast<f64>(options.repeats) : 0.0;
  result.mem_time_ms = mem_h2d_ms + mem_d2h_ms;
  result.flops = static_cast<u64>(2.0 * static_cast<f64>(n) * static_cast<f64>(n) * static_cast<f64>(n));
  result.bytes_moved = static_cast<u64>(3 * size * sizeof(f32));
  result.gflops = result.calc_time_ms > 0.0 ? static_cast<f64>(result.flops) / (result.calc_time_ms * 1.0e6) : 0.0;
  result.gbytes =
      result.calc_time_ms > 0.0 ? static_cast<f64>(result.bytes_moved) / (result.calc_time_ms * 1.0e6) : 0.0;
  result.checksum = bench::checksum(c.data(), size);
  return result;
}

BenchResult HybridRunner::run_nbody(const BenchOptions& options) {
  const usize n = options.n;

  auto px = bench::make_random(n, options.seed, -100.0f, 100.0f);
  auto py = bench::make_random(n, options.seed + 1u, -100.0f, 100.0f);
  auto mass = bench::make_random(n, options.seed + 2u, 0.1f, 2.0f);
  std::vector<f32> fx(n, 0.0f);
  std::vector<f32> fy(n, 0.0f);

  auto bpx = vk_create_buffer(ctx_,
                              n * sizeof(f32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bpy = vk_create_buffer(ctx_,
                              n * sizeof(f32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bmass = vk_create_buffer(ctx_,
                                n * sizeof(f32),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  // Pre-allocate full n-size output buffers to allow adaptive target_n changes.
  auto bfx = vk_create_buffer(ctx_,
                              n * sizeof(f32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bfy = vk_create_buffer(ctx_,
                              n * sizeof(f32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  if (bpx.buffer == VK_NULL_HANDLE || bpy.buffer == VK_NULL_HANDLE || bmass.buffer == VK_NULL_HANDLE ||
      bfx.buffer == VK_NULL_HANDLE || bfy.buffer == VK_NULL_HANDLE) {
    vk_destroy_buffer(ctx_, bpx);
    vk_destroy_buffer(ctx_, bpy);
    vk_destroy_buffer(ctx_, bmass);
    vk_destroy_buffer(ctx_, bfx);
    vk_destroy_buffer(ctx_, bfy);
    BenchResult result{};
    result.status = -4;
    return result;
  }

  const f64 mem_h2d_ms = time_ms_fn([&] {
    vk_write_buffer(ctx_, bpx, px.data(), bpx.size);
    vk_write_buffer(ctx_, bpy, py.data(), bpy.size);
    vk_write_buffer(ctx_, bmass, mass.data(), bmass.size);
  });

  VkDescriptorSet set = vk_allocate_descriptor_set(ctx_, resources_);
  const auto infos = make_bindings(dummy_, {bpx, bpy, bmass, bfx, bfy});
  vk_update_descriptor_set(ctx_, set, infos);

  // Adaptive ratio: start with heuristic default, refine each repeat.
  const bool has_override = parse_cpu_ratio_override() >= 0.0;
  double ratio = cpu_ratio_for(BenchAlgo::kBenchAlgoNbody);

  using clock = std::chrono::steady_clock;
  f64 total_ms = 0.0;
  usize last_gpu_n = 0;

  for (u32 repeat = 0; repeat < options.repeats; ++repeat) {
    usize cpu_n = static_cast<usize>(std::round(static_cast<double>(n) * ratio));
    if (n > 1) {
      cpu_n = std::min(std::max(cpu_n, usize{1}), n - 1);
    }
    const usize gpu_n = n - cpu_n;
    last_gpu_n = gpu_n;

    const NBodyPush push{static_cast<u32>(n), static_cast<u32>(gpu_n), kSoftening2};
    const u32 groups = static_cast<u32>((gpu_n + 255) / 256);

    std::atomic<f64> cpu_time_ms{0.0};
    const auto rep_start = clock::now();

    std::thread cpu_thread([&, gpu_n_cap = gpu_n] {
      const auto t0 = clock::now();
      parallel_for_range(gpu_n_cap, n, [&](const usize begin, const usize end) {
        nbody_range(px.data(), py.data(), mass.data(), fx.data(), fy.data(), n, begin, end);
      });
      cpu_time_ms.store(
          std::chrono::duration_cast<std::chrono::duration<f64, std::milli>>(clock::now() - t0).count());
    });

    const auto gpu_t0 = clock::now();
    dispatch_nbody(ctx_, resources_, nbody_pipeline_, set, push, groups);
    const f64 gpu_time_ms =
        std::chrono::duration_cast<std::chrono::duration<f64, std::milli>>(clock::now() - gpu_t0).count();

    cpu_thread.join();
    total_ms +=
        std::chrono::duration_cast<std::chrono::duration<f64, std::milli>>(clock::now() - rep_start).count();

    // Adaptive update: equalize CPU and GPU throughput rates.
    // Skip update when timings are below 1 ms — clock noise dominates at sub-ms
    // granularity and causes the ratio to oscillate rather than converge.
    if (!has_override) {
      const f64 t_cpu = cpu_time_ms.load();
      constexpr f64 kMinAdaptiveMs = 1.0;
      if (t_cpu > kMinAdaptiveMs && gpu_time_ms > kMinAdaptiveMs) {
        const double r_cpu = static_cast<double>(cpu_n) / t_cpu;
        const double r_gpu = static_cast<double>(gpu_n) / gpu_time_ms;
        const double target = r_cpu / (r_cpu + r_gpu);
        ratio = clamp_ratio(0.9 * ratio + 0.1 * target);
      }
    }
  }

  // Read back the GPU portion computed in the last iteration.
  const f64 mem_d2h_ms = time_ms_fn([&] {
    vk_read_buffer(ctx_, bfx, fx.data(), last_gpu_n * sizeof(f32));
    vk_read_buffer(ctx_, bfy, fy.data(), last_gpu_n * sizeof(f32));
  });
  VK_CHECK(vkFreeDescriptorSets(ctx_.device, resources_.descriptor_pool, 1, &set));

  vk_destroy_buffer(ctx_, bpx);
  vk_destroy_buffer(ctx_, bpy);
  vk_destroy_buffer(ctx_, bmass);
  vk_destroy_buffer(ctx_, bfx);
  vk_destroy_buffer(ctx_, bfy);

  BenchResult result{};
  result.status = 0;
  result.total_time_ms = mem_h2d_ms + total_ms + mem_d2h_ms;
  result.calc_time_ms = options.repeats > 0 ? total_ms / static_cast<f64>(options.repeats) : 0.0;
  result.mem_time_ms = mem_h2d_ms + mem_d2h_ms;
  result.flops = static_cast<u64>(20.0 * static_cast<f64>(n) * static_cast<f64>(n - 1));
  result.bytes_moved = static_cast<u64>(5 * n * sizeof(f32));
  result.gflops = result.calc_time_ms > 0.0 ? static_cast<f64>(result.flops) / (result.calc_time_ms * 1.0e6) : 0.0;
  result.gbytes =
      result.calc_time_ms > 0.0 ? static_cast<f64>(result.bytes_moved) / (result.calc_time_ms * 1.0e6) : 0.0;
  result.checksum = bench::checksum(fx.data(), n) + bench::checksum(fy.data(), n);
  return result;
}
} // namespace bench::hybrid
