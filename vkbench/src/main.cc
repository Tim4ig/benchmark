#include "../../bench_settings.h"
#include "../../common_abi/bench_abi.h"
#include "../../common_abi/types.h"
#include "vk_buffer.hpp"
#include "vk_common.hpp"
#include "vk_context.hpp"
#include "vk_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

struct Timing {
  f64 total_ms = 0.0;
  f64 avg_ms = 0.0;
};

static std::string shader_dir() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&shader_dir), &info) != 0 && info.dli_fname) {
    std::string path(info.dli_fname);
    const std::string::size_type pos = path.find_last_of('/');
    if (pos != std::string::npos) {
      path.resize(pos);
      path += "/shaders";
      return path;
    }
  }
  return "./shaders";
}

static std::vector<f32> make_random(usize n, u32 seed, f32 lo = -1.0f, f32 hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<f32> dist(lo, hi);
  std::vector<f32> data(n);
  for (usize i = 0; i < n; ++i) {
    data[i] = dist(rng);
  }
  return data;
}

static std::vector<u32> make_random_u32(usize n, u32 seed, u32 max_value) {
  std::mt19937 rng(seed);
  const u32 hi = max_value > 0 ? max_value - 1u : 0u;
  std::uniform_int_distribution<u32> dist(0u, hi);
  std::vector<u32> data(n);
  for (usize i = 0; i < n; ++i) {
    data[i] = dist(rng);
  }
  return data;
}

struct CSRMatrix {
  u32 rows = 0;
  u32 cols = 0;
  std::vector<u32> row_ptr;
  std::vector<u32> col_idx;
  std::vector<f32> values;
};

static CSRMatrix make_csr_matrix(u32 rows, u32 cols, u32 nnz_per_row, u32 seed) {
  CSRMatrix mat;
  mat.rows = rows;
  mat.cols = cols;
  const u32 nnz_row = std::min(nnz_per_row, cols == 0 ? 1u : cols);
  const u32 nnz_total = rows * nnz_row;
  mat.row_ptr.resize(static_cast<usize>(rows) + 1u);
  mat.col_idx.resize(nnz_total);
  mat.values.resize(nnz_total);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);

  for (u32 row = 0; row < rows; ++row) {
    const u32 base = row * nnz_row;
    mat.row_ptr[row] = base;
    for (u32 k = 0; k < nnz_row; ++k) {
      const u32 idx = base + k;
      const u32 col = (row * 1315423911u + k * 2654435761u) % (cols == 0 ? 1u : cols);
      mat.col_idx[idx] = col;
      mat.values[idx] = dist(rng);
    }
  }
  mat.row_ptr[rows] = nnz_total;
  return mat;
}

static std::vector<f32> make_kernel(usize ksize, u32 seed) {
  const usize k = ksize == 0 ? 1 : ksize;
  auto kernel = make_random(k * k, seed);
  f32 sum = 0.0f;
  for (f32 v : kernel) {
    sum += v;
  }
  if (sum == 0.0f) {
    return kernel;
  }
  for (f32& v : kernel) {
    v /= sum;
  }
  return kernel;
}

static f64 checksum(const std::vector<f32>& data) {
  f64 sum = 0.0;
  for (f32 v : data) {
    sum += static_cast<f64>(v);
  }
  return sum;
}

static f64 checksum_u32(const std::vector<u32>& data) {
  f64 sum = 0.0;
  for (u32 v : data) {
    sum += static_cast<f64>(v);
  }
  return sum;
}

static Timing measure_dispatch(VkContext& ctx,
                               VkPipeline pipeline,
                               VkPipelineLayout layout,
                               VkDescriptorSet set,
                               const void* push_data,
                               usize push_size,
                               u32 group_x,
                               u32 group_y,
                               u32 group_z,
                               i32 repeats) {
  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  for (i32 i = 0; i < repeats; ++i) {
    VK_CHECK(vkResetCommandBuffer(ctx.command_buffer, 0));

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(ctx.command_buffer, &begin_info));

    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    if (push_data && push_size > 0) {
      vkCmdPushConstants(ctx.command_buffer,
                         layout,
                         VK_SHADER_STAGE_COMPUTE_BIT,
                         0,
                         static_cast<u32>(push_size),
                         push_data);
    }
    vkCmdDispatch(ctx.command_buffer, group_x, group_y, group_z);
    VK_CHECK(vkEndCommandBuffer(ctx.command_buffer));

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &ctx.command_buffer;
    VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(ctx.queue));
  }
  const auto end = clock::now();
  const f64 total_ms = std::chrono::duration_cast<std::chrono::duration<f64, std::milli>>(end - start).count();
  Timing t;
  t.total_ms = total_ms;
  t.avg_ms = repeats > 0 ? total_ms / static_cast<f64>(repeats) : 0.0;
  return t;
}

static VkDescriptorBufferInfo buffer_info(const VkBufferHandle& buf) {
  VkDescriptorBufferInfo info{};
  info.buffer = buf.buffer;
  info.offset = 0;
  info.range = buf.size;
  return info;
}

static std::string shader_path(const std::string& name) {
  std::string path = shader_dir();
  if (!path.empty() && path.back() != '/') {
    path += '/';
  }
  path += name;
  path += ".spv";
  return path;
}

static std::vector<VkDescriptorBufferInfo> make_bindings(const VkBufferHandle& dummy,
                                                         const std::vector<VkBufferHandle>& bufs) {
  std::vector<VkDescriptorBufferInfo> infos(6, buffer_info(dummy));
  for (usize i = 0; i < bufs.size() && i < infos.size(); ++i) {
    infos[i] = buffer_info(bufs[i]);
  }
  return infos;
}

static void run_vecadd(VkContext& ctx,
                       const VkPipelineResources& res,
                       VkDescriptorSet set,
                       const VkBufferHandle& dummy,
                       VkPipeline pipeline,
                       const BenchOptions& opt,
                       BenchResult& result) {
  const usize n = opt.n;
  const f32 alpha = 0.7f;
  auto x = make_random(n, opt.seed);
  auto y = make_random(n, opt.seed + 1u);
  std::vector<f32> out(n, 0.0f);

  auto bx = vk_create_buffer(ctx,
                             n * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto by = vk_create_buffer(ctx,
                             n * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bo = vk_create_buffer(ctx,
                             n * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  vk_write_buffer(ctx, bx, x.data(), bx.size);
  vk_write_buffer(ctx, by, y.data(), by.size);

  auto infos = make_bindings(dummy, {bx, by, bo});
  vk_update_descriptor_set(ctx, set, infos);

  struct Push {
    u32 n;
    f32 alpha;
  } push{static_cast<u32>(n), alpha};

  const u32 groups = static_cast<u32>((n + 255) / 256);
  auto timing = measure_dispatch(ctx,
                                 pipeline,
                                 res.pipeline_layout,
                                 set,
                                 &push,
                                 sizeof(push),
                                 groups,
                                 1,
                                 1,
                                 static_cast<i32>(opt.repeats));

  vk_read_buffer(ctx, bo, out.data(), bo.size);
  const f64 sum = checksum(out);
  const f64 flops = 2.0 * static_cast<f64>(n);
  result.total_time_ms = timing.total_ms;
  result.calc_time_ms = timing.avg_ms;
  result.mem_time_ms = 0.0;
  result.flops = static_cast<u64>(flops);
  result.gflops = flops / (timing.avg_ms * 1.0e6);
  result.bytes_moved = static_cast<u64>(3 * n * sizeof(f32));
  result.gbytes = static_cast<f64>(result.bytes_moved) / (timing.avg_ms * 1.0e6);
  result.checksum = sum;
  result.status = 0;

  vk_destroy_buffer(ctx, bx);
  vk_destroy_buffer(ctx, by);
  vk_destroy_buffer(ctx, bo);
}

static void run_reduce(VkContext& ctx,
                       const VkPipelineResources& res,
                       VkDescriptorSet set,
                       const VkBufferHandle& dummy,
                       VkPipeline pipeline,
                       const BenchOptions& opt,
                       BenchResult& result) {
  const usize n = opt.n;
  auto data = make_random(n, opt.seed);

  auto bx = vk_create_buffer(ctx,
                             n * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vk_write_buffer(ctx, bx, data.data(), bx.size);

  const u32 groups = static_cast<u32>((n + 255) / 256);
  auto bout = vk_create_buffer(ctx,
                               groups * sizeof(f32),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto infos = make_bindings(dummy, {bx, bout});
  vk_update_descriptor_set(ctx, set, infos);

  struct Push {
    u32 n;
  } push{static_cast<u32>(n)};

  auto timing = measure_dispatch(ctx,
                                 pipeline,
                                 res.pipeline_layout,
                                 set,
                                 &push,
                                 sizeof(push),
                                 groups,
                                 1,
                                 1,
                                 static_cast<i32>(opt.repeats));

  std::vector<f32> partial(groups);
  vk_read_buffer(ctx, bout, partial.data(), bout.size);
  f64 sum = 0.0;
  for (f32 v : partial) {
    sum += v;
  }

  const f64 flops = static_cast<f64>(n);
  result.total_time_ms = timing.total_ms;
  result.calc_time_ms = timing.avg_ms;
  result.mem_time_ms = 0.0;
  result.flops = static_cast<u64>(flops);
  result.gflops = flops / (timing.avg_ms * 1.0e6);
  result.bytes_moved = static_cast<u64>(n * sizeof(f32));
  result.gbytes = static_cast<f64>(result.bytes_moved) / (timing.avg_ms * 1.0e6);
  result.checksum = sum;
  result.status = 0;

  vk_destroy_buffer(ctx, bx);
  vk_destroy_buffer(ctx, bout);
}

static void run_prefix(VkContext& ctx,
                       const VkPipelineResources& res,
                       VkDescriptorSet set,
                       const VkBufferHandle& dummy,
                       VkPipeline pipeline_block,
                       VkPipeline pipeline_add,
                       const BenchOptions& opt,
                       BenchResult& result) {
  const usize n = opt.n;
  auto data = make_random(n, opt.seed);
  std::vector<f32> out(n, 0.0f);

  const u32 groups = static_cast<u32>((n + 255) / 256);

  auto bin = vk_create_buffer(ctx,
                              n * sizeof(f32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bout = vk_create_buffer(ctx,
                               n * sizeof(f32),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bsum = vk_create_buffer(ctx,
                               groups * sizeof(f32),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto boff = vk_create_buffer(ctx,
                               groups * sizeof(f32),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  vk_write_buffer(ctx, bin, data.data(), bin.size);
  auto infos_block = make_bindings(dummy, {bin, bout, bsum});
  auto infos_add = make_bindings(dummy, {bout, boff});

  struct Push {
    u32 n;
  } push{static_cast<u32>(n)};

  Timing timing{};
  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  for (i32 i = 0; i < opt.repeats; ++i) {
    VK_CHECK(vkResetCommandBuffer(ctx.command_buffer, 0));
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(ctx.command_buffer, &begin_info));
    vk_update_descriptor_set(ctx, set, infos_block);
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_block);
    vkCmdBindDescriptorSets(ctx.command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            res.pipeline_layout,
                            0,
                            1,
                            &set,
                            0,
                            nullptr);
    vkCmdPushConstants(ctx.command_buffer, res.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(ctx.command_buffer, groups, 1, 1);
    VK_CHECK(vkEndCommandBuffer(ctx.command_buffer));
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &ctx.command_buffer;
    VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(ctx.queue));

    std::vector<f32> block_sums(groups);
    vk_read_buffer(ctx, bsum, block_sums.data(), bsum.size);
    std::vector<f32> offsets(groups, 0.0f);
    for (usize k = 1; k < groups; ++k) {
      offsets[k] = offsets[k - 1] + block_sums[k - 1];
    }
    vk_write_buffer(ctx, boff, offsets.data(), boff.size);

    VK_CHECK(vkResetCommandBuffer(ctx.command_buffer, 0));
    VK_CHECK(vkBeginCommandBuffer(ctx.command_buffer, &begin_info));
    vk_update_descriptor_set(ctx, set, infos_add);
    vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_add);
    vkCmdBindDescriptorSets(ctx.command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            res.pipeline_layout,
                            0,
                            1,
                            &set,
                            0,
                            nullptr);
    vkCmdPushConstants(ctx.command_buffer, res.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(ctx.command_buffer, groups, 1, 1);
    VK_CHECK(vkEndCommandBuffer(ctx.command_buffer));
    VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(ctx.queue));
  }
  const auto end = clock::now();
  const f64 total_ms = std::chrono::duration_cast<std::chrono::duration<f64, std::milli>>(end - start).count();
  timing.total_ms = total_ms;
  timing.avg_ms = opt.repeats > 0 ? total_ms / static_cast<f64>(opt.repeats) : 0.0;

  vk_read_buffer(ctx, bout, out.data(), bout.size);
  const f64 sum = checksum(out);
  const f64 flops = static_cast<f64>(n);
  result.total_time_ms = timing.total_ms;
  result.calc_time_ms = timing.avg_ms;
  result.mem_time_ms = 0.0;
  result.flops = static_cast<u64>(flops);
  result.gflops = flops / (timing.avg_ms * 1.0e6);
  result.bytes_moved = static_cast<u64>(2 * n * sizeof(f32));
  result.gbytes = static_cast<f64>(result.bytes_moved) / (timing.avg_ms * 1.0e6);
  result.checksum = sum;
  result.status = 0;

  vk_destroy_buffer(ctx, bin);
  vk_destroy_buffer(ctx, bout);
  vk_destroy_buffer(ctx, bsum);
  vk_destroy_buffer(ctx, boff);
}

static void run_hist(VkContext& ctx,
                     const VkPipelineResources& res,
                     VkDescriptorSet set,
                     const VkBufferHandle& dummy,
                     VkPipeline pipeline,
                     const BenchOptions& opt,
                     BenchResult& result) {
  const usize n = opt.n;
  const usize bins = opt.bins == 0 ? 256 : opt.bins;
  auto data = make_random_u32(n, opt.seed, static_cast<u32>(bins));
  std::vector<u32> hist(bins, 0u);

  auto bdata = vk_create_buffer(ctx,
                                n * sizeof(u32),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bbins = vk_create_buffer(ctx,
                                bins * sizeof(u32),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vk_write_buffer(ctx, bdata, data.data(), bdata.size);
  auto infos = make_bindings(dummy, {bdata, bbins});
  vk_update_descriptor_set(ctx, set, infos);

  struct Push {
    u32 n;
    u32 bins;
  } push{static_cast<u32>(n), static_cast<u32>(bins)};

  const u32 groups = static_cast<u32>((n + 255) / 256);
  Timing timing{};
  for (i32 i = 0; i < opt.repeats; ++i) {
    std::fill(hist.begin(), hist.end(), 0u);
    vk_write_buffer(ctx, bbins, hist.data(), bbins.size);
    auto t = measure_dispatch(ctx, pipeline, res.pipeline_layout, set, &push, sizeof(push), groups, 1, 1, 1);
    timing.total_ms += t.total_ms;
  }
  const f64 total_ms = timing.total_ms;
  timing.avg_ms = opt.repeats > 0 ? total_ms / static_cast<f64>(opt.repeats) : 0.0;

  vk_read_buffer(ctx, bbins, hist.data(), bbins.size);
  const f64 sum = checksum_u32(hist);
  const f64 flops = static_cast<f64>(n);
  result.total_time_ms = total_ms;
  result.calc_time_ms = timing.avg_ms;
  result.mem_time_ms = 0.0;
  result.flops = static_cast<u64>(flops);
  result.gflops = flops / (timing.avg_ms * 1.0e6);
  result.bytes_moved = static_cast<u64>(n * sizeof(u32) + bins * sizeof(u32));
  result.gbytes = static_cast<f64>(result.bytes_moved) / (timing.avg_ms * 1.0e6);
  result.checksum = sum;
  result.status = 0;

  vk_destroy_buffer(ctx, bdata);
  vk_destroy_buffer(ctx, bbins);
}

static void run_conv2d(VkContext& ctx,
                       const VkPipelineResources& res,
                       VkDescriptorSet set,
                       const VkBufferHandle& dummy,
                       VkPipeline pipeline,
                       const BenchOptions& opt,
                       BenchResult& result) {
  const usize n = opt.n;
  const usize ksize = opt.ksize;
  const usize size = n * n;
  auto input = make_random(size, opt.seed);
  auto kernel = make_kernel(ksize, opt.seed + 1u);
  std::vector<f32> output(size, 0.0f);

  auto bin = vk_create_buffer(ctx,
                              size * sizeof(f32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bkernel = vk_create_buffer(ctx,
                                  kernel.size() * sizeof(f32),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bout = vk_create_buffer(ctx,
                               size * sizeof(f32),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  vk_write_buffer(ctx, bin, input.data(), bin.size);
  vk_write_buffer(ctx, bkernel, kernel.data(), bkernel.size);
  auto infos = make_bindings(dummy, {bin, bkernel, bout});
  vk_update_descriptor_set(ctx, set, infos);

  struct Push {
    u32 width;
    u32 height;
    u32 ksize;
  } push{static_cast<u32>(n), static_cast<u32>(n), static_cast<u32>(ksize)};

  const u32 group_x = static_cast<u32>((n + 15) / 16);
  const u32 group_y = static_cast<u32>((n + 15) / 16);
  auto timing = measure_dispatch(ctx,
                                 pipeline,
                                 res.pipeline_layout,
                                 set,
                                 &push,
                                 sizeof(push),
                                 group_x,
                                 group_y,
                                 1,
                                 static_cast<i32>(opt.repeats));

  vk_read_buffer(ctx, bout, output.data(), bout.size);
  const f64 sum = checksum(output);
  const f64 flops = 2.0 * static_cast<f64>(ksize) * static_cast<f64>(ksize) * static_cast<f64>(size);
  result.total_time_ms = timing.total_ms;
  result.calc_time_ms = timing.avg_ms;
  result.mem_time_ms = 0.0;
  result.flops = static_cast<u64>(flops);
  result.gflops = flops / (timing.avg_ms * 1.0e6);
  result.bytes_moved = static_cast<u64>((2 * size + ksize * ksize) * sizeof(f32));
  result.gbytes = static_cast<f64>(result.bytes_moved) / (timing.avg_ms * 1.0e6);
  result.checksum = sum;
  result.status = 0;

  vk_destroy_buffer(ctx, bin);
  vk_destroy_buffer(ctx, bkernel);
  vk_destroy_buffer(ctx, bout);
}

static void run_spmv(VkContext& ctx,
                     const VkPipelineResources& res,
                     VkDescriptorSet set,
                     const VkBufferHandle& dummy,
                     VkPipeline pipeline,
                     const BenchOptions& opt,
                     BenchResult& result) {
  const u32 n = static_cast<u32>(opt.n);
  const u32 nnz = static_cast<u32>(opt.nnz_per_row == 0 ? 1 : opt.nnz_per_row);
  auto mat = make_csr_matrix(n, n, nnz, opt.seed);
  auto x = make_random(n, opt.seed + 1u);
  std::vector<f32> y(n, 0.0f);

  auto brow = vk_create_buffer(ctx,
                               mat.row_ptr.size() * sizeof(u32),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bcol = vk_create_buffer(ctx,
                               mat.col_idx.size() * sizeof(u32),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bval = vk_create_buffer(ctx,
                               mat.values.size() * sizeof(f32),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bx = vk_create_buffer(ctx,
                             x.size() * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto by = vk_create_buffer(ctx,
                             y.size() * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  vk_write_buffer(ctx, brow, mat.row_ptr.data(), brow.size);
  vk_write_buffer(ctx, bcol, mat.col_idx.data(), bcol.size);
  vk_write_buffer(ctx, bval, mat.values.data(), bval.size);
  vk_write_buffer(ctx, bx, x.data(), bx.size);
  auto infos = make_bindings(dummy, {brow, bcol, bval, bx, by});
  vk_update_descriptor_set(ctx, set, infos);

  struct Push {
    u32 rows;
  } push{n};

  const u32 groups = static_cast<u32>((n + 255) / 256);
  auto timing = measure_dispatch(ctx,
                                 pipeline,
                                 res.pipeline_layout,
                                 set,
                                 &push,
                                 sizeof(push),
                                 groups,
                                 1,
                                 1,
                                 static_cast<i32>(opt.repeats));

  vk_read_buffer(ctx, by, y.data(), by.size);
  const f64 sum = checksum(y);
  const f64 flops = 2.0 * static_cast<f64>(mat.values.size());
  result.total_time_ms = timing.total_ms;
  result.calc_time_ms = timing.avg_ms;
  result.mem_time_ms = 0.0;
  result.flops = static_cast<u64>(flops);
  result.gflops = flops / (timing.avg_ms * 1.0e6);
  result.bytes_moved =
      static_cast<u64>(mat.values.size() * sizeof(f32) + mat.col_idx.size() * sizeof(u32) +
                       mat.row_ptr.size() * sizeof(u32) + x.size() * sizeof(f32) + y.size() * sizeof(f32));
  result.gbytes = static_cast<f64>(result.bytes_moved) / (timing.avg_ms * 1.0e6);
  result.checksum = sum;
  result.status = 0;

  vk_destroy_buffer(ctx, brow);
  vk_destroy_buffer(ctx, bcol);
  vk_destroy_buffer(ctx, bval);
  vk_destroy_buffer(ctx, bx);
  vk_destroy_buffer(ctx, by);
}

static void run_matmul(VkContext& ctx,
                       const VkPipelineResources& res,
                       VkDescriptorSet set,
                       const VkBufferHandle& dummy,
                       VkPipeline pipeline,
                       const BenchOptions& opt,
                       BenchResult& result) {
  const usize n = opt.n;
  const usize size = n * n;
  auto a = make_random(size, opt.seed);
  auto b = make_random(size, opt.seed + 1u);
  std::vector<f32> c(size, 0.0f);

  auto ba = vk_create_buffer(ctx,
                             size * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bb = vk_create_buffer(ctx,
                             size * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bc = vk_create_buffer(ctx,
                             size * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  vk_write_buffer(ctx, ba, a.data(), ba.size);
  vk_write_buffer(ctx, bb, b.data(), bb.size);
  auto infos = make_bindings(dummy, {ba, bb, bc});
  vk_update_descriptor_set(ctx, set, infos);

  struct Push {
    u32 n;
  } push{static_cast<u32>(n)};

  const u32 group_x = static_cast<u32>((n + 15) / 16);
  const u32 group_y = static_cast<u32>((n + 15) / 16);
  auto timing = measure_dispatch(ctx,
                                 pipeline,
                                 res.pipeline_layout,
                                 set,
                                 &push,
                                 sizeof(push),
                                 group_x,
                                 group_y,
                                 1,
                                 static_cast<i32>(opt.repeats));

  vk_read_buffer(ctx, bc, c.data(), bc.size);
  const f64 sum = checksum(c);
  const f64 flops = 2.0 * static_cast<f64>(n) * static_cast<f64>(n) * static_cast<f64>(n);
  result.total_time_ms = timing.total_ms;
  result.calc_time_ms = timing.avg_ms;
  result.mem_time_ms = 0.0;
  result.flops = static_cast<u64>(flops);
  result.gflops = flops / (timing.avg_ms * 1.0e6);
  result.bytes_moved = static_cast<u64>(3 * size * sizeof(f32));
  result.gbytes = static_cast<f64>(result.bytes_moved) / (timing.avg_ms * 1.0e6);
  result.checksum = sum;
  result.status = 0;

  vk_destroy_buffer(ctx, ba);
  vk_destroy_buffer(ctx, bb);
  vk_destroy_buffer(ctx, bc);
}

// ---------------------------------------------------------------- Black-Scholes
static void run_blackscholes(VkContext& ctx,
                             const VkPipelineResources& res,
                             VkDescriptorSet set,
                             const VkBufferHandle& dummy,
                             VkPipeline pipeline,
                             const BenchOptions& opt,
                             BenchResult& result) {
  const usize n = opt.n;
  auto s = make_random(n, opt.seed, 80.0f, 120.0f);
  std::vector<f32> out(n, 0.0f);

  auto bs = vk_create_buffer(ctx,
                             n * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bo = vk_create_buffer(ctx,
                             n * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vk_write_buffer(ctx, bs, s.data(), bs.size);
  auto infos = make_bindings(dummy, {bs, bo});
  vk_update_descriptor_set(ctx, set, infos);

  struct Push {
    u32 n;
    f32 K;
    f32 T;
    f32 r;
    f32 sigma;
  } push{static_cast<u32>(n), 100.0f, 1.0f, 0.05f, 0.2f};

  const u32 groups = static_cast<u32>((n + 255) / 256);
  auto timing = measure_dispatch(ctx,
                                 pipeline,
                                 res.pipeline_layout,
                                 set,
                                 &push,
                                 sizeof(push),
                                 groups,
                                 1,
                                 1,
                                 static_cast<i32>(opt.repeats));

  vk_read_buffer(ctx, bo, out.data(), bo.size);
  const f64 sum = checksum(out);
  const f64 flops = 50.0 * static_cast<f64>(n);
  result.total_time_ms = timing.total_ms;
  result.calc_time_ms = timing.avg_ms;
  result.flops = static_cast<u64>(flops);
  result.gflops = flops / (timing.avg_ms * 1.0e6);
  result.bytes_moved = static_cast<u64>(2 * n * sizeof(f32));
  result.gbytes = static_cast<f64>(result.bytes_moved) / (timing.avg_ms * 1.0e6);
  result.checksum = sum;
  result.status = 0;

  vk_destroy_buffer(ctx, bs);
  vk_destroy_buffer(ctx, bo);
}

// ---------------------------------------------------------------- Bitonic Sort
static void run_bsort(VkContext& ctx,
                      const VkPipelineResources& res,
                      VkDescriptorSet set,
                      const VkBufferHandle& dummy,
                      VkPipeline pipeline,
                      const BenchOptions& opt,
                      BenchResult& result) {
  // Round n up to next power-of-2
  usize n2 = 1;
  while (n2 < opt.n) {
    n2 <<= 1;
  }

  auto orig = make_random(static_cast<usize>(n2), opt.seed);
  std::vector<f32> data(n2);

  auto buf = vk_create_buffer(ctx,
                              n2 * sizeof(f32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  usize log2n = 0;
  for (usize t = n2; t > 1; t >>= 1) {
    ++log2n;
  }
  const u64 comparisons = (static_cast<u64>(n2) >> 1) * log2n * (log2n + 1) / 2;

  using clock = std::chrono::steady_clock;
  f64 total_ms = 0.0;

  for (i32 rep = 0; rep < opt.repeats; ++rep) {
    vk_write_buffer(ctx, buf, orig.data(), buf.size);
    auto infos = make_bindings(dummy, {buf});
    vk_update_descriptor_set(ctx, set, infos);

    const auto t0 = clock::now();
    for (u32 stage = 1; stage <= static_cast<u32>(log2n); ++stage) {
      for (u32 step = stage; step >= 1; --step) {
        struct Push {
          u32 n;
          u32 step;
          u32 stage;
        } push{static_cast<u32>(n2), step, stage};

        VK_CHECK(vkResetCommandBuffer(ctx.command_buffer, 0));
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(ctx.command_buffer, &begin_info));
        vkCmdBindPipeline(ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(ctx.command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                res.pipeline_layout,
                                0,
                                1,
                                &set,
                                0,
                                nullptr);
        vkCmdPushConstants(ctx.command_buffer,
                           res.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(push),
                           &push);
        const u32 grps = static_cast<u32>(((n2 / 2) + 255) / 256);
        vkCmdDispatch(ctx.command_buffer, grps, 1, 1);
        VK_CHECK(vkEndCommandBuffer(ctx.command_buffer));
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &ctx.command_buffer;
        VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(ctx.queue));
      }
    }
    const auto t1 = clock::now();
    total_ms += std::chrono::duration_cast<std::chrono::duration<f64, std::milli>>(t1 - t0).count();
  }

  vk_read_buffer(ctx, buf, data.data(), buf.size);
  const f64 avg_ms = opt.repeats > 0 ? total_ms / static_cast<f64>(opt.repeats) : 0.0;
  result.total_time_ms = total_ms;
  result.calc_time_ms = avg_ms;
  result.flops = comparisons;
  result.gflops = static_cast<f64>(comparisons) / (avg_ms * 1.0e6);
  result.bytes_moved = static_cast<u64>(2 * n2 * sizeof(f32));
  result.gbytes = static_cast<f64>(result.bytes_moved) / (avg_ms * 1.0e6);
  result.checksum = checksum(data);
  result.status = 0;

  vk_destroy_buffer(ctx, buf);
}

// ---------------------------------------------------------------- N-Body
static void run_nbody(VkContext& ctx,
                      const VkPipelineResources& res,
                      VkDescriptorSet set,
                      const VkBufferHandle& dummy,
                      VkPipeline pipeline,
                      const BenchOptions& opt,
                      BenchResult& result) {
  const u32 n = static_cast<u32>(opt.n);
  auto px = make_random(n, opt.seed, -100.0f, 100.0f);
  auto py = make_random(n, opt.seed + 1u, -100.0f, 100.0f);
  auto mass = make_random(n, opt.seed + 2u, 0.1f, 2.0f);
  std::vector<f32> fx(n, 0.0f), fy(n, 0.0f);

  auto bpx = vk_create_buffer(ctx,
                              n * sizeof(f32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bpy = vk_create_buffer(ctx,
                              n * sizeof(f32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bm = vk_create_buffer(ctx,
                             n * sizeof(f32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bfx = vk_create_buffer(ctx,
                              n * sizeof(f32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto bfy = vk_create_buffer(ctx,
                              n * sizeof(f32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  vk_write_buffer(ctx, bpx, px.data(), bpx.size);
  vk_write_buffer(ctx, bpy, py.data(), bpy.size);
  vk_write_buffer(ctx, bm, mass.data(), bm.size);
  auto infos = make_bindings(dummy, {bpx, bpy, bm, bfx, bfy});
  vk_update_descriptor_set(ctx, set, infos);

  struct Push {
    u32 source_n;
    u32 target_n;
    f32 softening2;
  } push{n, n, 1e-4f};

  const u32 groups = (n + 255) / 256;
  auto timing = measure_dispatch(ctx,
                                 pipeline,
                                 res.pipeline_layout,
                                 set,
                                 &push,
                                 sizeof(push),
                                 groups,
                                 1,
                                 1,
                                 static_cast<i32>(opt.repeats));

  vk_read_buffer(ctx, bfx, fx.data(), bfx.size);
  vk_read_buffer(ctx, bfy, fy.data(), bfy.size);
  const f64 flops = 20.0 * static_cast<f64>(n) * static_cast<f64>(n - 1);
  result.total_time_ms = timing.total_ms;
  result.calc_time_ms = timing.avg_ms;
  result.flops = static_cast<u64>(flops);
  result.gflops = flops / (timing.avg_ms * 1.0e6);
  result.bytes_moved = static_cast<u64>(5 * n * sizeof(f32));
  result.gbytes = static_cast<f64>(result.bytes_moved) / (timing.avg_ms * 1.0e6);
  result.checksum = checksum(fx) + checksum(fy);
  result.status = 0;

  vk_destroy_buffer(ctx, bpx);
  vk_destroy_buffer(ctx, bpy);
  vk_destroy_buffer(ctx, bm);
  vk_destroy_buffer(ctx, bfx);
  vk_destroy_buffer(ctx, bfy);
}

static const BenchEntry kEntries[] = {
    {"vecadd", BenchAlgo::kBenchAlgoVecadd},
    {"reduce", BenchAlgo::kBenchAlgoReduce},
    {"prefix", BenchAlgo::kBenchAlgoPrefix},
    {"hist", BenchAlgo::kBenchAlgoHist},
    {"conv2d", BenchAlgo::kBenchAlgoConV2D},
    {"spmv", BenchAlgo::kBenchAlgoSpmv},
    {"matmul", BenchAlgo::kBenchAlgoMatmul},
    {"blackscholes", BenchAlgo::kBenchAlgoBlackscholes},
    {"bsort", BenchAlgo::kBenchAlgoBsort},
    {"nbody", BenchAlgo::kBenchAlgoNbody},
};

static const char* get_name() {
  return "vulkan";
}

static u32 get_entries(const BenchEntry** out_entries) {
  if (out_entries) {
    *out_entries = kEntries;
  }
  return static_cast<u32>(sizeof(kEntries) / sizeof(kEntries[0]));
}

static i32 run_bench(const BenchOptions* options, BenchResult* out_result) {
  if (options == nullptr || out_result == nullptr) {
    return -1;
  }
  BenchOptions opt = bench::resolve_options(*options);

  const bool enable_validation = std::getenv("VKBENCH_VALIDATION") != nullptr;
  VkContext ctx;
  if (!vk_init(ctx, enable_validation)) {
    out_result->status = -3;
    return -3;
  }

  VkPipelineResources resources = vk_create_pipeline_resources(ctx);

  VkBufferHandle dummy = vk_create_buffer(ctx,
                                          4,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  u32 zero = 0u;
  vk_write_buffer(ctx, dummy, &zero, sizeof(zero));

  VkDescriptorSet set = vk_allocate_descriptor_set(ctx, resources);

  BenchResult result{};
  switch (opt.algo) {
    case BenchAlgo::kBenchAlgoVecadd: {
      VkPipelineBundle pipe = vk_create_compute_pipeline(ctx, resources, shader_path("vecadd"));
      run_vecadd(ctx, resources, set, dummy, pipe.pipeline, opt, result);
      vk_destroy_pipeline(ctx, pipe);
    } break;
    case BenchAlgo::kBenchAlgoReduce: {
      VkPipelineBundle pipe = vk_create_compute_pipeline(ctx, resources, shader_path("reduce"));
      run_reduce(ctx, resources, set, dummy, pipe.pipeline, opt, result);
      vk_destroy_pipeline(ctx, pipe);
    } break;
    case BenchAlgo::kBenchAlgoPrefix: {
      VkPipelineBundle pipe_block = vk_create_compute_pipeline(ctx, resources, shader_path("scan_block"));
      VkPipelineBundle pipe_add = vk_create_compute_pipeline(ctx, resources, shader_path("scan_add"));
      run_prefix(ctx, resources, set, dummy, pipe_block.pipeline, pipe_add.pipeline, opt, result);
      vk_destroy_pipeline(ctx, pipe_block);
      vk_destroy_pipeline(ctx, pipe_add);
    } break;
    case BenchAlgo::kBenchAlgoHist: {
      VkPipelineBundle pipe = vk_create_compute_pipeline(ctx, resources, shader_path("hist"));
      run_hist(ctx, resources, set, dummy, pipe.pipeline, opt, result);
      vk_destroy_pipeline(ctx, pipe);
    } break;
    case BenchAlgo::kBenchAlgoConV2D: {
      VkPipelineBundle pipe = vk_create_compute_pipeline(ctx, resources, shader_path("conv2d"));
      run_conv2d(ctx, resources, set, dummy, pipe.pipeline, opt, result);
      vk_destroy_pipeline(ctx, pipe);
    } break;
    case BenchAlgo::kBenchAlgoSpmv: {
      VkPipelineBundle pipe = vk_create_compute_pipeline(ctx, resources, shader_path("spmv"));
      run_spmv(ctx, resources, set, dummy, pipe.pipeline, opt, result);
      vk_destroy_pipeline(ctx, pipe);
    } break;
    case BenchAlgo::kBenchAlgoMatmul: {
      VkPipelineBundle pipe = vk_create_compute_pipeline(ctx, resources, shader_path("matmul"));
      run_matmul(ctx, resources, set, dummy, pipe.pipeline, opt, result);
      vk_destroy_pipeline(ctx, pipe);
    } break;
    case BenchAlgo::kBenchAlgoBlackscholes: {
      VkPipelineBundle pipe = vk_create_compute_pipeline(ctx, resources, shader_path("blackscholes"));
      run_blackscholes(ctx, resources, set, dummy, pipe.pipeline, opt, result);
      vk_destroy_pipeline(ctx, pipe);
    } break;
    case BenchAlgo::kBenchAlgoBsort: {
      VkPipelineBundle pipe = vk_create_compute_pipeline(ctx, resources, shader_path("bsort"));
      run_bsort(ctx, resources, set, dummy, pipe.pipeline, opt, result);
      vk_destroy_pipeline(ctx, pipe);
    } break;
    case BenchAlgo::kBenchAlgoNbody: {
      VkPipelineBundle pipe = vk_create_compute_pipeline(ctx, resources, shader_path("nbody"));
      run_nbody(ctx, resources, set, dummy, pipe.pipeline, opt, result);
      vk_destroy_pipeline(ctx, pipe);
    } break;
    default:
      vk_destroy_buffer(ctx, dummy);
      vk_destroy_pipeline_resources(ctx, resources);
      vk_destroy(ctx);
      return -2;
  }

  vk_destroy_buffer(ctx, dummy);
  vk_destroy_pipeline_resources(ctx, resources);
  vk_destroy(ctx);

  *out_result = result;
  return 0;
}

extern "C" const BenchApi* bench_get_api() {
  static BenchApi api{get_name, get_entries, run_bench};
  return &api;
}
