#pragma once

#include "../../common_abi/bench_abi.h"
#include "../../vkbench/src/vk_buffer.hpp"
#include "../../vkbench/src/vk_context.hpp"
#include "../../vkbench/src/vk_pipeline.hpp"

#include <string>

namespace bench::hybrid {
class HybridRunner final {
 public:
  HybridRunner();
  ~HybridRunner();

  HybridRunner(const HybridRunner&) = delete;
  HybridRunner& operator=(const HybridRunner&) = delete;
  HybridRunner(HybridRunner&&) = delete;
  HybridRunner& operator=(HybridRunner&&) = delete;

  BenchResult run(const BenchOptions& options);

 private:
  BenchResult run_matmul(const BenchOptions& options);
  BenchResult run_nbody(const BenchOptions& options);

  bool ready_ = false;
  std::string shader_dir_;
  VkContext ctx_{};
  VkPipelineResources resources_{};
  VkPipelineBundle matmul_pipeline_{};
  VkPipelineBundle nbody_pipeline_{};
  VkBufferHandle dummy_{};
};
} // namespace bench::hybrid
