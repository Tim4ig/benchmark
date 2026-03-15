#include "vk_pipeline.hpp"

#include "vk_common.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <vector>

namespace {
std::vector<std::uint32_t> read_spv(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return {};
  }
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<std::uint32_t> data(static_cast<std::size_t>(size / sizeof(std::uint32_t)));
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}
} // namespace

VkPipelineResources vk_create_pipeline_resources(const VkContext& ctx) {
  VkPipelineResources res{};

  std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
  for (std::uint32_t i = 0; i < bindings.size(); ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();
  VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr, &res.set_layout));

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = 64;

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &res.set_layout;
  pipeline_layout_info.pushConstantRangeCount = 1;
  pipeline_layout_info.pPushConstantRanges = &push_range;
  VK_CHECK(vkCreatePipelineLayout(ctx.device, &pipeline_layout_info, nullptr, &res.pipeline_layout));

  const uint32_t max_sets = 32;
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = max_sets * static_cast<uint32_t>(bindings.size());

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  pool_info.maxSets = max_sets;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  VK_CHECK(vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &res.descriptor_pool));

  return res;
}

void vk_destroy_pipeline_resources(const VkContext& ctx, VkPipelineResources& res) {
  if (res.descriptor_pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(ctx.device, res.descriptor_pool, nullptr);
  }
  if (res.pipeline_layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(ctx.device, res.pipeline_layout, nullptr);
  }
  if (res.set_layout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(ctx.device, res.set_layout, nullptr);
  }
  res = VkPipelineResources{};
}

VkPipelineBundle
vk_create_compute_pipeline(const VkContext& ctx, const VkPipelineResources& res, const std::string& spv_path) {
  VkPipelineBundle bundle{};
  auto code = read_spv(spv_path);
  if (code.empty()) {
    return bundle;
  }

  VkShaderModuleCreateInfo shader_info{};
  shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_info.codeSize = code.size() * sizeof(std::uint32_t);
  shader_info.pCode = code.data();

  VkShaderModule shader_module = VK_NULL_HANDLE;
  VK_CHECK(vkCreateShaderModule(ctx.device, &shader_info, nullptr, &shader_module));

  VkPipelineShaderStageCreateInfo stage_info{};
  stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = shader_module;
  stage_info.pName = "main";

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = stage_info;
  pipeline_info.layout = res.pipeline_layout;

  VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &bundle.pipeline));

  vkDestroyShaderModule(ctx.device, shader_module, nullptr);
  return bundle;
}

void vk_destroy_pipeline(const VkContext& ctx, VkPipelineBundle& pipe) {
  if (pipe.pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(ctx.device, pipe.pipeline, nullptr);
  }
  pipe = VkPipelineBundle{};
}

VkDescriptorSet vk_allocate_descriptor_set(const VkContext& ctx, const VkPipelineResources& res) {
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = res.descriptor_pool;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &res.set_layout;

  VkDescriptorSet set = VK_NULL_HANDLE;
  VK_CHECK(vkAllocateDescriptorSets(ctx.device, &alloc_info, &set));
  return set;
}

void vk_update_descriptor_set(const VkContext& ctx,
                              VkDescriptorSet set,
                              const std::vector<VkDescriptorBufferInfo>& buffers) {
  std::vector<VkWriteDescriptorSet> writes;
  writes.reserve(buffers.size());
  for (uint32_t i = 0; i < buffers.size(); ++i) {
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = i;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &buffers[i];
    writes.push_back(write);
  }
  vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}