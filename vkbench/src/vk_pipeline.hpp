#pragma once

#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "vk_context.hpp"

struct VkPipelineBundle
{
  VkPipeline pipeline = VK_NULL_HANDLE;
};

struct VkPipelineResources
{
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
};

VkPipelineResources vk_create_pipeline_resources(const VkContext &ctx);
void vk_destroy_pipeline_resources(const VkContext &ctx, VkPipelineResources &res);

VkPipelineBundle vk_create_compute_pipeline(const VkContext &ctx, const VkPipelineResources &res,
                                            const std::string &spv_path);
void vk_destroy_pipeline(const VkContext &ctx, VkPipelineBundle &pipe);

VkDescriptorSet vk_allocate_descriptor_set(const VkContext &ctx, const VkPipelineResources &res);
void vk_update_descriptor_set(const VkContext &ctx, VkDescriptorSet set,
                              const std::vector<VkDescriptorBufferInfo> &buffers);