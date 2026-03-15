#pragma once

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

struct VkContext {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;

  VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceMemoryProperties memory_properties{};
};

bool vk_init(VkContext& ctx, bool enable_validation);
void vk_destroy(VkContext& ctx);
uint32_t vk_find_memory_type(const VkContext& ctx, uint32_t type_bits, VkMemoryPropertyFlags properties);