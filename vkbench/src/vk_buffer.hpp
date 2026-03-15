#pragma once

#include "vk_context.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

struct VkBufferHandle {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  VkMemoryPropertyFlags properties = 0;
};

VkBufferHandle
vk_create_buffer(const VkContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
void vk_destroy_buffer(const VkContext& ctx, VkBufferHandle& buf);
void vk_write_buffer(
    const VkContext& ctx, VkBufferHandle& buf, const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
void vk_read_buffer(const VkContext& ctx, VkBufferHandle& buf, void* data, VkDeviceSize size, VkDeviceSize offset = 0);