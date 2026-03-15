#include "vk_buffer.hpp"

#include "vk_common.hpp"

#include <cstring>

VkBufferHandle
vk_create_buffer(const VkContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
  VkBufferHandle handle;
  handle.size = size;
  handle.properties = properties;

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VK_CHECK(vkCreateBuffer(ctx.device, &buffer_info, nullptr, &handle.buffer));

  VkMemoryRequirements mem_req{};
  vkGetBufferMemoryRequirements(ctx.device, handle.buffer, &mem_req);

  uint32_t memory_type = vk_find_memory_type(ctx, mem_req.memoryTypeBits, properties);
  if (memory_type == UINT32_MAX) {
    vkDestroyBuffer(ctx.device, handle.buffer, nullptr);
    handle.buffer = VK_NULL_HANDLE;
    return handle;
  }

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_req.size;
  alloc_info.memoryTypeIndex = memory_type;

  VK_CHECK(vkAllocateMemory(ctx.device, &alloc_info, nullptr, &handle.memory));
  VK_CHECK(vkBindBufferMemory(ctx.device, handle.buffer, handle.memory, 0));

  return handle;
}

void vk_destroy_buffer(const VkContext& ctx, VkBufferHandle& buf) {
  if (buf.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(ctx.device, buf.buffer, nullptr);
  }
  if (buf.memory != VK_NULL_HANDLE) {
    vkFreeMemory(ctx.device, buf.memory, nullptr);
  }
  buf = VkBufferHandle{};
}

static void flush_if_needed(const VkContext& ctx, VkBufferHandle& buf, VkDeviceSize size, VkDeviceSize offset) {
  if (buf.properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
    return;
  }
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = buf.memory;
  range.offset = offset;
  range.size = size;
  VK_CHECK(vkFlushMappedMemoryRanges(ctx.device, 1, &range));
}

static void invalidate_if_needed(const VkContext& ctx, VkBufferHandle& buf, VkDeviceSize size, VkDeviceSize offset) {
  if (buf.properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
    return;
  }
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = buf.memory;
  range.offset = offset;
  range.size = size;
  VK_CHECK(vkInvalidateMappedMemoryRanges(ctx.device, 1, &range));
}

void vk_write_buffer(
    const VkContext& ctx, VkBufferHandle& buf, const void* data, VkDeviceSize size, VkDeviceSize offset) {
  void* mapped = nullptr;
  VK_CHECK(vkMapMemory(ctx.device, buf.memory, offset, size, 0, &mapped));
  std::memcpy(mapped, data, static_cast<std::size_t>(size));
  flush_if_needed(ctx, buf, size, offset);
  vkUnmapMemory(ctx.device, buf.memory);
}

void vk_read_buffer(const VkContext& ctx, VkBufferHandle& buf, void* data, VkDeviceSize size, VkDeviceSize offset) {
  void* mapped = nullptr;
  VK_CHECK(vkMapMemory(ctx.device, buf.memory, offset, size, 0, &mapped));
  invalidate_if_needed(ctx, buf, size, offset);
  std::memcpy(data, mapped, static_cast<std::size_t>(size));
  vkUnmapMemory(ctx.device, buf.memory);
}