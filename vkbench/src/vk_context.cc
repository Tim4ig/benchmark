#include "vk_context.hpp"

#include "vk_common.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace {
bool layer_available(const char* name) {
  uint32_t count = 0;
  VK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
  std::vector<VkLayerProperties> layers(count);
  VK_CHECK(vkEnumerateInstanceLayerProperties(&count, layers.data()));
  for (const auto& layer : layers) {
    if (std::strcmp(layer.layerName, name) == 0) {
      return true;
    }
  }
  return false;
}

bool device_is_suitable(VkPhysicalDevice device, uint32_t& queue_family) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> props(count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());
  for (uint32_t i = 0; i < count; ++i) {
    if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      queue_family = i;
      return true;
    }
  }
  return false;
}
} // namespace

bool vk_init(VkContext& ctx, bool enable_validation) {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "vkbench";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "vkbench";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_1;

  std::vector<const char*> layers;
  if (enable_validation && layer_available("VK_LAYER_KHRONOS_validation")) {
    layers.push_back("VK_LAYER_KHRONOS_validation");
  }

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
  create_info.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

  VK_CHECK(vkCreateInstance(&create_info, nullptr, &ctx.instance));

  uint32_t device_count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &device_count, nullptr));
  if (device_count == 0) {
    return false;
  }
  std::vector<VkPhysicalDevice> devices(device_count);
  VK_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &device_count, devices.data()));

  VkPhysicalDevice chosen = VK_NULL_HANDLE;
  uint32_t chosen_queue_family = 0;
  int best_score = -1;

  for (VkPhysicalDevice device : devices) {
    uint32_t qf = 0;
    if (!device_is_suitable(device, qf)) {
      continue;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device, &props);
    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      score = 2;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
      score = 1;
    }
    if (score > best_score) {
      best_score = score;
      chosen = device;
      chosen_queue_family = qf;
    }
  }

  if (chosen == VK_NULL_HANDLE) {
    return false;
  }

  ctx.physical_device = chosen;
  ctx.queue_family = chosen_queue_family;
  vkGetPhysicalDeviceProperties(ctx.physical_device, &ctx.properties);
  vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &ctx.memory_properties);

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = ctx.queue_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;

  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;

  VK_CHECK(vkCreateDevice(ctx.physical_device, &device_info, nullptr, &ctx.device));
  vkGetDeviceQueue(ctx.device, ctx.queue_family, 0, &ctx.queue);

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = ctx.queue_family;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VK_CHECK(vkCreateCommandPool(ctx.device, &pool_info, nullptr, &ctx.command_pool));

  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = ctx.command_pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(ctx.device, &alloc_info, &ctx.command_buffer));

  return true;
}

void vk_destroy(VkContext& ctx) {
  if (ctx.device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(ctx.device);
  }
  if (ctx.command_pool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(ctx.device, ctx.command_pool, nullptr);
  }
  if (ctx.device != VK_NULL_HANDLE) {
    vkDestroyDevice(ctx.device, nullptr);
  }
  if (ctx.instance != VK_NULL_HANDLE) {
    vkDestroyInstance(ctx.instance, nullptr);
  }
  ctx = VkContext{};
}

uint32_t vk_find_memory_type(const VkContext& ctx, uint32_t type_bits, VkMemoryPropertyFlags properties) {
  for (uint32_t i = 0; i < ctx.memory_properties.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) && (ctx.memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  return UINT32_MAX;
}