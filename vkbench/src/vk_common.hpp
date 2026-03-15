#pragma once

#include <cstdio>
#include <cstdlib>
#include <vulkan/vulkan.h>

inline void vk_check(VkResult result, const char* expr, const char* file, int line) {
  if (result != VK_SUCCESS) {
    std::fprintf(stderr, "Vulkan error %d at %s:%d for %s\n", result, file, line, expr);
    std::abort();
  }
}

#define VK_CHECK(expr) vk_check((expr), #expr, __FILE__, __LINE__)