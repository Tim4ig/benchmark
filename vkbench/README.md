# vkbench

Standalone Vulkan compute benchmark module.

## Requirements

- Vulkan headers + loader (Vulkan SDK or system packages)
- `glslangValidator`
- CMake 3.16+

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/vkbench --algo vecadd --n 1048576
./build/vkbench --algo matmul --n 512
```

Set `VKBENCH_VALIDATION=1` to enable validation layers when available.
