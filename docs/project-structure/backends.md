# Backends and Support Modules

## Backend Matrix

| Backend | Artifact | Purpose | Algorithms | Notes |
| --- | --- | --- | --- | --- |
| `cpu_ref` | shared library | scalar baseline | 10 | wraps `cpu_scalar`, built with low optimization |
| `cpu_auto` | shared library | compiler-optimized scalar baseline | 10 | wraps `cpu_scalar`, built with `-O3 -march=native` |
| `cpu_avx512` | shared library | manual SIMD backend | 10 | explicit AVX-512 kernels |
| `cpu_mt` | shared library | multi-threaded CPU backend | 10 | parallel wrappers around AVX-512 kernels |
| `vulkan` | shared library | GPU compute backend | 10 | Vulkan host code plus GLSL shaders |
| `hybrid` | shared library | direct CPU+GPU split backend | 2 | `matmul`, `nbody` |
| `bench_orchestrator` | executable | runtime plugin loader and CSV writer | n/a | drives all discovered backends |

## 1. `cpu_ref`

Directory: [`cpu_ref/`](../../cpu_ref)

Role:

- correctness-oriented scalar baseline
- speedup reference for plotting
- readable version of the benchmark wrappers

Structure:

- `cpu_ref_api.cc`
- `cpu_ref_runner.cc/.h`
- `algos/algorithm.h`
- `algos/algorithms.h`

The algorithm logic itself lives in [`cpu_scalar/algos/`](../../cpu_scalar/algos).

## 2. `cpu_auto`

Directory: [`cpu_auto/`](../../cpu_auto)

Role:

- same scalar algorithm source as `cpu_ref`
- different build configuration
- used to measure what the compiler can achieve automatically

This backend exists to answer a practical question: how much of the speedup can be obtained without hand-written intrinsics?

## 3. `cpu_scalar`

Directory: [`cpu_scalar/`](../../cpu_scalar)

Role:

- shared scalar implementation layer for `cpu_ref` and `cpu_auto`
- single source of truth for scalar algorithms and metric formulas

Files:

- `algos/algorithm.h`
- `algos/algorithms.h`
- `algos/algorithms.cc`
- one `.cc` file per algorithm

This layer removed the old duplication between `cpu_ref` and `cpu_auto`.

## 4. `cpu_avx512`

Directory: [`cpu_avx512/`](../../cpu_avx512)

Role:

- manual SIMD backend for the same benchmark catalog
- benchmark wrappers remain separate from low-level kernels

Split:

- [`cpu_avx512/algos/`](../../cpu_avx512/algos): wrappers, analytical metrics, checksums
- [`cpu_avx512/algorithms_opt.cc`](../../cpu_avx512/algorithms_opt.cc): AVX-512 kernels

Representative optimization style:

- packed FMA for `vecadd`
- vector reduction and horizontal combine
- gather-based `spmv`
- vectorized inner loops for `matmul`, `conv2d`, and `nbody`

## 5. `cpu_mt`

Directory: [`cpu_mt/`](../../cpu_mt)

Role:

- extend CPU scaling from SIMD-only to SIMD plus threads
- exploit all hardware threads for the benchmark catalog

Implementation:

- [`cpu_mt/cpu_mt_runner.cc`](../../cpu_mt/cpu_mt_runner.cc)
- reuses AVX-512 kernels from `cpu_avx512/algorithms_opt.cc`

Typical pattern:

- split the iteration space into chunks
- launch worker threads
- call the AVX-512 kernel on each slice where possible
- merge partial outputs if needed

`cpu_mt` is now part of the standard benchmark run and all plot scripts.

## 6. `vulkan`

Directory: [`vkbench/`](../../vkbench)

Role:

- GPU backend using Vulkan compute
- same public benchmark surface as the CPU backends

Main host-side files:

- [`vkbench/src/main.cc`](../../vkbench/src/main.cc)
- [`vkbench/src/vk_context.hpp`](../../vkbench/src/vk_context.hpp)
- [`vkbench/src/vk_buffer.hpp`](../../vkbench/src/vk_buffer.hpp)
- [`vkbench/src/vk_pipeline.hpp`](../../vkbench/src/vk_pipeline.hpp)

Shader set:

- one compute shader per algorithm
- prefix scan uses two passes, so there are 11 shader files for 10 algorithms

Design choice:

The Vulkan path favors clarity and repeatability over peak tuning. It uses host-visible buffers directly and waits for the queue to finish after dispatch.

## 7. `hybrid`

Directory: [`hybrid/`](../../hybrid)

Role:

- direct CPU+GPU co-execution backend
- concurrent CPU and GPU execution for a restricted algorithm set

Supported algorithms:

- `matmul`
- `nbody`

Implementation:

- [`hybrid/src/hybrid_api.cc`](../../hybrid/src/hybrid_api.cc)
- [`hybrid/src/hybrid_runner.cc`](../../hybrid/src/hybrid_runner.cc)

Current design:

`hybrid` keeps one shared deterministic input, partitions the output domain between CPU and GPU, and merges the final result into one checksum-equivalent output. It uses a persistent Vulkan context and dedicated pipelines instead of delegating to the other backends through `dlopen`.

## 8. `orchestrator`

Directory: [`orchestrator/`](../../orchestrator)

Role:

- dynamic loader for the backend libraries
- stdout reporter
- CSV writer

The orchestrator is deliberately thin. It depends only on the shared ABI and does not know backend internals.

## 9. Scripts and Results

### `scripts/run_bench.sh`

- configures and builds the project
- discovers available backend libraries
- runs the orchestrator
- updates `results/latest.csv`
- runs CSV validation
- generates plots

### `scripts/validate_results.py`

- recomputes expected analytical metrics from each CSV row
- checks `gflops`, `gbytes`, and timing consistency
- compares checksums for the exact-comparison backends

### Plot scripts

- [`scripts/plot_results.py`](../../scripts/plot_results.py): general plots
- [`scripts/plot_thesis.py`](../../scripts/plot_thesis.py): presentation-style figures

### `results/`

This directory stores generated CSV files, logs, and figures. It is the output sink for the benchmark workflow, not a core source directory.
