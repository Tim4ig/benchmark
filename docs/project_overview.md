# cpugpu-bench Project Overview

## Scope

This repository compares CPU, GPU, and mixed CPU+GPU implementations of the same benchmark catalog. The goals are:

- keep one shared benchmark ABI across all backends
- run the same workloads with consistent default sizes
- report both timing and analytical throughput metrics
- make it easy to compare scalar, auto-vectorized, manual SIMD, multi-threaded CPU, and Vulkan compute paths

## Current Hardware

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads, AVX-512 capable |
| GPU | AMD Radeon RX 9060 XT 16 GB |
| OS | Linux |
| Graphics API | Vulkan |

## Backend Summary

| Backend | Artifact | Goal | Main implementation style |
| --- | --- | --- | --- |
| `cpu_ref` | `libcpu_ref.so` | readable scalar baseline | shared scalar code built with minimal optimization |
| `cpu_auto` | `libcpu_auto.so` | compiler-driven optimization baseline | same scalar code, built with `-O3 -march=native` |
| `cpu_avx512` | `libcpu_avx512.so` | manual SIMD tuning | explicit AVX-512 kernels plus benchmark wrappers |
| `cpu_mt` | `libcpu_mt.so` | scale CPU work across all cores | multi-threaded wrappers around AVX-512 kernels |
| `vulkan` | `libvkbench.so` | GPU compute backend | Vulkan host code plus GLSL compute shaders |
| `hybrid` | `libhybrid.so` | direct CPU+GPU co-execution | shared-input split for selected compute-heavy kernels |

## Algorithm Summary

| Algorithm | Default size | Dominant bottleneck | What it needs most |
| --- | --- | --- | --- |
| `vecadd` | `n = 2^20` | memory bandwidth | wide loads, stores, and streaming writes |
| `reduce` | `n = 2^20` | memory bandwidth plus accumulation | efficient partial reduction |
| `prefix` | `n = 2^20` | loop-carried dependency | good scan decomposition |
| `hist` | `n = 2^20`, `bins = 256` | write contention | atomic throughput or private bins |
| `conv2d` | `n = 1024`, `ksize = 3` | stencil access pattern | locality and efficient multiply-accumulate |
| `spmv` | `n = 2^18`, `nnz_per_row = 16` | irregular memory access | gather efficiency and cache behavior |
| `matmul` | `n = 512` | arithmetic throughput | reuse and dense compute |
| `blackscholes` | `n = 2^20` | transcendental math throughput | high compute-to-memory ratio |
| `bsort` | `n = 2^20` | staged synchronization | predictable compare-swap scheduling |
| `nbody` | `n = 4096` | all-pairs compute | very high arithmetic intensity |

Detailed per-algorithm notes live in [project-structure/algorithms.md](./project-structure/algorithms.md).

## Verified Result Notes

The current result pipeline now validates:

- raw analytical `flops`
- raw analytical `bytes_moved`
- derived `gflops` and `gbytes`
- `calc_ms ~= total_ms / repeats`
- checksum consistency across the exact-comparison backends

That validation is implemented in [`scripts/validate_results.py`](../scripts/validate_results.py) and runs automatically from [`scripts/run_bench.sh`](../scripts/run_bench.sh).

High-level observations from the latest validated run on this machine:

- `cpu_auto` and `cpu_avx512` are often close on simple streaming kernels, which suggests the compiler handles those loops well.
- `cpu_mt` is the strongest CPU backend for large compute-heavy kernels because it combines AVX-512 with thread-level parallelism.
- `vulkan` is strongest on dense compute kernels such as `matmul` and `blackscholes`.
- `spmv` remains difficult for the GPU backend because the implementation is intentionally simple and the access pattern is irregular.
- `hybrid` now has exact shared-input paths for `matmul` and `nbody`, which makes those rows directly comparable to the other backends.

## Repository Layout

```text
cpugpu-bench/
|- common_abi/              shared ABI, types, timing, generators, checksums
|- bench_settings.h         default benchmark sizes and option policy
|- cpu_scalar/              shared scalar algorithm implementations
|- cpu_ref/                 scalar baseline backend
|- cpu_auto/                auto-vectorized backend
|- cpu_avx512/              manual SIMD backend
|- cpu_mt/                  multi-threaded AVX-512 backend
|- vkbench/                 Vulkan backend and shaders
|- hybrid/                  direct hybrid backend for selected kernels
|- orchestrator/            plugin loader and CSV writer
|- scripts/                 build, run, plot, and validation helpers
|- results/                 generated CSV files, logs, and plots
`- docs/                    repository documentation
```
