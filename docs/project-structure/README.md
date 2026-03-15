# Project Structure Guide

This documentation set explains how `cpugpu-bench` is assembled, how benchmark data flows through the repository, and where to find each backend implementation.

## Reading Order

- [architecture.md](./architecture.md): build graph, ABI, runtime flow, metrics, and validation model
- [backends.md](./backends.md): detailed notes for each backend and its support modules
- [algorithms.md](./algorithms.md): what each workload measures and how it is implemented

## Repository Map

```text
cpugpu-bench/
|- CMakeLists.txt
|- bench_settings.h
|- common_abi/
|  |- types.h
|  |- bench_abi.h
|  `- bench_utils.hpp
|- cpu_scalar/
|  `- algos/                    shared scalar algorithm implementations
|- cpu_ref/
|  |- cpu_ref_api.cc
|  |- cpu_ref_runner.cc/.h
|  `- algos/                    thin compatibility wrappers
|- cpu_auto/
|  |- cpu_auto_api.cc
|  |- cpu_auto_runner.cc/.h
|  `- algos/                    thin compatibility wrappers
|- cpu_avx512/
|  |- cpu_avx512_api.cc
|  |- cpu_avx512_runner.cc/.h
|  |- algorithms_opt.cc/.hpp
|  `- algos/
|- cpu_mt/
|  |- cpu_mt_api.cc
|  `- cpu_mt_runner.cc/.h
|- vkbench/
|  |- shaders/
|  `- src/
|     |- main.cc
|     |- vk_context.* 
|     |- vk_buffer.*
|     `- vk_pipeline.*
|- hybrid/
|  `- src/
|     |- hybrid_api.cc
|     `- hybrid_runner.cc/.h
|- orchestrator/
|  `- orchestrator.cc
|- scripts/
|  |- run_bench.sh
|  |- validate_results.py
|  |- plot_results.py
|  `- plot_thesis.py
|- results/
`- docs/
```

## High-Level Relationship

```text
bench_settings.h + common_abi/*
            |
            v
  +-----------------------------+
  | backend shared libraries    |
  | cpu_ref cpu_auto cpu_avx512 |
  | cpu_mt vulkan hybrid        |
  +-----------------------------+
            |
      bench_get_api()
            |
            v
  +--------------------+
  | bench_orchestrator |
  +--------------------+
            |
       stdout + CSV
            |
            v
  +-------------------------------+
  | scripts/run_bench.sh          |
  | scripts/validate_results.py   |
  | scripts/plot_results.py       |
  | scripts/plot_thesis.py        |
  +-------------------------------+
            |
            v
         results/
```
