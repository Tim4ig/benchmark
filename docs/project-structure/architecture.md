# Architecture

## 1. Root Build Graph

The root [`CMakeLists.txt`](../../CMakeLists.txt) adds these subdirectories:

- `cpu_ref`
- `cpu_avx512`
- `cpu_auto`
- `vkbench`
- `cpu_mt`
- `hybrid`
- `orchestrator`

That produces one executable, `bench_orchestrator`, and six loadable backend libraries.

## 2. Shared ABI

The stable boundary between the orchestrator and every backend is [`common_abi/bench_abi.h`](../../common_abi/bench_abi.h).

Core ABI types:

- `BenchAlgo`: algorithm identifiers
- `BenchOptions`: algorithm selection plus runtime parameters
- `BenchResult`: timing, analytical counters, checksum, and status
- `BenchEntry`: exported algorithm name plus enum
- `BenchApi`: function table returned by `bench_get_api()`

Every backend exports exactly one symbol:

```c++
extern "C" const BenchApi* bench_get_api();
```

The orchestrator does not need to know any backend internals beyond that.

## 3. Shared Defaults and Utilities

### `bench_settings.h`

[`bench_settings.h`](../../bench_settings.h) defines:

- default repeat count
- default random seed
- per-algorithm default problem sizes
- default auxiliary parameters such as `bins`, `nnz_per_row`, and `ksize`

The orchestrator starts from `bench::default_options()`, selects one `BenchAlgo`, and lets `bench::apply_algorithm_defaults()` fill the rest. This keeps the effective workload consistent across backends.

### `common_abi/bench_utils.hpp`

[`common_abi/bench_utils.hpp`](../../common_abi/bench_utils.hpp) provides the shared CPU-side support layer:

- deterministic random float generation
- deterministic random `u32` generation
- deterministic CSR matrix generation for `spmv`
- normalized kernel generation for `conv2d`
- the common timing helper `measure_ms`
- checksum helpers

The Vulkan backend implements equivalent host-side helpers locally in [`vkbench/src/main.cc`](../../vkbench/src/main.cc) so that it can prepare buffers without depending on CPU backend code.

## 4. Backend Layering

### Scalar layer: `cpu_scalar`

[`cpu_scalar/`](../../cpu_scalar) contains the shared scalar algorithm implementations used by both `cpu_ref` and `cpu_auto`.

This keeps one source of truth for:

- scalar algorithm logic
- analytical FLOP and byte formulas
- checksum behavior

`cpu_ref` and `cpu_auto` differ mainly by compile flags and backend identity, not by algorithm source.

### Backend wrappers

Each backend still has its own:

- `*_api.cc`: exports backend name and supported algorithms
- `*_runner.cc/.h`: resolves options and dispatches one algorithm

This keeps the plugin boundary separate from the algorithm code.

## 5. CPU Backend Families

### `cpu_ref`

- wraps `cpu_scalar`
- built with minimal optimization
- used as the reference baseline for plots and checksum comparison

### `cpu_auto`

- wraps `cpu_scalar`
- built with `-O3 -march=native -ftree-vectorize -ffast-math`
- answers the question: how far can the compiler go without manual intrinsics?

### `cpu_avx512`

- has its own benchmark wrappers in [`cpu_avx512/algos/`](../../cpu_avx512/algos)
- calls low-level kernels from [`cpu_avx512/algorithms_opt.cc`](../../cpu_avx512/algorithms_opt.cc)
- uses explicit AVX-512 intrinsics for the vector-friendly kernels

### `cpu_mt`

- implemented in [`cpu_mt/cpu_mt_runner.cc`](../../cpu_mt/cpu_mt_runner.cc)
- reuses AVX-512 kernels from `cpu_avx512`
- adds thread-level parallelism on top of those kernels
- is the strongest CPU backend for many compute-heavy workloads on the target machine

## 6. Vulkan Backend

The GPU backend is implemented under [`vkbench/`](../../vkbench).

Host-side structure:

- [`vkbench/src/main.cc`](../../vkbench/src/main.cc): backend API plus per-algorithm run functions
- [`vkbench/src/vk_context.*`](../../vkbench/src/vk_context.hpp): instance, device, queue, and command buffer setup
- [`vkbench/src/vk_buffer.*`](../../vkbench/src/vk_buffer.hpp): buffer allocation and host-visible memory helpers
- [`vkbench/src/vk_pipeline.*`](../../vkbench/src/vk_pipeline.hpp): descriptor set layout, pipeline layout, and pipeline creation

Shader-side structure:

- one GLSL compute shader per algorithm
- prefix scan uses two shaders: `scan_block.comp` and `scan_add.comp`

Runtime model:

- allocate host-visible buffers
- upload generated inputs
- bind a generic descriptor set
- push algorithm parameters through push constants
- dispatch one or more compute pipelines
- wait for queue completion
- read back outputs
- compute checksum and metrics on the host

This is intentionally simple and reproducible, not a fully optimized Vulkan data path.

## 7. Hybrid Backend

The hybrid backend lives in [`hybrid/src/`](../../hybrid/src).

Current design:

- creates its own persistent Vulkan context and pipelines
- supports `matmul` and `nbody`
- generates one shared deterministic input per benchmark run
- splits the output domain between CPU and GPU
- runs both sides concurrently inside one backend
- combines the final output into one checksum-equivalent result

## 8. Orchestrator Flow

[`orchestrator/orchestrator.cc`](../../orchestrator/orchestrator.cc) is the benchmark driver.

For each loaded backend library it:

1. opens the shared object with `dlopen`
2. resolves `bench_get_api`
3. reads the backend name and exported algorithm entries
4. creates default options
5. applies per-algorithm defaults
6. runs each exported algorithm once
7. prints a human-readable line to stdout
8. writes a CSV row if `--csv` was requested

The orchestrator now writes both derived metrics and raw analytical counters:

- `total_ms`
- `calc_ms`
- `mem_ms`
- `gflops`
- `gbytes`
- `checksum`
- `flops`
- `bytes_moved`
- `status`

## 9. Timing and Metric Model

Across the backends, the intended meaning is:

- `total_ms`: total wall-clock time reported by the backend; for CPU backends this is effectively `calc_ms * repeats`, while GPU and hybrid backends also include one-time transfer/setup overhead outside the repeated loop
- `calc_ms`: average wall-clock time of one repeated dispatch/compute iteration
- `mem_ms`: transfer/setup overhead kept outside that repeated loop; for most GPU paths it is dominated by H2D + D2H, with a few kernel-specific extras
- `flops`: analytical operation count for the configured problem size
- `bytes_moved`: analytical logical data movement for the configured problem size
- `gflops = flops / (calc_ms * 1e6)`
- `gbytes = bytes_moved / (calc_ms * 1e6)`

## 10. Validation Pipeline

The result pipeline is:

1. [`scripts/run_bench.sh`](../../scripts/run_bench.sh)
2. [`bench_orchestrator`](../../orchestrator/orchestrator.cc)
3. `results/latest.csv`
4. [`scripts/validate_results.py`](../../scripts/validate_results.py)
5. [`scripts/plot_results.py`](../../scripts/plot_results.py)
6. [`scripts/plot_thesis.py`](../../scripts/plot_thesis.py)

The validator checks:

- analytical `flops`
- analytical `bytes_moved`
- derived `gflops`
- derived `gbytes`
- `calc_ms ~= total_ms / repeats` for rows where `mem_ms == 0`; GPU and hybrid rows are validated with `mem_ms` separated from the repeated loop
- checksum consistency for the exact-comparison backends

Hybrid rows are validated normally for the currently supported direct paths.

## 11. Practical Consequences

The repository is now consistent in these ways:

- CPU scalar logic is not duplicated between `cpu_ref` and `cpu_auto`
- `cpu_mt` is part of both benchmark execution and plotting
- GPU and hybrid rows separate repeated-loop time from external transfer/setup overhead
- raw counters are stored in CSV, so throughput can be audited after the run
- plots and documentation reflect the current backend set

The main architectural gap that remains is extending the same direct shared-input strategy to more kernels.
