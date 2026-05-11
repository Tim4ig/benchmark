# cpugpu-bench

`cpugpu-bench` is a plugin-based benchmark suite for comparing the same algorithm set across several backends:

- `cpu_ref`: scalar baseline, built for readability and correctness
- `cpu_auto`: the same scalar algorithm layer, compiled with aggressive optimization and auto-vectorization
- `cpu_avx512`: manually vectorized AVX-512 kernels
- `cpu_mt`: multi-threaded CPU backend that reuses the AVX-512 kernels
- `vulkan`: GPU backend implemented with Vulkan compute shaders
- `hybrid`: direct CPU+GPU co-execution backend for selected compute-heavy kernels

The project is organized around one shared C ABI. Every backend is built as a `.so` and loaded at runtime by `bench_orchestrator`.

## Algorithms

The benchmark catalog currently includes 10 workloads:

- `vecadd`
- `reduce`
- `prefix`
- `hist`
- `conv2d`
- `spmv`
- `matmul`
- `blackscholes`
- `bsort`
- `nbody`

Default problem sizes live in [`bench_settings.h`](./bench_settings.h). CPU reference and auto-vectorized implementations share one source tree in [`cpu_scalar/`](./cpu_scalar).

## Build Requirements

- CMake 3.16+
- A C++17 compiler
- Python 3 for plotting and CSV validation
- `matplotlib` and `numpy` for plots
- Vulkan runtime and development headers for `vkbench`
- `glslangValidator` to compile GLSL shaders to SPIR-V

## Build

```bash
cmake -S . -B build/root -DCMAKE_BUILD_TYPE=Release
cmake --build build/root -j"$(nproc)"
```

## Run

The easiest path is the automation script:

```bash
./scripts/run_bench.sh build/root results
```

That script:

- builds the project
- attempts to switch the host into a performance-oriented measurement state (`sudo`, governor/EPP/RAPL/GPU cap)
- runs all discovered backends
- uses time-based execution by default: cold calibration, warm-up of about `4.5 s`, then a measured phase of about `5.0 s`
- writes a timestamped CSV and updates `results/latest.csv`
- validates analytical metrics and checksum consistency
- generates plots in `results/plots_<timestamp>/`

You can also run the orchestrator directly:

```bash
./build/root/orchestrator/bench_orchestrator \
  --csv results/manual.csv \
  ./build/root/cpu_ref/libcpu_ref.so \
  ./build/root/cpu_auto/libcpu_auto.so \
  ./build/root/cpu_avx512/libcpu_avx512.so \
  ./build/root/cpu_mt/libcpu_mt.so \
  ./build/root/vkbench/libvkbench.so \
  ./build/root/hybrid/libhybrid.so
```

## Result Semantics

Each CSV row contains:

- `repeats`: automatically chosen repeat count for the measured phase in the default time-based mode
- `total_ms`: end-to-end wall time for the measured phase; for GPU and hybrid backends this includes the one-time H2D/D2H transfers
- `calc_ms`: average per-repeat dispatch-loop wall time during the measured phase
- `mem_ms`: transfer/setup overhead kept outside the repeated dispatch loop; for most GPU paths this is H2D + D2H, with a few kernel-specific extras such as histogram bin reset/upload
- `flops`: analytical operation count for the benchmarked workload
- `bytes_moved`: analytical logical data movement for the same workload
- `gflops`: `flops / (calc_ms * 1e6)`
- `gbytes`: `bytes_moved / (calc_ms * 1e6)`
- `checksum`: lightweight output sanity signal
- `watts_cpu`, `watts_gpu`: average power sampled by the orchestrator around the backend `run()` call; this covers the measured loop plus backend-local setup/finalization, and for `hybrid` also its internal adaptive calibration
- `status`: backend return code (`0` = success)

The CSV file also starts with `# git_hash=...`, `# build_ts=...`, and `# compiler=...` comment lines so each measurement campaign stays tied to one concrete build.

The validator is available as a standalone script:

```bash
python3 scripts/validate_results.py results/latest.csv
```

## Important Note About `hybrid`

`hybrid` now implements direct shared-input CPU+GPU execution for:

- `matmul`
- `nbody`

These paths keep one deterministic input and split the output domain between CPU and GPU, so their checksum is directly comparable to the other backends.

The CPU/GPU split is always adaptive. `HYBRID_CPU_RATIO` is only an initial hint for the starting ratio; the runtime still recalibrates and updates the split from measured CPU and GPU throughput.

The hybrid backend is still intentionally narrow in scope. It only exposes kernels where the split is structurally clean and worth demonstrating.

## Documentation

- [Project overview](./docs/project_overview.md)
- [Project structure guide](./docs/project-structure/README.md)
- [Architecture notes](./docs/project-structure/architecture.md)
- [Backend notes](./docs/project-structure/backends.md)
- [Algorithm notes](./docs/project-structure/algorithms.md)
- [Project status](./docs/completion_plan.md)
- [Verification notes](./docs/code_review.md)

## License

`SPDX-License-Identifier: MIT`

Copyright (c) 2025-2026 Tymur Kramar. See [LICENSE](./LICENSE) for details.
