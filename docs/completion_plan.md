# Project Status and Remaining Work

## Current State

The repository is in a consistent, runnable state:

- all backends build from the root CMake project
- `cpu_ref` and `cpu_auto` no longer duplicate scalar algorithm code
- `cpu_mt` is included in benchmark runs and plotting
- CSV output includes raw `flops` and `bytes_moved`
- `scripts/validate_results.py` checks metric formulas and checksum consistency
- repository documentation is now in English
- source code and docs are ASCII-only

## What Is Already Verified

- `results/latest.csv` is regenerated from the current code
- `gflops` and `gbytes` are derived from the raw analytical counters in the CSV
- `conv2d` byte accounting includes output writes
- `spmv` byte accounting uses logical `u32` index sizes across backends
- plots read the current backend set, including `cpu_mt`

## Known Limitations

- `mem_time_ms` is still reported as `0.0` everywhere. The project does not yet split compute time from transfer or setup time.
- Vulkan measurements intentionally use a simple execution model: host-visible buffers and `vkQueueWaitIdle` after dispatch.
- `hybrid` now has exact shared-input paths for `matmul` and `nbody`.

## Recommended Next Technical Steps

1. Extend the direct shared-input `hybrid` approach from `matmul` and `nbody` to other kernels where partitioning is structurally clean.
2. Add CI that builds the project and runs `scripts/validate_results.py` on a reference CSV fixture or smoke benchmark.
3. If thesis evaluation needs it, split `mem_time_ms` into setup, transfer, and compute phases.
4. Add optional energy-measurement automation once `powerstat` or another measurement tool is available on the target machine.

## Recommended Thesis Notes

If this repository is used as the implementation base for a report or thesis, the safest claims are:

- scalar, auto-vectorized, manual AVX-512, multi-threaded CPU, and Vulkan results are directly comparable under the shared benchmark contract
- raw throughput metrics are analytically validated
- the hybrid backend is now a narrow but real correctness-equivalent benchmark for `matmul` and `nbody`
