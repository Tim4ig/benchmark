# Verification Notes - cpugpu-bench (2026-03-14)

## Summary

The repository was re-audited for:

- benchmark metric correctness
- backend coverage in CSV output and plots
- checksum consistency across exact-comparison backends
- documentation accuracy
- ASCII-only source and docs

## Confirmed Fixes

### Metric accounting

- `conv2d` byte accounting now includes output writes in all relevant backends.
- `spmv` byte accounting now uses logical `u32` index sizes consistently instead of mixing pointer-size host types into the throughput metric.
- the orchestrator CSV now stores raw `flops` and `bytes_moved`, not only the derived `gflops` and `gbytes`.

### Result workflow

- `cpu_mt` is present in benchmark execution, `results/latest.csv`, and both plotting scripts.
- `scripts/run_bench.sh` now runs CSV validation automatically after a benchmark pass.
- `scripts/validate_results.py` recomputes analytical formulas from each row and verifies derived metrics.

### Hybrid backend safety

- hybrid support is restricted to direct shared-input `matmul` and `nbody`
- unsupported algorithms return an error instead of producing misleading rows

## Current Validation Policy

The following backends are treated as exact-comparison peers:

- `cpu_ref`
- `cpu_auto`
- `cpu_avx512`
- `cpu_mt`
- `vulkan`

They are compared through analytical counters and checksum tolerances.

`hybrid` is now checksum-validated for the currently supported direct paths.

## Known Remaining Limitation

The main remaining architectural limitation is hybrid coverage:

- the direct shared-input design now works for `matmul` and `nbody`
- more kernels could be migrated to the same model later

## Practical Result

The current benchmark data in `results/latest.csv` is now logically consistent for the exact-comparison backends, and the validation script can re-check that state after any future change.
