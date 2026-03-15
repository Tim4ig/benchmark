# Algorithm Catalog

This document explains what each benchmark computes, what resource usually dominates it, and how the repository models its analytical metrics.

## Overview Table

| Algorithm | Default parameters | Complexity | Dominant resource | Logical FLOPs | Logical bytes |
| --- | --- | --- | --- | --- | --- |
| `vecadd` | `n = 2^20` | `O(n)` | memory bandwidth | `2 * n` | `3 * n * sizeof(float)` |
| `reduce` | `n = 2^20` | `O(n)` | memory bandwidth plus accumulation | `n` | `n * sizeof(float)` |
| `prefix` | `n = 2^20` | `O(n)` | dependency management | `n` | `2 * n * sizeof(float)` |
| `hist` | `n = 2^20`, `bins = 256` | `O(n)` | write contention | `n` | `n * sizeof(u32) + bins * sizeof(u32)` |
| `conv2d` | `n = 1024`, `ksize = 3` | `O(n^2 * k^2)` | stencil locality | `2 * ksize^2 * n^2` | `(2 * n^2 + ksize^2) * sizeof(float)` |
| `spmv` | `n = 2^18`, `nnz_per_row = 16` | `O(nnz)` | irregular memory access | `2 * nnz_total` | values + indices + row pointers + input + output |
| `matmul` | `n = 512` | `O(n^3)` | compute throughput | `2 * n^3` | `3 * n^2 * sizeof(float)` |
| `blackscholes` | `n = 2^20` | `O(n)` | transcendental math throughput | `50 * n` | `2 * n * sizeof(float)` |
| `bsort` | `n = 2^20` | `O(n log^2 n)` | staged synchronization | bitonic comparisons | `2 * next_pow2(n) * sizeof(float)` |
| `nbody` | `n = 4096` | `O(n^2)` | arithmetic intensity | `20 * n * (n - 1)` | `5 * n * sizeof(float)` |

## Input Generation Rules

The repository uses deterministic synthetic inputs:

- float arrays come from a fixed-seed PRNG
- histogram inputs are deterministic `u32` values
- `spmv` matrices are generated in CSR form with deterministic row structure
- `conv2d` kernels are generated once and normalized
- `nbody` uses deterministic positions and masses sampled from fixed ranges

This matters because checksum comparison only makes sense when every backend sees the same logical input for the same benchmark row.

## 1. `vecadd`

Computation:

```text
out[i] = alpha * x[i] + y[i]
```

What it stresses:

- streaming reads and writes
- SIMD width
- memory bandwidth

Implementation notes:

- scalar backends use a plain loop
- `cpu_avx512` uses packed multiply-add
- `cpu_mt` splits the vector into thread slices
- Vulkan launches one thread per element
- hybrid splits the length into CPU and GPU regions

Usually needs more:

- bandwidth, not control logic

## 2. `reduce`

Computation:

```text
sum = sum_i x[i]
```

What it stresses:

- partial accumulation
- reduction tree shape
- memory bandwidth

Implementation notes:

- scalar backends accumulate in one loop
- `cpu_avx512` accumulates packed lanes before a horizontal combine
- `cpu_mt` computes per-thread partial sums and merges them on the host
- Vulkan reduces per workgroup and finishes on the host

Usually needs more:

- bandwidth and efficient partial aggregation

## 3. `prefix`

Computation:

```text
out[i] = x[0] + x[1] + ... + x[i]
```

What it stresses:

- loop-carried dependency
- scan decomposition
- synchronization between partial scans

Implementation notes:

- scalar path is a straightforward inclusive scan
- `cpu_mt` uses a three-pass chunked scan
- Vulkan uses block scan, host-side block-prefix stitching, then offset addition

Usually needs more:

- structured dependency handling, not raw arithmetic throughput

## 4. `hist`

Computation:

- count how many input values map to each bin

What it stresses:

- atomic or contended updates
- privatization strategies
- merge cost

Implementation notes:

- scalar backends increment bins directly
- `cpu_mt` uses private per-thread histograms and merges them
- Vulkan uses `atomicAdd` in the shader

Usually needs more:

- contention management

## 5. `conv2d`

Computation:

- dense 2D convolution over an `n x n` image with an odd `ksize x ksize` kernel

What it stresses:

- stencil reuse
- border handling
- multiply-accumulate throughput

Implementation notes:

- scalar backends use direct nested loops
- `cpu_avx512` vectorizes the interior x-range
- `cpu_mt` splits rows across threads
- Vulkan maps one thread to one output pixel

Usually needs more:

- locality and efficient MAC loops

## 6. `spmv`

Computation:

```text
y = A * x
```

with `A` stored in CSR form.

What it stresses:

- indirect addressing
- sparse row traversal
- gather behavior

Implementation notes:

- scalar backends iterate row by row
- `cpu_avx512` uses gather loads for `x[col_idx[idx]]`
- `cpu_mt` parallelizes over rows
- Vulkan maps one thread to one row

Usually needs more:

- cache efficiency and low-overhead irregular access

## 7. `matmul`

Computation:

```text
C = A * B
```

for dense square matrices.

What it stresses:

- arithmetic throughput
- reuse of loaded values
- loop ordering

Implementation notes:

- scalar backends use direct triple loops
- `cpu_avx512` broadcasts one scalar from `A` and updates vector chunks of `C`
- `cpu_mt` splits output rows across threads
- Vulkan maps threads to output elements
- hybrid splits the output rows between CPU and GPU while both read the same full `A` and `B`

Usually needs more:

- compute throughput and data reuse

## 8. `blackscholes`

Computation:

- option pricing based on the Black-Scholes formula

What it stresses:

- transcendental math throughput
- vector math quality
- compute-to-memory ratio

Implementation notes:

- scalar backends use scalar math calls
- `cpu_avx512` keeps the benchmark wrapper separate from the optimized kernel path
- `cpu_mt` splits the option list across threads
- Vulkan maps one thread to one option
- hybrid supports this kernel because each option is independent

Usually needs more:

- compute throughput, not bandwidth

## 9. `bsort`

Computation:

- bitonic sorting network over `next_pow2(n)` elements

What it stresses:

- stage-by-stage synchronization
- compare-swap control flow
- many repeated passes over the same data

Implementation notes:

- scalar and AVX-512 CPU versions keep the control-flow-heavy structure
- `cpu_mt` parallelizes independent compare-swap pairs per stage
- Vulkan launches one pass per `(stage, step)` pair

Usually needs more:

- predictable staging and low-overhead synchronization

## 10. `nbody`

Computation:

- all-pairs force accumulation for `n` particles

What it stresses:

- arithmetic intensity
- nested-loop throughput
- inner-loop vectorization

Implementation notes:

- scalar backends perform a direct all-pairs loop
- `cpu_avx512` vectorizes the inner source-particle loop
- `cpu_mt` parallelizes over target particles
- Vulkan maps one thread to one target particle and loops over all sources
- hybrid splits target particles between CPU and GPU while both read the same full particle set

Usually needs more:

- compute throughput

## Hybrid Support Notes

The hybrid backend currently exports:

- `matmul`
- `nbody`

These are the two kernels where the repository now uses a direct shared-input split:

- GPU computes a prefix region of the output domain
- CPU computes the remaining suffix region
- both sides read the same deterministic input
- the final checksum is directly comparable to the other backends
