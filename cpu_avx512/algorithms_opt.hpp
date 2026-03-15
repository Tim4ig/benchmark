#pragma once

#include "../common_abi/types.h"

namespace bench {
void vecadd_cpu_opt(const f32* x, const f32* y, f32* out, f32 a, usize n);
f32 reduce_sum_cpu_opt(const f32* x, usize n);
void prefix_sum_inclusive_cpu_opt(const f32* in, f32* out, usize n);
void histogram_u32_cpu_opt(const u32* data, usize n, u32* bins, usize nbins);
void conv2d_cpu_opt(const f32* input, const f32* kernel, f32* output, usize width, usize height, usize ksize);
void spmv_csr_cpu_opt(const usize* row_ptr, const usize* col_idx, const f32* values, const f32* x, f32* y, usize rows);
void matmul_cpu_opt(const f32* a, const f32* b, f32* c, usize n);
void blackscholes_cpu_opt(const f32* s, f32* out, usize n);
void bitonic_sort_cpu_opt(f32* data, usize n);
void nbody_step_cpu_opt(const f32* px, const f32* py, const f32* mass, f32* fx, f32* fy, usize n);
} // namespace bench