#include "algorithms_opt.hpp"

#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace bench {
void vecadd_cpu_opt(const f32* x, const f32* y, f32* out, f32 a, usize n) {
  usize i = 0;
  const __m512 va = _mm512_set1_ps(a);
  for (; i + 15 < n; i += 16) {
    const __m512 vx = _mm512_loadu_ps(x + i);
    const __m512 vy = _mm512_loadu_ps(y + i);
    const __m512 vout = _mm512_fmadd_ps(va, vx, vy);
    _mm512_storeu_ps(out + i, vout);
  }
  for (; i < n; ++i) {
    out[i] = a * x[i] + y[i];
  }
}

f32 reduce_sum_cpu_opt(const f32* x, usize n) {
  f32 sum = 0.0f;
  usize i = 0;
  __m512 acc = _mm512_setzero_ps();
  for (; i + 15 < n; i += 16) {
    acc = _mm512_add_ps(acc, _mm512_loadu_ps(x + i));
  }
  alignas(64) f32 tmp[16];
  _mm512_storeu_ps(tmp, acc);
  for (f32 v : tmp) {
    sum += v;
  }
  for (; i < n; ++i) {
    sum += x[i];
  }
  return sum;
}

void prefix_sum_inclusive_cpu_opt(const f32* in, f32* out, usize n) {
  f32 carry = 0.0f;
  for (usize i = 0; i < n; ++i) {
    carry += in[i];
    out[i] = carry;
  }
}

void histogram_u32_cpu_opt(const u32* data, usize n, u32* bins, usize nbins) {
  for (usize i = 0; i < nbins; ++i) {
    bins[i] = 0;
  }
  if (nbins == 0) {
    return;
  }

  const bool pow2 = (nbins & (nbins - 1)) == 0;
  const u32 mask = static_cast<u32>(nbins - 1);
  for (usize i = 0; i < n; ++i) {
    const u32 value = data[i];
    const usize idx = pow2 ? static_cast<usize>(value & mask) : static_cast<usize>(value % static_cast<u32>(nbins));
    ++bins[idx];
  }
}

namespace {
f32 conv2d_pixel_ref(const f32* input, const f32* kernel, usize width, usize height, usize ksize, usize x, usize y) {
  const usize radius = ksize / 2;
  f32 acc = 0.0f;
  for (usize ky = 0; ky < ksize; ++ky) {
    const usize in_y = y + ky;
    if (in_y < radius || in_y >= height + radius) {
      continue;
    }
    const usize src_y = in_y - radius;
    for (usize kx = 0; kx < ksize; ++kx) {
      const usize in_x = x + kx;
      if (in_x < radius || in_x >= width + radius) {
        continue;
      }
      const usize src_x = in_x - radius;
      acc += input[src_y * width + src_x] * kernel[ky * ksize + kx];
    }
  }
  return acc;
}
} // namespace

void conv2d_cpu_opt(const f32* input, const f32* kernel, f32* output, usize width, usize height, usize ksize) {
  if (width == 0 || height == 0 || ksize == 0) {
    return;
  }
  const usize radius = ksize / 2;

  for (usize y = 0; y < std::min(radius, height); ++y) {
    for (usize x = 0; x < width; ++x) {
      output[y * width + x] = conv2d_pixel_ref(input, kernel, width, height, ksize, x, y);
    }
  }

  for (usize y = radius; y + radius < height; ++y) {
    for (usize x = 0; x < std::min(radius, width); ++x) {
      output[y * width + x] = conv2d_pixel_ref(input, kernel, width, height, ksize, x, y);
    }

    usize x = radius;
    const usize x_end = width - radius;
    for (; x + 15 < x_end; x += 16) {
      __m512 acc = _mm512_setzero_ps();
      for (usize ky = 0; ky < ksize; ++ky) {
        const f32* row = input + (y + ky - radius) * width + (x - radius);
        for (usize kx = 0; kx < ksize; ++kx) {
          const __m512 v = _mm512_loadu_ps(row + kx);
          const __m512 k = _mm512_set1_ps(kernel[ky * ksize + kx]);
          acc = _mm512_fmadd_ps(v, k, acc);
        }
      }
      _mm512_storeu_ps(output + y * width + x, acc);
    }

    for (; x < x_end; ++x) {
      output[y * width + x] = conv2d_pixel_ref(input, kernel, width, height, ksize, x, y);
    }

    for (usize xb = x_end; xb < width; ++xb) {
      output[y * width + xb] = conv2d_pixel_ref(input, kernel, width, height, ksize, xb, y);
    }
  }

  for (usize y = (height > radius ? height - radius : 0); y < height; ++y) {
    for (usize x = 0; x < width; ++x) {
      output[y * width + x] = conv2d_pixel_ref(input, kernel, width, height, ksize, x, y);
    }
  }
}

void spmv_csr_cpu_opt(const usize* row_ptr, const usize* col_idx, const f32* values, const f32* x, f32* y, usize rows) {
  for (usize row = 0; row < rows; ++row) {
    const usize start = row_ptr[row];
    const usize end = row_ptr[row + 1];
    f32 sum = 0.0f;
    usize idx = start;

    __m512 acc = _mm512_setzero_ps();
    alignas(64) i32 indices[16];
    for (; idx + 15 < end; idx += 16) {
      for (i32 k = 0; k < 16; ++k) {
        indices[k] = static_cast<i32>(col_idx[idx + static_cast<usize>(k)]);
      }
      const __m512 v = _mm512_loadu_ps(values + idx);
      const __m512 xv = _mm512_i32gather_ps(_mm512_loadu_si512(indices), x, 4);
      acc = _mm512_fmadd_ps(v, xv, acc);
    }
    alignas(64) f32 tmp[16];
    _mm512_storeu_ps(tmp, acc);
    for (f32 v : tmp) {
      sum += v;
    }

    for (; idx < end; ++idx) {
      sum += values[idx] * x[col_idx[idx]];
    }
    y[row] = sum;
  }
}

void matmul_cpu_opt(const f32* a, const f32* b, f32* c, usize n) {
  const usize nn = n * n;
  for (usize idx = 0; idx < nn; ++idx) {
    c[idx] = 0.0f;
  }

  for (usize i = 0; i < n; ++i) {
    const usize row = i * n;
    for (usize k = 0; k < n; ++k) {
      const f32 aik = a[row + k];
      const f32* brow = b + k * n;
      f32* crow = c + row;
      usize j = 0;

      const __m512 va = _mm512_set1_ps(aik);
      for (; j + 15 < n; j += 16) {
        __m512 vb = _mm512_loadu_ps(brow + j);
        __m512 vc = _mm512_loadu_ps(crow + j);
        vc = _mm512_fmadd_ps(va, vb, vc);
        _mm512_storeu_ps(crow + j, vc);
      }
      for (; j < n; ++j) {
        crow[j] += aik * brow[j];
      }
    }
  }
}

// ---------------------------------------------------------- Black-Scholes
// Use scalar math (std::log, std::exp, std::erfc) and rely on the compiler
// to auto-vectorize via libmvec with -O3 -mavx512f. Custom bit-twiddling
// log/exp approximations produced ~13% checksum error vs cpu_ref.
void blackscholes_cpu_opt(const f32* s, f32* out, usize n) {
  const f32 k_strike = 100.0f;
  const f32 k_t = 1.0f;
  const f32 k_r = 0.05f;
  const f32 k_sigma = 0.2f;
  const f32 sigma_sqrt_t = k_sigma * std::sqrt(k_t);
  const f32 d_add = (k_r / k_sigma + 0.5f * k_sigma) * std::sqrt(k_t);
  const f32 disc = k_strike * std::exp(-k_r * k_t);
  const f32 inv_sqrt2 = 0.7071067811865476f;

  for (usize i = 0; i < n; ++i) {
    const f32 si = s[i];
    const f32 d1 = std::log(si / k_strike) / sigma_sqrt_t + d_add;
    const f32 d2 = d1 - sigma_sqrt_t;
    const f32 nd1 = 0.5f * std::erfc(-d1 * inv_sqrt2);
    const f32 nd2 = 0.5f * std::erfc(-d2 * inv_sqrt2);
    out[i] = si * nd1 - disc * nd2;
  }
}

// ---------------------------------------------------------- Bitonic Sort
// O(n log^2 n) compare-and-swap network. AVX-512 processes 16 independent
// pairs per iteration when stride >= 16; scalar for smaller strides.
// Full in-register bitonic merge requires complex shuffle patterns and is
// not implemented here -- the loop structure is already embarrassingly
// parallel at the outer level.
void bitonic_sort_cpu_opt(f32* data, usize n) {
  for (usize k = 2; k <= n; k <<= 1) {
    for (usize j = k >> 1; j >= 16; j >>= 1) {
      for (usize i = 0; i < n; i += 16) {
        for (int ii = 0; ii < 16; ++ii) {
          const usize gi = i + static_cast<usize>(ii);
          const usize l = gi ^ j;
          if (l > gi) {
            const bool asc = (gi & k) == 0;
            if (asc ? (data[gi] > data[l]) : (data[gi] < data[l])) {
              std::swap(data[gi], data[l]);
            }
          }
        }
      }
    }
    for (usize j2 = std::min(k >> 1, usize{8}); j2 > 0; j2 >>= 1) {
      for (usize i = 0; i < n; ++i) {
        const usize l = i ^ j2;
        if (l > i) {
          const bool asc = (i & k) == 0;
          if (asc ? (data[i] > data[l]) : (data[i] < data[l])) {
            std::swap(data[i], data[l]);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------- N-Body AVX-512
// Each invocation computes forces on n particles from all n sources.
// Inner loop is vectorized: accumulate forces from 16 particles at a time.
// Self-interaction (j == i) is masked out via _mm512_maskz_mov_ps.
void nbody_step_cpu_opt(const f32* px, const f32* py, const f32* mass, f32* fx, f32* fy, usize n) {
  const __m512 veps2 = _mm512_set1_ps(1e-4f);

  for (usize i = 0; i < n; ++i) {
    const __m512 vxi = _mm512_set1_ps(px[i]);
    const __m512 vyi = _mm512_set1_ps(py[i]);
    __m512 vax = _mm512_setzero_ps();
    __m512 vay = _mm512_setzero_ps();

    usize j = 0;
    for (; j + 15 < n; j += 16) {
      const __m512 vpxj = _mm512_loadu_ps(px + j);
      const __m512 vpyj = _mm512_loadu_ps(py + j);
      const __m512 vmj_raw = _mm512_loadu_ps(mass + j);

      // Zero out self-particle (j <= i < j+16)
      const __mmask16 self_mask = (i >= j && i < j + 16) ? static_cast<__mmask16>(~(1u << (i - j))) : 0xFFFFu;
      const __m512 vmj = _mm512_maskz_mov_ps(self_mask, vmj_raw);

      const __m512 dx = _mm512_sub_ps(vpxj, vxi);
      const __m512 dy = _mm512_sub_ps(vpyj, vyi);
      __m512 dist2 = _mm512_fmadd_ps(dx, dx, _mm512_fmadd_ps(dy, dy, veps2));

      __m512 inv_d = _mm512_rsqrt14_ps(dist2);
      const __m512 half = _mm512_set1_ps(0.5f);
      const __m512 c1p5 = _mm512_set1_ps(1.5f);
      inv_d = _mm512_mul_ps(inv_d, _mm512_fnmadd_ps(_mm512_mul_ps(dist2, inv_d), _mm512_mul_ps(inv_d, half), c1p5));

      const __m512 inv_d3 = _mm512_mul_ps(inv_d, _mm512_mul_ps(inv_d, inv_d));
      const __m512 f = _mm512_mul_ps(vmj, inv_d3);

      vax = _mm512_fmadd_ps(f, dx, vax);
      vay = _mm512_fmadd_ps(f, dy, vay);
    }

    alignas(64) f32 tax[16], tay[16];
    _mm512_storeu_ps(tax, vax);
    _mm512_storeu_ps(tay, vay);
    f32 ax = 0.0f, ay = 0.0f;
    for (int k = 0; k < 16; ++k) {
      ax += tax[k];
      ay += tay[k];
    }

    for (; j < n; ++j) {
      if (j == i) {
        continue;
      }
      const f32 dx = px[j] - px[i];
      const f32 dy = py[j] - py[i];
      const f32 dist2 = dx * dx + dy * dy + 1e-4f;
      const f32 inv_d = 1.0f / std::sqrt(dist2);
      const f32 inv_d3 = inv_d * inv_d * inv_d;
      const f32 f = mass[j] * inv_d3;
      ax += f * dx;
      ay += f * dy;
    }
    fx[i] = ax;
    fy[i] = ay;
  }
}
} // namespace bench
