#include "algorithms_opt.hpp"

#include <algorithm>
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
    const usize idx = pow2 ? static_cast<usize>(value & mask)
                           : static_cast<usize>(value % static_cast<u32>(nbins));
    ++bins[idx];
  }
}

namespace {

f32 Conv2dPixelRef(const f32* input, const f32* kernel, usize width, usize height, usize ksize,
                   usize x, usize y) {
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

}  // namespace

void conv2d_cpu_opt(const f32* input, const f32* kernel, f32* output, usize width, usize height,
                    usize ksize) {
  if (width == 0 || height == 0 || ksize == 0) {
    return;
  }
  const usize radius = ksize / 2;

  for (usize y = 0; y < std::min(radius, height); ++y) {
    for (usize x = 0; x < width; ++x) {
      output[y * width + x] = Conv2dPixelRef(input, kernel, width, height, ksize, x, y);
    }
  }

  for (usize y = radius; y + radius < height; ++y) {
    for (usize x = 0; x < std::min(radius, width); ++x) {
      output[y * width + x] = Conv2dPixelRef(input, kernel, width, height, ksize, x, y);
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
      output[y * width + x] = Conv2dPixelRef(input, kernel, width, height, ksize, x, y);
    }

    for (usize xb = x_end; xb < width; ++xb) {
      output[y * width + xb] = Conv2dPixelRef(input, kernel, width, height, ksize, xb, y);
    }
  }

  for (usize y = (height > radius ? height - radius : 0); y < height; ++y) {
    for (usize x = 0; x < width; ++x) {
      output[y * width + x] = Conv2dPixelRef(input, kernel, width, height, ksize, x, y);
    }
  }
}

void spmv_csr_cpu_opt(const usize* row_ptr, const usize* col_idx, const f32* values, const f32* x,
                      f32* y, usize rows) {
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

}  // namespace bench
