#include "algorithms_opt.hpp"

#include <algorithm>
#include <immintrin.h>

namespace bench
{
  void vecadd_cpu_opt(const f32 *x, const f32 *y, f32 *out, f32 a, usize n)
  {
    usize i = 0;
#if defined(BENCH_USE_AVX512)
    const __m512 va = _mm512_set1_ps(a);
    for (; i + 15 < n; i += 16)
    {
      const __m512 vx = _mm512_loadu_ps(x + i);
      const __m512 vy = _mm512_loadu_ps(y + i);
      const __m512 vout = _mm512_fmadd_ps(va, vx, vy);
      _mm512_storeu_ps(out + i, vout);
    }
#elif defined(BENCH_USE_AVX2)
    const __m256 va = _mm256_set1_ps(a);
    for (; i + 7 < n; i += 8)
    {
      const __m256 vx = _mm256_loadu_ps(x + i);
      const __m256 vy = _mm256_loadu_ps(y + i);
      const __m256 vout = _mm256_fmadd_ps(va, vx, vy);
      _mm256_storeu_ps(out + i, vout);
    }
#elif defined(BENCH_USE_SSE)
    const __m128 va = _mm_set1_ps(a);
    for (; i + 3 < n; i += 4)
    {
      const __m128 vx = _mm_loadu_ps(x + i);
      const __m128 vy = _mm_loadu_ps(y + i);
      const __m128 vout = _mm_add_ps(_mm_mul_ps(va, vx), vy);
      _mm_storeu_ps(out + i, vout);
    }
#endif
    for (; i < n; ++i) { out[i] = a * x[i] + y[i]; }
  }

  f32 reduce_sum_cpu_opt(const f32 *x, usize n)
  {
    f32 sum = 0.0f;
    usize i = 0;
#if defined(BENCH_USE_AVX512)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 15 < n; i += 16) { acc = _mm512_add_ps(acc, _mm512_loadu_ps(x + i)); }
    alignas(64) f32 tmp[16];
    _mm512_storeu_ps(tmp, acc);
    for (f32 v: tmp) { sum += v; }
#elif defined(BENCH_USE_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) { acc = _mm256_add_ps(acc, _mm256_loadu_ps(x + i)); }
    alignas(32) f32 tmp[8];
    _mm256_storeu_ps(tmp, acc);
    for (f32 v: tmp) { sum += v; }
#elif defined(BENCH_USE_SSE)
    __m128 acc = _mm_setzero_ps();
    for (; i + 3 < n; i += 4) { acc = _mm_add_ps(acc, _mm_loadu_ps(x + i)); }
    alignas(16) f32 tmp[4];
    _mm_storeu_ps(tmp, acc);
    for (f32 v: tmp) { sum += v; }
#endif
    for (; i < n; ++i) { sum += x[i]; }
    return sum;
  }

#if defined(BENCH_USE_AVX2)
  static inline f32 prefix_block_avx2(const f32 *in, f32 *out, f32 carry)
  {
    __m256 t = _mm256_loadu_ps(in);
    __m256 shift = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(t), 4));
    t = _mm256_add_ps(t, shift);
    shift = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(t), 8));
    t = _mm256_add_ps(t, shift);
    shift = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(t), 12));
    t = _mm256_add_ps(t, shift);
    t = _mm256_add_ps(t, _mm256_set1_ps(carry));

    __m128 low = _mm256_castps256_ps128(t);
    __m128 low_last = _mm_shuffle_ps(low, low, _MM_SHUFFLE(3, 3, 3, 3));
    __m256 offset = _mm256_setzero_ps();
    offset = _mm256_insertf128_ps(offset, low_last, 1);
    t = _mm256_add_ps(t, offset);

    _mm256_storeu_ps(out, t);
    __m128 high = _mm256_extractf128_ps(t, 1);
    __m128 high_last = _mm_shuffle_ps(high, high, _MM_SHUFFLE(3, 3, 3, 3));
    return _mm_cvtss_f32(high_last);
  }
#endif

#if defined(BENCH_USE_SSE)
  static inline f32 prefix_block_sse(const f32 *in, f32 *out, f32 carry)
  {
    __m128 t = _mm_loadu_ps(in);
    __m128 shift = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(t), 4));
    t = _mm_add_ps(t, shift);
    shift = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(t), 8));
    t = _mm_add_ps(t, shift);
    shift = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(t), 12));
    t = _mm_add_ps(t, shift);
    t = _mm_add_ps(t, _mm_set1_ps(carry));
    _mm_storeu_ps(out, t);
    __m128 last = _mm_shuffle_ps(t, t, _MM_SHUFFLE(3, 3, 3, 3));
    return _mm_cvtss_f32(last);
  }
#endif

  void prefix_sum_inclusive_cpu_opt(const f32 *in, f32 *out, usize n)
  {
    f32 carry = 0.0f;
    usize i = 0;
#if defined(BENCH_USE_AVX2) && !defined(BENCH_DISABLE_VECTOR_PREFIX)
    for (; i + 7 < n; i += 8) { carry = prefix_block_avx2(in + i, out + i, carry); }
#elif defined(BENCH_USE_SSE) && !defined(BENCH_DISABLE_VECTOR_PREFIX)
    for (; i + 3 < n; i += 4) { carry = prefix_block_sse(in + i, out + i, carry); }
#endif
    for (; i < n; ++i)
    {
      carry += in[i];
      out[i] = carry;
    }
  }

  void histogram_u32_cpu_opt(const u32 *data, usize n, u32 *bins, usize nbins)
  {
    for (usize i = 0; i < nbins; ++i) { bins[i] = 0; }
    if (nbins == 0) { return; }

    const bool pow2 = (nbins & (nbins - 1)) == 0;
    usize i = 0;

#if defined(BENCH_USE_AVX512)
    if (pow2)
    {
      const __m512i mask = _mm512_set1_epi32(static_cast<u32>(nbins - 1));
      alignas(64) u32 idx[16];
      for (; i + 15 < n; i += 16)
      {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const void *>(data + i));
        v = _mm512_and_si512(v, mask);
        _mm512_storeu_si512(reinterpret_cast<void *>(idx), v);
        for (i32 k = 0; k < 16; ++k) { ++bins[idx[k]]; }
      }
    }
#elif defined(BENCH_USE_AVX2)
    if (pow2)
    {
      const __m256i mask = _mm256_set1_epi32(static_cast<u32>(nbins - 1));
      alignas(32) u32 idx[8];
      for (; i + 7 < n; i += 8)
      {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i));
        v = _mm256_and_si256(v, mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(idx), v);
        for (i32 k = 0; k < 8; ++k) { ++bins[idx[k]]; }
      }
    }
#elif defined(BENCH_USE_SSE)
    if (pow2)
    {
      const __m128i mask = _mm_set1_epi32(static_cast<u32>(nbins - 1));
      alignas(16) u32 idx[4];
      for (; i + 3 < n; i += 4)
      {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + i));
        v = _mm_and_si128(v, mask);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(idx), v);
        for (i32 k = 0; k < 4; ++k) { ++bins[idx[k]]; }
      }
    }
#endif

    for (; i < n; ++i)
    {
      const usize idx = static_cast<usize>(data[i]) % nbins;
      ++bins[idx];
    }
  }

  static inline f32 conv2d_pixel_ref(const f32 *input, const f32 *kernel, usize width, usize height,
                                     usize ksize, usize x, usize y)
  {
    const usize radius = ksize / 2;
    f32 acc = 0.0f;
    for (usize ky = 0; ky < ksize; ++ky)
    {
      const usize in_y = y + ky;
      if (in_y < radius || in_y >= height + radius) { continue; }
      const usize src_y = in_y - radius;
      for (usize kx = 0; kx < ksize; ++kx)
      {
        const usize in_x = x + kx;
        if (in_x < radius || in_x >= width + radius) { continue; }
        const usize src_x = in_x - radius;
        acc += input[src_y * width + src_x] * kernel[ky * ksize + kx];
      }
    }
    return acc;
  }

  void conv2d_cpu_opt(const f32 *input, const f32 *kernel, f32 *output, usize width, usize height,
                      usize ksize)
  {
    if (width == 0 || height == 0 || ksize == 0) { return; }
    const usize radius = ksize / 2;

    // Top border
    for (usize y = 0; y < std::min(radius, height); ++y)
    {
      for (usize x = 0; x < width; ++x)
      {
        output[y * width + x] = conv2d_pixel_ref(input, kernel, width, height, ksize, x, y);
      }
    }

    // Middle rows
    for (usize y = radius; y + radius < height; ++y)
    {
      // Left border
      for (usize x = 0; x < std::min(radius, width); ++x)
      {
        output[y * width + x] = conv2d_pixel_ref(input, kernel, width, height, ksize, x, y);
      }

      usize x = radius;
      const usize x_end = width - radius;

#if defined(BENCH_USE_AVX512)
      for (; x + 15 < x_end; x += 16)
      {
        __m512 acc = _mm512_setzero_ps();
        for (usize ky = 0; ky < ksize; ++ky)
        {
          const f32 *row = input + (y + ky - radius) * width + (x - radius);
          for (usize kx = 0; kx < ksize; ++kx)
          {
            const __m512 v = _mm512_loadu_ps(row + kx);
            const __m512 k = _mm512_set1_ps(kernel[ky * ksize + kx]);
            acc = _mm512_fmadd_ps(v, k, acc);
          }
        }
        _mm512_storeu_ps(output + y * width + x, acc);
      }
#elif defined(BENCH_USE_AVX2)
      for (; x + 7 < x_end; x += 8)
      {
        __m256 acc = _mm256_setzero_ps();
        for (usize ky = 0; ky < ksize; ++ky)
        {
          const f32 *row = input + (y + ky - radius) * width + (x - radius);
          for (usize kx = 0; kx < ksize; ++kx)
          {
            const __m256 v = _mm256_loadu_ps(row + kx);
            const __m256 k = _mm256_set1_ps(kernel[ky * ksize + kx]);
            acc = _mm256_fmadd_ps(v, k, acc);
          }
        }
        _mm256_storeu_ps(output + y * width + x, acc);
      }
#elif defined(BENCH_USE_SSE)
      for (; x + 3 < x_end; x += 4)
      {
        __m128 acc = _mm_setzero_ps();
        for (usize ky = 0; ky < ksize; ++ky)
        {
          const f32 *row = input + (y + ky - radius) * width + (x - radius);
          for (usize kx = 0; kx < ksize; ++kx)
          {
            const __m128 v = _mm_loadu_ps(row + kx);
            const __m128 k = _mm_set1_ps(kernel[ky * ksize + kx]);
            acc = _mm_add_ps(acc, _mm_mul_ps(v, k));
          }
        }
        _mm_storeu_ps(output + y * width + x, acc);
      }
#endif

      for (; x < x_end; ++x) { output[y * width + x] = conv2d_pixel_ref(input, kernel, width, height, ksize, x, y); }

      // Right border
      for (usize xb = x_end; xb < width; ++xb)
      {
        output[y * width + xb] = conv2d_pixel_ref(input, kernel, width, height, ksize, xb, y);
      }
    }

    // Bottom border
    for (usize y = (height > radius ? height - radius : 0); y < height; ++y)
    {
      for (usize x = 0; x < width; ++x)
      {
        output[y * width + x] = conv2d_pixel_ref(input, kernel, width, height, ksize, x, y);
      }
    }
  }

  void spmv_csr_cpu_opt(const usize *row_ptr, const usize *col_idx, const f32 *values, const f32 *x,
                        f32 *y, usize rows)
  {
    for (usize row = 0; row < rows; ++row)
    {
      const usize start = row_ptr[row];
      const usize end = row_ptr[row + 1];
      f32 sum = 0.0f;
      usize idx = start;

#if defined(BENCH_USE_AVX512)
      __m512 acc = _mm512_setzero_ps();
      alignas(64) i32 indices[16];
      for (; idx + 15 < end; idx += 16)
      {
        for (i32 k = 0; k < 16; ++k) { indices[k] = static_cast<i32>(col_idx[idx + k]); }
        const __m512 v = _mm512_loadu_ps(values + idx);
        const __m512 xv = _mm512_i32gather_ps(_mm512_loadu_si512(indices), x, 4);
        acc = _mm512_fmadd_ps(v, xv, acc);
      }
      alignas(64) f32 tmp[16];
      _mm512_storeu_ps(tmp, acc);
      for (f32 v: tmp) { sum += v; }
#elif defined(BENCH_USE_AVX2)
      __m256 acc = _mm256_setzero_ps();
      for (; idx + 7 < end; idx += 8)
      {
        const __m256 v = _mm256_loadu_ps(values + idx);
        const __m256i idxv = _mm256_setr_epi32(
          static_cast<i32>(col_idx[idx + 0]), static_cast<i32>(col_idx[idx + 1]),
          static_cast<i32>(col_idx[idx + 2]), static_cast<i32>(col_idx[idx + 3]),
          static_cast<i32>(col_idx[idx + 4]), static_cast<i32>(col_idx[idx + 5]),
          static_cast<i32>(col_idx[idx + 6]), static_cast<i32>(col_idx[idx + 7]));
        const __m256 xv = _mm256_i32gather_ps(x, idxv, 4);
        acc = _mm256_fmadd_ps(v, xv, acc);
      }
      alignas(32) f32 tmp[8];
      _mm256_storeu_ps(tmp, acc);
      for (f32 v: tmp) { sum += v; }
#endif

      for (; idx < end; ++idx) { sum += values[idx] * x[col_idx[idx]]; }
      y[row] = sum;
    }
  }

  void matmul_cpu_opt(const f32 *a, const f32 *b, f32 *c, usize n)
  {
    const usize nn = n * n;
    for (usize idx = 0; idx < nn; ++idx) { c[idx] = 0.0f; }

    for (usize i = 0; i < n; ++i)
    {
      const usize row = i * n;
      for (usize k = 0; k < n; ++k)
      {
        const f32 aik = a[row + k];
        const f32 *brow = b + k * n;
        f32 *crow = c + row;
        usize j = 0;

#if defined(BENCH_USE_AVX512)
        const __m512 va = _mm512_set1_ps(aik);
        for (; j + 15 < n; j += 16)
        {
          __m512 vb = _mm512_loadu_ps(brow + j);
          __m512 vc = _mm512_loadu_ps(crow + j);
          vc = _mm512_fmadd_ps(va, vb, vc);
          _mm512_storeu_ps(crow + j, vc);
        }
#elif defined(BENCH_USE_AVX2)
        const __m256 va = _mm256_set1_ps(aik);
        for (; j + 7 < n; j += 8)
        {
          __m256 vb = _mm256_loadu_ps(brow + j);
          __m256 vc = _mm256_loadu_ps(crow + j);
          vc = _mm256_fmadd_ps(va, vb, vc);
          _mm256_storeu_ps(crow + j, vc);
        }
#elif defined(BENCH_USE_SSE)
        const __m128 va = _mm_set1_ps(aik);
        for (; j + 3 < n; j += 4)
        {
          __m128 vb = _mm_loadu_ps(brow + j);
          __m128 vc = _mm_loadu_ps(crow + j);
          vc = _mm_add_ps(vc, _mm_mul_ps(va, vb));
          _mm_storeu_ps(crow + j, vc);
        }
#endif
        for (; j < n; ++j) { crow[j] += aik * brow[j]; }
      }
    }
  }
} // namespace bench