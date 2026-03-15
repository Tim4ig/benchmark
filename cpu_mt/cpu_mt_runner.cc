// Multi-threaded AVX-512 CPU backend.
// All hardware threads used; work split evenly across threads.
// Reuses single-thread AVX-512 kernels from cpu_avx512/algorithms_opt.hpp
// wherever slicing is trivial; writes parallel versions for the rest.

#include "cpu_mt_runner.h"

#include "../bench_settings.h"
#include "../common_abi/bench_utils.hpp"
#include "../cpu_avx512/algorithms_opt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <mutex>
#include <thread>
#include <vector>

namespace bench::cpu_mt {
// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const usize thread_count = std::max(1u, std::thread::hardware_concurrency());

// Run fn(start, end) in parallel across up to thread_count threads, splitting [0..n).
template <typename Fn> static void pfor(usize n, Fn fn) {
  const usize nt = std::min(n, thread_count);
  const usize chunk = (n + nt - 1) / nt;
  std::vector<std::thread> ts;
  ts.reserve(nt);
  for (usize t = 0; t < nt; ++t) {
    const usize s = t * chunk;
    const usize e = std::min(s + chunk, n);
    if (s < e) {
      ts.emplace_back(fn, s, e);
    }
  }
  for (auto& t : ts) {
    t.join();
  }
}

struct BenchmarkSpec {
  u64 flops = 0;
  u64 bytes_moved = 0;
};

template <typename Fn> static BenchResult run_bench(const BenchOptions& opts, Fn fn, const BenchmarkSpec& spec) {
  const auto timing = bench::measure_ms(static_cast<i32>(opts.repeats), fn);
  BenchResult r{};
  r.status = 0;
  r.total_time_ms = timing.total_ms;
  r.calc_time_ms = timing.avg_ms;
  r.flops = spec.flops;
  r.bytes_moved = spec.bytes_moved;
  r.gflops = spec.flops > 0 ? static_cast<f64>(spec.flops) / (timing.avg_ms * 1.0e6) : 0.0;
  r.gbytes = spec.bytes_moved > 0 ? static_cast<f64>(spec.bytes_moved) / (timing.avg_ms * 1.0e6) : 0.0;
  return r;
}

// ---------------------------------------------------------------------------
// VecAdd - trivially parallel slices
// ---------------------------------------------------------------------------
static BenchResult run_vecadd(const BenchOptions& opts) {
  const usize n = opts.n;
  const f32 alpha = 0.7f;
  auto x = bench::make_random(n, opts.seed);
  auto y = bench::make_random(n, opts.seed + 1u);
  std::vector<f32> out(n, 0.0f);

  auto fn = [&]() {
    pfor(n, [&](usize s, usize e) {
      bench::vecadd_cpu_opt(x.data() + s, y.data() + s, out.data() + s, alpha, e - s);
    });
  };
  auto r = run_bench(opts, fn, {static_cast<u64>(2 * n), static_cast<u64>(3 * n * sizeof(f32))});
  r.checksum = bench::checksum(out.data(), n);
  return r;
}

// ---------------------------------------------------------------------------
// Reduce - parallel partial sums, then sum the partials
// ---------------------------------------------------------------------------
static BenchResult run_reduce(const BenchOptions& opts) {
  const usize n = opts.n;
  auto data = bench::make_random(n, opts.seed);
  std::vector<f64> partials(thread_count, 0.0);
  f32 total = 0.0f;

  auto fn = [&]() {
    const usize nt = std::min(n, thread_count);
    const usize chunk = (n + nt - 1) / nt;
    std::vector<std::thread> ts;
    ts.reserve(nt);
    for (usize t = 0; t < nt; ++t) {
      const usize s = t * chunk;
      const usize e = std::min(s + chunk, n);
      if (s < e) {
        ts.emplace_back([&, t, s, e]() {
          partials[t] = bench::reduce_sum_cpu_opt(data.data() + s, e - s);
        });
      }
    }
    for (auto& t : ts) {
      t.join();
    }
    total = 0.0f;
    for (usize t = 0; t < nt; ++t) {
      total += static_cast<f32>(partials[t]);
    }
  };
  auto r = run_bench(opts, fn, {static_cast<u64>(n), static_cast<u64>(n * sizeof(f32))});
  r.checksum = total;
  return r;
}

// ---------------------------------------------------------------------------
// Prefix sum - 3-pass parallel inclusive scan
//   Pass 1: each thread scans its chunk (local prefix)
//   Pass 2: sequential prefix on per-chunk end values (compute offsets)
//   Pass 3: each thread (except first) adds its carry-in to its chunk
// ---------------------------------------------------------------------------
static BenchResult run_prefix(const BenchOptions& opts) {
  const usize n = opts.n;
  auto data = bench::make_random(n, opts.seed);
  std::vector<f32> out(n, 0.0f);

  const usize nt = std::min(n, thread_count);
  const usize chunk = (n + nt - 1) / nt;
  std::vector<f32> carry(nt, 0.0f);

  auto fn = [&]() {
    // Pass 1
    {
      std::vector<std::thread> ts;
      ts.reserve(nt);
      for (usize t = 0; t < nt; ++t) {
        const usize s = t * chunk;
        const usize e = std::min(s + chunk, n);
        if (s < e) {
          ts.emplace_back([&, t, s, e]() {
            bench::prefix_sum_inclusive_cpu_opt(data.data() + s, out.data() + s, e - s);
            carry[t] = out[e - 1];
          });
        }
      }
      for (auto& t : ts) {
        t.join();
      }
    }
    // Pass 2 - sequential prefix on carry[]
    for (usize t = 1; t < nt; ++t) {
      carry[t] += carry[t - 1];
    }
    // Pass 3 - add offset to each chunk (skip first thread, its offset is 0)
    {
      std::vector<std::thread> ts;
      ts.reserve(nt - 1);
      for (usize t = 1; t < nt; ++t) {
        const usize s = t * chunk;
        const usize e = std::min(s + chunk, n);
        const f32 offset = carry[t - 1];
        if (s < e) {
          ts.emplace_back([&, s, e, offset]() {
            for (usize i = s; i < e; ++i) {
              out[i] += offset;
            }
          });
        }
      }
      for (auto& t : ts) {
        t.join();
      }
    }
  };
  auto r = run_bench(opts, fn, {static_cast<u64>(n), static_cast<u64>(2 * n * sizeof(f32))});
  r.checksum = bench::checksum(out.data(), n);
  return r;
}

// ---------------------------------------------------------------------------
// Histogram - per-thread private bins, merge afterwards
// ---------------------------------------------------------------------------
static BenchResult run_hist(const BenchOptions& opts) {
  const usize n = opts.n;
  const usize bins = opts.bins;
  auto data = bench::make_random_u32(n, opts.seed, static_cast<u32>(bins));
  std::vector<u32> hist_out(bins, 0u);

  const usize nt = std::min(n, thread_count);
  std::vector<std::vector<u32>> local(nt, std::vector<u32>(bins, 0u));

  auto fn = [&]() {
    for (auto& lb : local) {
      std::fill(lb.begin(), lb.end(), 0u);
    }
    pfor(n, [&](usize s, usize e) {
      const usize t = s / ((n + nt - 1) / nt);
      bench::histogram_u32_cpu_opt(data.data() + s, e - s, local[t].data(), bins);
    });
    std::fill(hist_out.begin(), hist_out.end(), 0u);
    for (usize t = 0; t < nt; ++t) {
      for (usize b = 0; b < bins; ++b) {
        hist_out[b] += local[t][b];
      }
    }
  };
  auto r = run_bench(opts, fn, {static_cast<u64>(n), static_cast<u64>(n * sizeof(u32) + bins * sizeof(u32))});
  r.checksum = bench::checksum_u32(hist_out.data(), bins);
  return r;
}

// ---------------------------------------------------------------------------
// Conv2d - parallel over output rows
// Each thread handles a contiguous y-range; reads whole input freely.
// ---------------------------------------------------------------------------
namespace {
inline f32 conv2d_pixel_ref(const f32* inp, const f32* ker, usize w, usize h, usize ks, usize px, usize py) {
  const usize radius = ks / 2;
  f32 acc = 0.0f;
  for (usize ky = 0; ky < ks; ++ky) {
    const usize iy = py + ky;
    if (iy < radius || iy >= h + radius) {
      continue;
    }
    const usize sy = iy - radius;
    for (usize kx = 0; kx < ks; ++kx) {
      const usize ix = px + kx;
      if (ix < radius || ix >= w + radius) {
        continue;
      }
      acc += inp[sy * w + (ix - radius)] * ker[ky * ks + kx];
    }
  }
  return acc;
}

void conv2d_rows_range(const f32* inp, const f32* ker, f32* out, usize w, usize h, usize ks, usize y0, usize y1) {
  const usize radius = ks / 2;
  for (usize y = y0; y < y1; ++y) {
    if (y < radius || y + radius >= h) {
      for (usize x = 0; x < w; ++x) {
        out[y * w + x] = conv2d_pixel_ref(inp, ker, w, h, ks, x, y);
      }
      continue;
    }
    // Border columns - scalar
    for (usize x = 0; x < std::min(radius, w); ++x) {
      out[y * w + x] = conv2d_pixel_ref(inp, ker, w, h, ks, x, y);
    }
    // Inner columns - AVX-512
    usize x = radius;
    const usize xe = w > radius ? w - radius : 0;
    for (; x + 15 < xe; x += 16) {
      __m512 acc = _mm512_setzero_ps();
      for (usize ky = 0; ky < ks; ++ky) {
        const f32* row = inp + (y + ky - radius) * w + (x - radius);
        for (usize kx = 0; kx < ks; ++kx) {
          acc = _mm512_fmadd_ps(_mm512_loadu_ps(row + kx), _mm512_set1_ps(ker[ky * ks + kx]), acc);
        }
      }
      _mm512_storeu_ps(out + y * w + x, acc);
    }
    for (; x < xe; ++x) {
      out[y * w + x] = conv2d_pixel_ref(inp, ker, w, h, ks, x, y);
    }
    for (usize xb = xe; xb < w; ++xb) {
      out[y * w + xb] = conv2d_pixel_ref(inp, ker, w, h, ks, xb, y);
    }
  }
}
} // namespace

static BenchResult run_conv2d(const BenchOptions& opts) {
  const usize n = opts.n;
  const usize ks = opts.ksize;
  const usize size = n * n;
  auto inp = bench::make_random(size, opts.seed);
  auto ker = bench::make_kernel(ks, opts.seed + 1u);
  std::vector<f32> out(size, 0.0f);

  const f64 flops = 2.0 * static_cast<f64>(ks) * static_cast<f64>(ks) * static_cast<f64>(size);
  auto fn = [&]() {
    pfor(n, [&](usize y0, usize y1) {
      conv2d_rows_range(inp.data(), ker.data(), out.data(), n, n, ks, y0, y1);
    });
  };
  auto r = run_bench(opts, fn, {static_cast<u64>(flops), static_cast<u64>((2 * size + ks * ks) * sizeof(f32))});
  r.checksum = bench::checksum(out.data(), size);
  return r;
}

// ---------------------------------------------------------------------------
// SpMV - parallel over rows
// row_ptr values are absolute NNZ indices so passing row_ptr+rs works directly.
// ---------------------------------------------------------------------------
static BenchResult run_spmv(const BenchOptions& opts) {
  const usize n = opts.n;
  const auto mat = bench::make_csr_matrix(n, n, opts.nnz_per_row, opts.seed);
  auto x = bench::make_random(n, opts.seed + 1u);
  std::vector<f32> y(n, 0.0f);

  const f64 flops = 2.0 * static_cast<f64>(mat.values.size());
  const usize bm = mat.values.size() * sizeof(f32) + mat.col_idx.size() * sizeof(u32) +
                   mat.row_ptr.size() * sizeof(u32) + x.size() * sizeof(f32) + y.size() * sizeof(f32);
  auto fn = [&]() {
    pfor(n, [&](usize rs, usize re) {
      bench::spmv_csr_cpu_opt(mat.row_ptr.data() + rs,
                              mat.col_idx.data(),
                              mat.values.data(),
                              x.data(),
                              y.data() + rs,
                              re - rs);
    });
  };
  auto r = run_bench(opts, fn, {static_cast<u64>(flops), static_cast<u64>(bm)});
  r.checksum = bench::checksum(y.data(), n);
  return r;
}

// ---------------------------------------------------------------------------
// Matmul - parallel over output rows (i-loop)
// Each thread writes C[is*n .. ie*n), reads full A and B.
// ---------------------------------------------------------------------------
static BenchResult run_matmul(const BenchOptions& opts) {
  const usize n = opts.n;
  const usize nn = n * n;
  auto a = bench::make_random(nn, opts.seed);
  auto b = bench::make_random(nn, opts.seed + 1u);
  std::vector<f32> c(nn, 0.0f);

  const f64 flops = 2.0 * static_cast<f64>(n) * static_cast<f64>(n) * static_cast<f64>(n);
  auto fn = [&]() {
    std::fill(c.begin(), c.end(), 0.0f);
    pfor(n, [&](usize is, usize ie) {
      for (usize i = is; i < ie; ++i) {
        const usize row = i * n;
        for (usize k = 0; k < n; ++k) {
          const f32 aik = a[row + k];
          const f32* brow = b.data() + k * n;
          f32* crow = c.data() + row;
          usize j = 0;
          const __m512 va = _mm512_set1_ps(aik);
          for (; j + 15 < n; j += 16) {
            _mm512_storeu_ps(crow + j, _mm512_fmadd_ps(va, _mm512_loadu_ps(brow + j), _mm512_loadu_ps(crow + j)));
          }
          for (; j < n; ++j) {
            crow[j] += aik * brow[j];
          }
        }
      }
    });
  };
  auto r = run_bench(opts, fn, {static_cast<u64>(flops), static_cast<u64>(3 * nn * sizeof(f32))});
  r.checksum = bench::checksum(c.data(), nn);
  return r;
}

// ---------------------------------------------------------------------------
// Black-Scholes - trivially parallel slices
// ---------------------------------------------------------------------------
static BenchResult run_blackscholes(const BenchOptions& opts) {
  const usize n = opts.n;
  auto s = bench::make_random(n, opts.seed, 80.0f, 120.0f);
  std::vector<f32> out(n, 0.0f);

  auto fn = [&]() {
    pfor(n, [&](usize s0, usize s1) {
      bench::blackscholes_cpu_opt(s.data() + s0, out.data() + s0, s1 - s0);
    });
  };
  auto r = run_bench(opts, fn, {static_cast<u64>(50 * n), static_cast<u64>(2 * n * sizeof(f32))});
  r.checksum = bench::checksum(out.data(), n);
  return r;
}

// ---------------------------------------------------------------------------
// Bitonic sort - parallel compare-and-swap per (k,j) stage
// Each (i, l=i^j) pair with l>i is independent, so it is safe to split the i-loop.
// ---------------------------------------------------------------------------
static BenchResult run_bsort(const BenchOptions& opts) {
  usize n2 = 1;
  while (n2 < opts.n) {
    n2 <<= 1;
  }
  std::vector<f32> data = bench::make_random(n2, opts.seed);
  const std::vector<f32> orig = data;

  usize log2n = 0;
  for (usize t = n2; t > 1; t >>= 1) {
    ++log2n;
  }
  const u64 comparisons = (static_cast<u64>(n2) >> 1) * log2n * (log2n + 1) / 2;

  auto fn = [&]() {
    data = orig;
    for (usize k = 2; k <= n2; k <<= 1) {
      for (usize j = k >> 1; j > 0; j >>= 1) {
        pfor(n2, [&](usize s, usize e) {
          for (usize i = s; i < e; ++i) {
            const usize l = i ^ j;
            if (l > i) {
              const bool asc = (i & k) == 0;
              if (asc ? (data[i] > data[l]) : (data[i] < data[l])) {
                std::swap(data[i], data[l]);
              }
            }
          }
        });
      }
    }
  };
  auto r = run_bench(opts, fn, {comparisons, static_cast<u64>(2 * n2 * sizeof(f32))});
  r.checksum = bench::checksum(data.data(), n2);
  return r;
}

// ---------------------------------------------------------------------------
// N-Body - parallel over particles (outer i-loop)
// Each thread writes fx/fy for its particles; reads all positions (shared, read-only).
// ---------------------------------------------------------------------------
static void nbody_range(const f32* px, const f32* py, const f32* mass, f32* fx, f32* fy, usize n, usize i0, usize i1) {
  const __m512 veps2 = _mm512_set1_ps(1e-4f);
  for (usize i = i0; i < i1; ++i) {
    const __m512 vxi = _mm512_set1_ps(px[i]);
    const __m512 vyi = _mm512_set1_ps(py[i]);
    __m512 vax = _mm512_setzero_ps();
    __m512 vay = _mm512_setzero_ps();
    usize j = 0;
    for (; j + 15 < n; j += 16) {
      const __m512 vmj_raw = _mm512_loadu_ps(mass + j);
      const __mmask16 self_mask = (i >= j && i < j + 16) ? static_cast<__mmask16>(~(1u << (i - j))) : 0xFFFFu;
      const __m512 vmj = _mm512_maskz_mov_ps(self_mask, vmj_raw);
      const __m512 dx = _mm512_sub_ps(_mm512_loadu_ps(px + j), vxi);
      const __m512 dy = _mm512_sub_ps(_mm512_loadu_ps(py + j), vyi);
      __m512 d2 = _mm512_fmadd_ps(dx, dx, _mm512_fmadd_ps(dy, dy, veps2));
      __m512 inv = _mm512_rsqrt14_ps(d2);
      const __m512 half = _mm512_set1_ps(0.5f);
      const __m512 c15 = _mm512_set1_ps(1.5f);
      inv = _mm512_mul_ps(inv, _mm512_fnmadd_ps(_mm512_mul_ps(d2, inv), _mm512_mul_ps(inv, half), c15));
      const __m512 inv3 = _mm512_mul_ps(inv, _mm512_mul_ps(inv, inv));
      const __m512 f = _mm512_mul_ps(vmj, inv3);
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
      const f32 d2 = dx * dx + dy * dy + 1e-4f;
      const f32 inv = 1.0f / std::sqrt(d2);
      const f32 f = mass[j] * inv * inv * inv;
      ax += f * dx;
      ay += f * dy;
    }
    fx[i] = ax;
    fy[i] = ay;
  }
}

static BenchResult run_nbody(const BenchOptions& opts) {
  const usize n = opts.n;
  auto px = bench::make_random(n, opts.seed, -100.0f, 100.0f);
  auto py = bench::make_random(n, opts.seed + 1u, -100.0f, 100.0f);
  auto mass = bench::make_random(n, opts.seed + 2u, 0.1f, 2.0f);
  std::vector<f32> fx(n, 0.0f), fy(n, 0.0f);

  const f64 flops = 20.0 * static_cast<f64>(n) * static_cast<f64>(n - 1);
  auto fn = [&]() {
    pfor(n, [&](usize i0, usize i1) {
      nbody_range(px.data(), py.data(), mass.data(), fx.data(), fy.data(), n, i0, i1);
    });
  };
  auto r = run_bench(opts, fn, {static_cast<u64>(flops), static_cast<u64>(5 * n * sizeof(f32))});
  r.checksum = bench::checksum(fx.data(), n) + bench::checksum(fy.data(), n);
  return r;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
BenchResult cpu_mt_runner_run(const BenchOptions& options) {
  const BenchOptions opts = bench::resolve_options(options);
  switch (opts.algo) {
    case BenchAlgo::kBenchAlgoVecadd:
      return run_vecadd(opts);
    case BenchAlgo::kBenchAlgoReduce:
      return run_reduce(opts);
    case BenchAlgo::kBenchAlgoPrefix:
      return run_prefix(opts);
    case BenchAlgo::kBenchAlgoHist:
      return run_hist(opts);
    case BenchAlgo::kBenchAlgoConV2D:
      return run_conv2d(opts);
    case BenchAlgo::kBenchAlgoSpmv:
      return run_spmv(opts);
    case BenchAlgo::kBenchAlgoMatmul:
      return run_matmul(opts);
    case BenchAlgo::kBenchAlgoBlackscholes:
      return run_blackscholes(opts);
    case BenchAlgo::kBenchAlgoBsort:
      return run_bsort(opts);
    case BenchAlgo::kBenchAlgoNbody:
      return run_nbody(opts);
    default: {
      BenchResult r{};
      r.status = -2;
      return r;
    }
  }
}
} // namespace bench::cpu_mt
