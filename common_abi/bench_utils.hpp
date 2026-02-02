#pragma once

#include "types.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace bench
{
  struct Options
  {
    std::string algo = "vecadd";
    usize n = 0;
    i32 repeats = 5;
    u32 seed = 1234;
    usize bins = 256;
    usize nnz_per_row = 16;
    usize ksize = 3;
  };

  struct Timing
  {
    f64 total_ms = 0.0;
    f64 avg_ms = 0.0;
  };

  inline std::vector<f32> make_random(usize n, u32 seed, f32 lo = -1.0f, f32 hi = 1.0f)
  {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f32> dist(lo, hi);
    std::vector<f32> data(n);
    for (usize i = 0; i < n; ++i)
    {
      data[i] = dist(rng);
    }
    return data;
  }

  inline std::vector<u32> make_random_u32(usize n, u32 seed, u32 max_value)
  {
    std::mt19937 rng(seed);
    const u32 hi = max_value > 0 ? max_value - 1u : 0u;
    std::uniform_int_distribution<u32> dist(0u, hi);
    std::vector<u32> data(n);
    for (usize i = 0; i < n; ++i)
    {
      data[i] = dist(rng);
    }
    return data;
  }

  struct CSRMatrix
  {
    usize rows = 0;
    usize cols = 0;
    std::vector<usize> row_ptr;
    std::vector<usize> col_idx;
    std::vector<f32> values;
  };

  inline CSRMatrix make_csr_matrix(usize rows, usize cols, usize nnz_per_row, u32 seed)
  {
    CSRMatrix mat;
    mat.rows = rows;
    mat.cols = cols;
    const usize nnz_row = std::min(nnz_per_row, cols == 0 ? usize{1} : cols);
    const usize nnz_total = rows * nnz_row;
    mat.row_ptr.resize(rows + 1);
    mat.col_idx.resize(nnz_total);
    mat.values.resize(nnz_total);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);

    for (usize row = 0; row < rows; ++row)
    {
      const usize base = row * nnz_row;
      mat.row_ptr[row] = base;
      for (usize k = 0; k < nnz_row; ++k)
      {
        const usize idx = base + k;
        const usize col = (row * 1315423911u + k * 2654435761u) % (cols == 0 ? 1 : cols);
        mat.col_idx[idx] = col;
        mat.values[idx] = dist(rng);
      }
    }
    mat.row_ptr[rows] = nnz_total;
    return mat;
  }

  inline std::vector<f32> make_kernel(usize ksize, u32 seed)
  {
    const usize k = ksize == 0 ? 1 : ksize;
    auto kernel = make_random(k * k, seed);
    f32 sum = 0.0f;
    for (f32 v: kernel)
    {
      sum += v;
    }
    if (sum == 0.0f)
    {
      return kernel;
    }
    for (f32 &v: kernel)
    {
      v /= sum;
    }
    return kernel;
  }

  inline Timing measure_ms(i32 repeats, const std::function<void()> &fn)
  {
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    for (i32 i = 0; i < repeats; ++i)
    {
      fn();
    }
    const auto end = clock::now();
    const f64 total_ms =
        std::chrono::duration_cast<std::chrono::duration<f64, std::milli> >(end - start).count();
    Timing t;
    t.total_ms = total_ms;
    t.avg_ms = repeats > 0 ? total_ms / static_cast<f64>(repeats) : 0.0;
    return t;
  }

  inline bool parse_arg_value(i32 argc, char **argv, i32 &i, const std::string &name,
                              std::string &out)
  {
    const std::string arg(argv[i]);
    const std::string prefix = name + "=";
    if (arg.rfind(prefix, 0) == 0)
    {
      out = arg.substr(prefix.size());
      return true;
    }
    if (arg == name && i + 1 < argc)
    {
      out = argv[++i];
      return true;
    }
    return false;
  }

  inline bool parse_arg_value(i32 argc, char **argv, i32 &i, const std::string &name, usize &out)
  {
    std::string value;
    if (!parse_arg_value(argc, argv, i, name, value))
    {
      return false;
    }
    out = static_cast<usize>(std::stoull(value));
    return true;
  }

  inline bool parse_arg_value(i32 argc, char **argv, i32 &i, const std::string &name, i32 &out)
  {
    std::string value;
    if (!parse_arg_value(argc, argv, i, name, value))
    {
      return false;
    }
    out = std::stoi(value);
    return true;
  }

  inline bool parse_arg_value(i32 argc, char **argv, i32 &i, const std::string &name, u32 &out)
  {
    std::string value;
    if (!parse_arg_value(argc, argv, i, name, value))
    {
      return false;
    }
    out = static_cast<u32>(std::stoul(value));
    return true;
  }

  inline void print_help()
  {
    std::printf("Usage: bench_cpu_ref --algo <name> [--n N] [--repeats R] [--seed S]\n");
    std::printf("                      [--bins B] [--nnz NNZ] [--ksize K]\n");
    std::printf("Algorithms: vecadd, reduce, prefix, hist, conv2d, spmv, matmul\n");
    std::printf("Defaults: vecadd n=1048576 repeats=5 seed=1234\n");
    std::printf("          reduce/prefix/hist n=1048576\n");
    std::printf("          conv2d n=1024 ksize=3\n");
    std::printf("          spmv n=262144 nnz=16\n");
    std::printf("          matmul n=512\n");
  }

  inline f64 checksum(const f32 *data, usize n)
  {
    f64 sum = 0.0;
    for (usize i = 0; i < n; ++i)
    {
      sum += static_cast<f64>(data[i]);
    }
    return sum;
  }

  inline f64 checksum_u32(const u32 *data, usize n)
  {
    f64 sum = 0.0;
    for (usize i = 0; i < n; ++i)
    {
      sum += static_cast<f64>(data[i]);
    }
    return sum;
  }
} // namespace bench
