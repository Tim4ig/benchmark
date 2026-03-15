#!/usr/bin/env python3
"""
Validate analytical benchmark metrics and result consistency.

Usage:
    python3 scripts/validate_results.py results/latest.csv
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict

F32 = 4
U32 = 4

ALGO_ORDER = [
    "vecadd",
    "reduce",
    "prefix",
    "hist",
    "conv2d",
    "spmv",
    "matmul",
    "blackscholes",
    "bsort",
    "nbody",
]

CHECKSUM_RTOL = {
    "vecadd": 2e-6,
    "reduce": 1e-5,
    "prefix": 1e-5,
    "hist": 0.0,
    "conv2d": 2e-6,
    "spmv": 2e-6,
    "matmul": 2e-6,
    "blackscholes": 5e-5,
    "bsort": 0.0,
    "nbody": 1e-4,
}


def next_pow2(value: int) -> int:
    out = 1
    while out < value:
        out <<= 1
    return out


def bsort_comparisons(n: int) -> int:
    n2 = next_pow2(n)
    log2n = 0
    value = n2
    while value > 1:
        value >>= 1
        log2n += 1
    return (n2 // 2) * log2n * (log2n + 1) // 2


def spmv_nnz_total(n: int, nnz_per_row: int) -> int:
    nnz_row = min(nnz_per_row, max(1, n))
    return n * nnz_row


def expected_metrics(row: dict[str, str]) -> tuple[int, int]:
    algo = row["algo"]
    n = int(row["n"])
    bins = int(row["bins"])
    nnz_per_row = int(row["nnz_per_row"])
    ksize = int(row["ksize"])

    if algo == "vecadd":
      return 2 * n, 3 * n * F32
    if algo == "reduce":
      return n, n * F32
    if algo == "prefix":
      return n, 2 * n * F32
    if algo == "hist":
      return n, n * U32 + bins * U32
    if algo == "conv2d":
      size = n * n
      return 2 * ksize * ksize * size, (2 * size + ksize * ksize) * F32
    if algo == "spmv":
      nnz_total = spmv_nnz_total(n, nnz_per_row)
      bytes_moved = nnz_total * F32 + nnz_total * U32 + (n + 1) * U32 + n * F32 + n * F32
      return 2 * nnz_total, bytes_moved
    if algo == "matmul":
      size = n * n
      return 2 * size * n, 3 * size * F32
    if algo == "blackscholes":
      return 50 * n, 2 * n * F32
    if algo == "bsort":
      n2 = next_pow2(n)
      return bsort_comparisons(n), 2 * n2 * F32
    if algo == "nbody":
      return 20 * n * max(0, n - 1), 5 * n * F32
    raise ValueError(f"unsupported algo: {algo}")


def approx_equal(lhs: float, rhs: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(lhs - rhs) <= max(abs_tol, rel_tol * max(abs(lhs), abs(rhs)))


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def checksum_status(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []
    by_algo: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["status"]) == 0:
            by_algo[row["algo"]].append(row)

    for algo in ALGO_ORDER:
        group = by_algo.get(algo, [])
        if not group:
            continue
        ref = next((row for row in group if row["backend"] == "cpu_ref"), None)
        if ref is None:
            continue
        ref_value = float(ref["checksum"])
        rel_tol = CHECKSUM_RTOL[algo]
        for row in group:
            backend = row["backend"]
            if backend == "hybrid" and algo not in {"matmul", "nbody"}:
                notes.append(
                    f"skip checksum equivalence for hybrid/{algo}: hybrid still runs split subproblems, not a shared input"
                )
                continue
            value = float(row["checksum"])
            if rel_tol == 0.0:
                if value != ref_value:
                    errors.append(
                        f"checksum mismatch for {algo}/{backend}: expected exact {ref_value:.12g}, got {value:.12g}"
                    )
            elif not approx_equal(value, ref_value, rel_tol, 1e-6):
                errors.append(
                    f"checksum mismatch for {algo}/{backend}: ref={ref_value:.12g}, got={value:.12g}, rtol={rel_tol:g}"
                )

    return errors, notes


def validate(path: str) -> int:
    rows = load_rows(path)
    if not rows:
        print(f"ERROR: no rows found in {path}")
        return 1

    errors: list[str] = []
    notes: list[str] = []
    status_counts = Counter()
    backend_counts = Counter()

    for row in rows:
        backend = row["backend"]
        algo = row["algo"]
        status = int(row["status"])
        repeats = int(row["repeats"])
        total_ms = float(row["total_ms"])
        calc_ms = float(row["calc_ms"])
        gflops = float(row["gflops"])
        gbytes = float(row["gbytes"])
        flops = int(row["flops"])
        bytes_moved = int(row["bytes_moved"])

        status_counts[status] += 1
        backend_counts[backend] += 1

        expected_flops, expected_bytes = expected_metrics(row)
        if flops != expected_flops:
            errors.append(f"flops mismatch for {backend}/{algo}: csv={flops}, expected={expected_flops}")
        if bytes_moved != expected_bytes:
            errors.append(
                f"bytes_moved mismatch for {backend}/{algo}: csv={bytes_moved}, expected={expected_bytes}"
            )

        if repeats > 0:
            expected_calc = total_ms / repeats
            if not approx_equal(calc_ms, expected_calc, 1e-5, 1e-5):
                errors.append(
                    f"calc_ms mismatch for {backend}/{algo}: csv={calc_ms:.9f}, expected={expected_calc:.9f}"
                )

        if calc_ms > 0.0:
            expected_gflops = flops / (calc_ms * 1.0e6)
            expected_gbytes = bytes_moved / (calc_ms * 1.0e6)
            if not approx_equal(gflops, expected_gflops, 5e-5, 1e-6):
                errors.append(
                    f"gflops mismatch for {backend}/{algo}: csv={gflops:.9f}, expected={expected_gflops:.9f}"
                )
            if not approx_equal(gbytes, expected_gbytes, 5e-5, 1e-6):
                errors.append(
                    f"gbytes mismatch for {backend}/{algo}: csv={gbytes:.9f}, expected={expected_gbytes:.9f}"
                )

    checksum_errors, checksum_notes = checksum_status(rows)
    errors.extend(checksum_errors)
    notes.extend(checksum_notes)

    print(f"CSV: {path}")
    print(f"Rows: {len(rows)}")
    print(f"Statuses: {dict(status_counts)}")
    print(f"Backend rows: {dict(sorted(backend_counts.items()))}")
    if notes:
        print("Notes:")
        for note in sorted(set(notes)):
            print(f"  - {note}")

    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Validation passed.")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip())
        return 1
    return validate(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
