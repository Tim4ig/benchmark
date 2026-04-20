#!/usr/bin/env python3
"""
Run one backend/algorithm case sequentially with warm-ups and repeated samples.

This script exists for accuracy-focused validation runs where one case should
occupy the machine exclusively instead of sharing a long orchestrated suite.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import sys
import threading
import time
from pathlib import Path


ALGO_ENUM = {
    "vecadd": 0,
    "reduce": 1,
    "prefix": 2,
    "hist": 3,
    "conv2d": 4,
    "spmv": 5,
    "matmul": 6,
    "blackscholes": 7,
    "bsort": 8,
    "nbody": 9,
}

DEFAULTS = {
    "vecadd": {"n": 1 << 20, "bins": 256, "nnz": 16, "ksize": 3},
    "reduce": {"n": 1 << 20, "bins": 256, "nnz": 16, "ksize": 3},
    "prefix": {"n": 1 << 20, "bins": 256, "nnz": 16, "ksize": 3},
    "hist": {"n": 1 << 20, "bins": 256, "nnz": 16, "ksize": 3},
    "conv2d": {"n": 1024, "bins": 256, "nnz": 16, "ksize": 3},
    "spmv": {"n": 1 << 18, "bins": 256, "nnz": 16, "ksize": 3},
    "matmul": {"n": 512, "bins": 256, "nnz": 16, "ksize": 3},
    "blackscholes": {"n": 1 << 20, "bins": 256, "nnz": 16, "ksize": 3},
    "bsort": {"n": 1 << 20, "bins": 256, "nnz": 16, "ksize": 3},
    "nbody": {"n": 4096, "bins": 256, "nnz": 16, "ksize": 3},
}

RAPL_PATH = Path("/sys/class/powercap/intel-rapl:0/energy_uj")


class BenchOptions(ctypes.Structure):
    _fields_ = [
        ("algo", ctypes.c_uint8),
        ("repeats", ctypes.c_uint32),
        ("seed", ctypes.c_uint32),
        ("n", ctypes.c_size_t),
        ("bins", ctypes.c_size_t),
        ("nnz_per_row", ctypes.c_size_t),
        ("ksize", ctypes.c_size_t),
    ]


class BenchResult(ctypes.Structure):
    _fields_ = [
        ("total_time_ms", ctypes.c_double),
        ("calc_time_ms", ctypes.c_double),
        ("mem_time_ms", ctypes.c_double),
        ("gflops", ctypes.c_double),
        ("gbytes", ctypes.c_double),
        ("checksum", ctypes.c_double),
        ("bytes_moved", ctypes.c_uint64),
        ("flops", ctypes.c_uint64),
        ("status", ctypes.c_int32),
    ]


class BenchEntry(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("algo", ctypes.c_uint8),
    ]


BENCH_GET_NAME_FN = ctypes.CFUNCTYPE(ctypes.c_char_p)
BENCH_GET_ENTRIES_FN = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(BenchEntry)))
BENCH_RUN_FN = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(BenchOptions), ctypes.POINTER(BenchResult))


class BenchApi(ctypes.Structure):
    _fields_ = [
        ("get_name", BENCH_GET_NAME_FN),
        ("get_entries", BENCH_GET_ENTRIES_FN),
        ("run", BENCH_RUN_FN),
    ]


BENCH_GET_API_FN = ctypes.CFUNCTYPE(ctypes.POINTER(BenchApi))


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        return None


def read_int(path: Path) -> int | None:
    text = read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def find_amdgpu_hwmon() -> Path | None:
    for path in Path("/sys/class/hwmon").glob("hwmon*"):
        name = read_text(path / "name")
        if name == "amdgpu":
            return path
    return None


class PowerTelemetry:
    def __init__(self) -> None:
        self.gpu_hwmon = find_amdgpu_hwmon()
        self.gpu_power = self.gpu_hwmon / "power1_average" if self.gpu_hwmon else None
        self.gpu_temp = self.gpu_hwmon / "temp1_input" if self.gpu_hwmon else None
        self.gpu_freq_core = self.gpu_hwmon / "freq1_input" if self.gpu_hwmon else None
        self.gpu_freq_mem = self.gpu_hwmon / "freq2_input" if self.gpu_hwmon else None

    def measure(self, work) -> dict[str, float | int]:
        stop = threading.Event()
        gpu_power_samples: list[int] = []
        gpu_temp_samples: list[int] = []
        gpu_core_freq_samples: list[int] = []
        gpu_mem_freq_samples: list[int] = []

        def sampler() -> None:
            while True:
                if self.gpu_power is not None:
                    value = read_int(self.gpu_power)
                    if value is not None:
                        gpu_power_samples.append(value)
                if self.gpu_temp is not None:
                    value = read_int(self.gpu_temp)
                    if value is not None:
                        gpu_temp_samples.append(value)
                if self.gpu_freq_core is not None:
                    value = read_int(self.gpu_freq_core)
                    if value is not None:
                        gpu_core_freq_samples.append(value)
                if self.gpu_freq_mem is not None:
                    value = read_int(self.gpu_freq_mem)
                    if value is not None:
                        gpu_mem_freq_samples.append(value)
                if stop.is_set():
                    break
                time.sleep(0.005)

        thread = threading.Thread(target=sampler, daemon=True)
        rapl_before = read_int(RAPL_PATH)
        wall_t0 = time.perf_counter()
        thread.start()
        work()
        wall_t1 = time.perf_counter()
        rapl_after = read_int(RAPL_PATH)
        stop.set()
        thread.join()

        wall_s = wall_t1 - wall_t0
        watts_cpu = -1.0
        if rapl_before is not None and rapl_after is not None and rapl_after >= rapl_before and wall_s > 0.0:
            watts_cpu = ((rapl_after - rapl_before) * 1.0e-6) / wall_s

        watts_gpu = -1.0
        if gpu_power_samples:
            watts_gpu = sum(gpu_power_samples) / len(gpu_power_samples) * 1.0e-6

        return {
            "wall_ms": wall_s * 1.0e3,
            "watts_cpu": watts_cpu,
            "watts_gpu": watts_gpu,
            "energy_cpu_j": watts_cpu * wall_s if watts_cpu >= 0.0 else -1.0,
            "energy_gpu_j": watts_gpu * wall_s if watts_gpu >= 0.0 else -1.0,
            "gpu_power_samples": len(gpu_power_samples),
            "gpu_temp_avg_c": (sum(gpu_temp_samples) / len(gpu_temp_samples) / 1000.0) if gpu_temp_samples else -1.0,
            "gpu_core_freq_avg_mhz": (sum(gpu_core_freq_samples) / len(gpu_core_freq_samples) / 1.0e6)
            if gpu_core_freq_samples else -1.0,
            "gpu_mem_freq_avg_mhz": (sum(gpu_mem_freq_samples) / len(gpu_mem_freq_samples) / 1.0e6)
            if gpu_mem_freq_samples else -1.0,
        }


def load_api(lib_path: Path) -> tuple[ctypes.CDLL, ctypes.POINTER(BenchApi), str]:
    cdll = ctypes.CDLL(str(lib_path))
    get_api = BENCH_GET_API_FN(("bench_get_api", cdll))
    api = get_api()
    if not api:
        raise RuntimeError(f"bench_get_api failed for {lib_path}")
    name = api.contents.get_name().decode("ascii")
    return cdll, api, name


def resolve_options(args: argparse.Namespace) -> BenchOptions:
    defaults = DEFAULTS[args.algo]
    options = BenchOptions()
    options.algo = ALGO_ENUM[args.algo]
    options.repeats = args.repeats
    options.seed = args.seed
    options.n = args.n if args.n is not None else defaults["n"]
    options.bins = args.bins if args.bins is not None else defaults["bins"]
    options.nnz_per_row = args.nnz if args.nnz is not None else defaults["nnz"]
    ksize = args.ksize if args.ksize is not None else defaults["ksize"]
    options.ksize = ksize + 1 if ksize % 2 == 0 else ksize
    return options


def ensure_algo_supported(api: ctypes.POINTER(BenchApi), algo_name: str) -> None:
    entries_ptr = ctypes.POINTER(BenchEntry)()
    count = api.contents.get_entries(ctypes.byref(entries_ptr))
    supported = {entries_ptr[i].name.decode("ascii") for i in range(count)}
    if algo_name not in supported:
        raise RuntimeError(f"backend does not support algo={algo_name}; supported={sorted(supported)}")


def run_once(api: ctypes.POINTER(BenchApi), options: BenchOptions, telemetry: PowerTelemetry) -> tuple[BenchResult, dict]:
    result = BenchResult()

    def work() -> None:
        rc = api.contents.run(ctypes.byref(options), ctypes.byref(result))
        if rc != 0:
            raise RuntimeError(f"bench api returned rc={rc}")

    power = telemetry.measure(work)
    return result, power


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", required=True, help="Path to one backend shared library")
    parser.add_argument("--algo", required=True, choices=sorted(ALGO_ENUM))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n", type=int)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--nnz", type=int)
    parser.add_argument("--ksize", type=int)
    parser.add_argument("--sleep-ms", type=int, default=250)
    parser.add_argument("--csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lib_path = Path(args.lib).resolve()
    if not lib_path.is_file():
        raise SystemExit(f"missing lib: {lib_path}")

    _, api, backend_name = load_api(lib_path)
    ensure_algo_supported(api, args.algo)
    options = resolve_options(args)
    telemetry = PowerTelemetry()

    rows: list[dict[str, object]] = []

    for warmup in range(args.warmups):
        result, power = run_once(api, options, telemetry)
        rows.append({
            "sample_type": "warmup",
            "sample_idx": warmup,
            "backend": backend_name,
            "algo": args.algo,
            "seed": options.seed,
            "n": options.n,
            "repeats": options.repeats,
            "bins": options.bins,
            "nnz_per_row": options.nnz_per_row,
            "ksize": options.ksize,
            "wall_ms": f"{power['wall_ms']:.6f}",
            "total_ms": f"{result.total_time_ms:.6f}",
            "calc_ms": f"{result.calc_time_ms:.6f}",
            "mem_ms": f"{result.mem_time_ms:.6f}",
            "gflops": f"{result.gflops:.6f}",
            "gbytes": f"{result.gbytes:.6f}",
            "checksum": f"{result.checksum:.12f}",
            "bytes_moved": result.bytes_moved,
            "flops": result.flops,
            "status": result.status,
            "watts_cpu": f"{power['watts_cpu']:.6f}",
            "watts_gpu": f"{power['watts_gpu']:.6f}",
            "energy_cpu_j": f"{power['energy_cpu_j']:.6f}",
            "energy_gpu_j": f"{power['energy_gpu_j']:.6f}",
            "gpu_power_samples": power["gpu_power_samples"],
            "gpu_temp_avg_c": f"{power['gpu_temp_avg_c']:.6f}",
            "gpu_core_freq_avg_mhz": f"{power['gpu_core_freq_avg_mhz']:.6f}",
            "gpu_mem_freq_avg_mhz": f"{power['gpu_mem_freq_avg_mhz']:.6f}",
        })
        time.sleep(args.sleep_ms / 1000.0)

    for idx in range(args.samples):
        result, power = run_once(api, options, telemetry)
        row = {
            "sample_type": "sample",
            "sample_idx": idx,
            "backend": backend_name,
            "algo": args.algo,
            "seed": options.seed,
            "n": options.n,
            "repeats": options.repeats,
            "bins": options.bins,
            "nnz_per_row": options.nnz_per_row,
            "ksize": options.ksize,
            "wall_ms": f"{power['wall_ms']:.6f}",
            "total_ms": f"{result.total_time_ms:.6f}",
            "calc_ms": f"{result.calc_time_ms:.6f}",
            "mem_ms": f"{result.mem_time_ms:.6f}",
            "gflops": f"{result.gflops:.6f}",
            "gbytes": f"{result.gbytes:.6f}",
            "checksum": f"{result.checksum:.12f}",
            "bytes_moved": result.bytes_moved,
            "flops": result.flops,
            "status": result.status,
            "watts_cpu": f"{power['watts_cpu']:.6f}",
            "watts_gpu": f"{power['watts_gpu']:.6f}",
            "energy_cpu_j": f"{power['energy_cpu_j']:.6f}",
            "energy_gpu_j": f"{power['energy_gpu_j']:.6f}",
            "gpu_power_samples": power["gpu_power_samples"],
            "gpu_temp_avg_c": f"{power['gpu_temp_avg_c']:.6f}",
            "gpu_core_freq_avg_mhz": f"{power['gpu_core_freq_avg_mhz']:.6f}",
            "gpu_mem_freq_avg_mhz": f"{power['gpu_mem_freq_avg_mhz']:.6f}",
        }
        rows.append(row)
        print(
            f"{backend_name}/{args.algo} sample={idx} "
            f"wall_ms={row['wall_ms']} total_ms={row['total_ms']} calc_ms={row['calc_ms']} mem_ms={row['mem_ms']} "
            f"watts_cpu={row['watts_cpu']} watts_gpu={row['watts_gpu']} status={row['status']}"
        )
        time.sleep(args.sleep_ms / 1000.0)

    if args.csv:
        write_rows(Path(args.csv), rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
