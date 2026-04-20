#!/usr/bin/env python3
"""
Sequentially run validation/performance campaigns for every backend/algorithm case.

The runner performs:
1. A pilot run with baseline repeats to estimate wall time.
2. A performance campaign with baseline repeats.
3. A power campaign with repeats scaled to reach a target wall-clock window.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import run_single_case as single


BACKENDS = {
    "cpu_ref": "cpu_ref/libcpu_ref.so",
    "cpu_auto": "cpu_auto/libcpu_auto.so",
    "cpu_avx512": "cpu_avx512/libcpu_avx512.so",
    "cpu_mt": "cpu_mt/libcpu_mt.so",
    "vulkan": "vkbench/libvkbench.so",
    "hybrid": "hybrid/libhybrid.so",
}


def supported_algorithms(api) -> list[str]:
    entries_ptr = single.ctypes.POINTER(single.BenchEntry)()
    count = api.contents.get_entries(single.ctypes.byref(entries_ptr))
    return [entries_ptr[i].name.decode("ascii") for i in range(count)]


def run_campaign_case(
    rows: list[dict[str, object]],
    api,
    backend_name: str,
    algo: str,
    repeats: int,
    mode: str,
    warmups: int,
    samples: int,
    sleep_ms: int,
    seed: int,
    pilot_wall_ms: float,
) -> None:
    options = single.resolve_options(
        type(
            "Args",
            (),
            {
                "algo": algo,
                "repeats": repeats,
                "seed": seed,
                "n": None,
                "bins": None,
                "nnz": None,
                "ksize": None,
            },
        )()
    )
    telemetry = single.PowerTelemetry()

    for warmup in range(warmups):
        result, power = single.run_once(api, options, telemetry)
        rows.append(build_row(mode, "warmup", warmup, backend_name, algo, options, result, power, pilot_wall_ms))
        time.sleep(sleep_ms / 1000.0)

    for idx in range(samples):
        result, power = single.run_once(api, options, telemetry)
        rows.append(build_row(mode, "sample", idx, backend_name, algo, options, result, power, pilot_wall_ms))
        print(
            f"mode={mode:5s} backend={backend_name:10s} algo={algo:12s} sample={idx} "
            f"repeats={repeats:<4d} wall_ms={power['wall_ms']:.3f} calc_ms={result.calc_time_ms:.3f} "
            f"mem_ms={result.mem_time_ms:.3f} watts_cpu={power['watts_cpu']:.3f} watts_gpu={power['watts_gpu']:.3f}"
        )
        time.sleep(sleep_ms / 1000.0)


def build_row(
    mode: str,
    sample_type: str,
    sample_idx: int,
    backend_name: str,
    algo: str,
    options,
    result,
    power,
    pilot_wall_ms: float,
) -> dict[str, object]:
    return {
        "mode": mode,
        "sample_type": sample_type,
        "sample_idx": sample_idx,
        "backend": backend_name,
        "algo": algo,
        "seed": options.seed,
        "n": options.n,
        "repeats": options.repeats,
        "bins": options.bins,
        "nnz_per_row": options.nnz_per_row,
        "ksize": options.ksize,
        "pilot_wall_ms": f"{pilot_wall_ms:.6f}",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--baseline-repeats", type=int, default=20)
    parser.add_argument("--perf-samples", type=int, default=7)
    parser.add_argument("--power-samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--sleep-ms", type=int, default=250)
    parser.add_argument("--power-target-ms", type=float, default=750.0)
    parser.add_argument("--backend")
    parser.add_argument("--algo")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dir = Path(args.build_dir).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for backend_key, rel in BACKENDS.items():
        if args.backend and backend_key != args.backend:
            continue
        lib_path = build_dir / rel
        if not lib_path.is_file():
            print(f"skip missing backend {backend_key}: {lib_path}", file=sys.stderr)
            continue

        _, api, backend_name = single.load_api(lib_path)
        algos = supported_algorithms(api)
        if args.algo:
            algos = [algo for algo in algos if algo == args.algo]
        for algo in algos:
            pilot_options = single.resolve_options(
                type(
                    "Args",
                    (),
                    {
                        "algo": algo,
                        "repeats": args.baseline_repeats,
                        "seed": args.seed,
                        "n": None,
                        "bins": None,
                        "nnz": None,
                        "ksize": None,
                    },
                )()
            )
            pilot_result, pilot_power = single.run_once(api, pilot_options, single.PowerTelemetry())
            pilot_wall_ms = float(pilot_power["wall_ms"])
            pilot_calc_ms = float(pilot_result.calc_time_ms)
            pilot_overhead_ms = max(0.0, pilot_wall_ms - args.baseline_repeats * pilot_calc_ms)
            if pilot_calc_ms <= 0.0:
                power_repeats = args.baseline_repeats
            else:
                target_compute_ms = max(0.0, args.power_target_ms - pilot_overhead_ms)
                power_repeats = max(args.baseline_repeats, int(math.ceil(target_compute_ms / pilot_calc_ms)))

            print(
                f"pilot backend={backend_name:10s} algo={algo:12s} "
                f"baseline_repeats={args.baseline_repeats:<4d} pilot_wall_ms={pilot_wall_ms:.3f} "
                f"pilot_calc_ms={pilot_calc_ms:.3f} pilot_overhead_ms={pilot_overhead_ms:.3f} "
                f"power_repeats={power_repeats}"
            )

            rows.append(
                build_row(
                    "pilot",
                    "sample",
                    0,
                    backend_name,
                    algo,
                    pilot_options,
                    pilot_result,
                    pilot_power,
                    pilot_wall_ms,
                )
            )
            time.sleep(args.sleep_ms / 1000.0)

            run_campaign_case(
                rows,
                api,
                backend_name,
                algo,
                args.baseline_repeats,
                "perf",
                args.warmups,
                args.perf_samples,
                args.sleep_ms,
                args.seed,
                pilot_wall_ms,
            )

            run_campaign_case(
                rows,
                api,
                backend_name,
                algo,
                power_repeats,
                "power",
                args.warmups,
                args.power_samples,
                args.sleep_ms,
                args.seed,
                pilot_wall_ms,
            )

    if rows:
        with out_csv.open("w", newline="", encoding="ascii") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
