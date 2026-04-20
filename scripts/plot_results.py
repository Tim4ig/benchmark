#!/usr/bin/env python3
"""
Plot benchmark results from the orchestrator CSV output.

Usage:
    python3 plot_results.py results.csv [output_dir]

Generates:
    gflops_by_algo.png      - Bar chart: GFLOPS per algorithm per backend
    gbytes_by_algo.png      - Bar chart: GB/s per algorithm per backend
    speedup_vs_ref.png      - Speedup relative to cpu_ref
    time_heatmap.png        - Heatmap: calc_ms per (backend, algo)
    radar_gflops.png        - Normalized radar chart across algorithms
"""

import collections
import csv
import math
import os
import sys

# Try to import matplotlib; give friendly error if missing
try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required. Install with:")
    print("  pip install matplotlib numpy")
    sys.exit(1)

# Helpers

ALGO_ORDER = [
    "vecadd", "reduce", "prefix", "hist", "conv2d", "spmv", "matmul",
    "blackscholes", "bsort", "nbody",
]

BACKEND_COLORS = {
    "cpu_ref": "#d62728",  # red
    "cpu_auto": "#ff7f0e",  # orange
    "cpu_avx512": "#2ca02c",  # green
    "cpu_mt": "#17a589",  # teal
    "vulkan": "#1f77b4",  # blue
    "hybrid": "#9467bd",  # purple
}

BACKEND_ORDER = ["cpu_ref", "cpu_auto", "cpu_avx512", "cpu_mt", "vulkan", "hybrid"]


def load_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(line for line in f if not line.startswith('#'))
        for row in reader:
            try:
                rows.append({
                    'backend': row['backend'],
                    'algo': row['algo'],
                    'n': int(row['n']),
                    'repeats': int(row['repeats']),
                    'total_ms': float(row['total_ms']),
                    'calc_ms': float(row['calc_ms']),
                    'gflops': float(row['gflops']),
                    'gbytes': float(row['gbytes']),
                    'checksum': float(row['checksum']),
                    'watts_cpu': float(row.get('watts_cpu', -1)),
                    'watts_gpu': float(row.get('watts_gpu', -1)),
                    'status': int(row['status']),
                })
            except (ValueError, KeyError):
                pass  # skip malformed rows
    return rows


def pivot(rows, metric):
    """Returns dict: {algo: {backend: value}}"""
    d = collections.defaultdict(dict)
    for r in rows:
        if r['status'] == 0:
            d[r['algo']][r['backend']] = r[metric]
    return d


def backend_color(b):
    return BACKEND_COLORS.get(b, '#7f7f7f')


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


# Grouped bar chart

def grouped_bar(pivot_data, metric_label, title, out_path, log_scale=False):
    algos = [a for a in ALGO_ORDER if a in pivot_data]
    if not algos:
        return

    all_backends = set()
    for a in algos:
        all_backends |= set(pivot_data[a].keys())
    backends = [b for b in BACKEND_ORDER if b in all_backends]
    if not backends:
        return

    x = np.arange(len(algos))
    width = 0.8 / len(backends)

    fig, ax = plt.subplots(figsize=(max(10, len(algos) * 1.5), 6))

    for i, b in enumerate(backends):
        vals = [pivot_data[a].get(b, 0) for a in algos]
        offset = (i - len(backends) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=b, color=backend_color(b), edgecolor='white', linewidth=0.5)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{v:.2g}', ha='center', va='bottom', fontsize=6, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('blackscholes', 'bs').replace('_', '\n') for a in algos],
                       fontsize=9)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    save(fig, out_path)


# Speedup chart

def speedup_chart(pivot_data, ref_backend, out_path):
    algos = [a for a in ALGO_ORDER if a in pivot_data]
    if not algos:
        return

    all_backends = set()
    for a in algos:
        all_backends |= set(pivot_data[a].keys())
    backends = [b for b in BACKEND_ORDER if b in all_backends and b != ref_backend]
    if not backends:
        return

    x = np.arange(len(algos))
    width = 0.8 / len(backends)

    fig, ax = plt.subplots(figsize=(max(10, len(algos) * 1.5), 6))

    for i, b in enumerate(backends):
        speedups = []
        for a in algos:
            ref = pivot_data[a].get(ref_backend, 0)
            val = pivot_data[a].get(b, 0)
            speedups.append(val / ref if ref > 0 and val > 0 else 0)
        offset = (i - len(backends) / 2 + 0.5) * width
        bars = ax.bar(x + offset, speedups, width * 0.9,
                      label=b, color=backend_color(b), edgecolor='white', linewidth=0.5)
        for bar, s in zip(bars, speedups):
            if s > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{s:.1f}x', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1, label=f'{ref_backend} (1x)')
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('blackscholes', 'bs') for a in algos], fontsize=9)
    ax.set_ylabel(f'Speedup vs {ref_backend}', fontsize=11)
    ax.set_title(f'Speedup relative to {ref_backend}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save(fig, out_path)


# Heatmap

def heatmap(rows, metric, title, out_path):
    algos = [a for a in ALGO_ORDER if any(r['algo'] == a for r in rows)]
    backends = [b for b in BACKEND_ORDER if any(r['backend'] == b for r in rows)]
    if not algos or not backends:
        return

    data = np.zeros((len(backends), len(algos)))
    for r in rows:
        if r['status'] == 0 and r['backend'] in backends and r['algo'] in algos:
            bi = backends.index(r['backend'])
            ai = algos.index(r['algo'])
            data[bi, ai] = r[metric]

    # Log-scale normalization for color
    log_data = np.log10(np.where(data > 0, data, np.nan))

    fig, ax = plt.subplots(figsize=(max(10, len(algos) * 1.2), max(4, len(backends) * 0.9)))
    im = ax.imshow(log_data, aspect='auto', cmap='RdYlGn')

    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([a.replace('blackscholes', 'bs') for a in algos], fontsize=9)
    ax.set_yticks(range(len(backends)))
    ax.set_yticklabels(backends, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')

    for bi in range(len(backends)):
        for ai in range(len(algos)):
            v = data[bi, ai]
            if v > 0:
                txt = f'{v:.2f}' if v < 1000 else f'{v:.0f}'
                ax.text(ai, bi, txt, ha='center', va='center', fontsize=8,
                        color='black' if not math.isnan(log_data[bi, ai]) else 'grey')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'log10({metric})', fontsize=9)
    fig.tight_layout()
    save(fig, out_path)


# Radar chart

def radar_chart(pivot_data, metric_label, out_path):
    """Normalized radar chart: each algo on an axis, each backend a line."""
    algos = [a for a in ALGO_ORDER if a in pivot_data]
    if len(algos) < 3:
        return

    all_backends = set()
    for a in algos:
        all_backends |= set(pivot_data[a].keys())
    backends = [b for b in BACKEND_ORDER if b in all_backends]

    # Normalize each algo column to [0,1]
    max_per_algo = {}
    for a in algos:
        vals = [pivot_data[a].get(b, 0) for b in backends]
        max_per_algo[a] = max(vals) if max(vals) > 0 else 1

    N = len(algos)
    angles = [2 * math.pi * i / N for i in range(N)] + [0]  # close the circle
    algo_labels = [a.replace('blackscholes', 'bs') for a in algos]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for b in backends:
        vals = [pivot_data[a].get(b, 0) / max_per_algo[a] for a in algos]
        vals += [vals[0]]  # close
        ax.plot(angles, vals, label=b, color=backend_color(b), linewidth=2)
        ax.fill(angles, vals, color=backend_color(b), alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(algo_labels, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title(f'Normalized {metric_label} (radar)', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    save(fig, out_path)


# Power charts

def power_heatmap(rows, out_dir):
    """Two heatmaps side-by-side: CPU package watts and total (CPU+GPU) watts.
    Falls back to GPU-only display when RAPL (watts_cpu) is unavailable."""
    algos    = [a for a in ALGO_ORDER if any(r['algo'] == a and r['status'] == 0 for r in rows)]
    backends = [b for b in BACKEND_ORDER if any(r['backend'] == b and r['status'] == 0 for r in rows)]
    if not algos or not backends:
        return

    # GPU idle estimate (median over CPU-only backends)
    gpu_only = [b for b in backends if b not in ('vulkan', 'hybrid')]
    idle_samples = [r['watts_gpu'] for r in rows
                    if r['status'] == 0 and r['backend'] in gpu_only and r['watts_gpu'] >= 0]
    gpu_idle = float(np.median(idle_samples)) if idle_samples else 0.0

    rapl_available = any(r['watts_cpu'] >= 0 for r in rows if r['status'] == 0)

    # Build matrices — when RAPL unavailable, cpu_mat stays NaN, total_mat uses GPU only
    cpu_mat   = np.full((len(backends), len(algos)), np.nan)
    total_mat = np.full((len(backends), len(algos)), np.nan)
    for r in rows:
        if r['status'] != 0:
            continue
        bi = backends.index(r['backend'])
        ai = algos.index(r['algo'])
        gw = r['watts_gpu'] if r['watts_gpu'] >= 0 else gpu_idle
        if rapl_available and r['watts_cpu'] >= 0:
            cw = r['watts_cpu']
            cpu_mat[bi, ai]   = cw
            total_mat[bi, ai] = cw + gw
        else:
            total_mat[bi, ai] = gw

    algo_labels = [a.replace('blackscholes', 'bs') for a in algos]
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(max(14, len(algos) * 1.3) * 2, max(4, len(backends) * 0.85)),
                                   gridspec_kw={'wspace': 0.35})

    cpu_title = 'CPU Package Power (W)' if rapl_available else 'CPU Package Power (W)\n[RAPL unavailable — no data]'
    total_label = 'Total System Power: CPU + GPU (W)' if rapl_available else 'GPU Power (W)\n[CPU RAPL unavailable]'
    for ax, mat, title, unit in [
        (ax1, cpu_mat,   cpu_title, 'W'),
        (ax2, total_mat, f'{total_label}\n(GPU idle ≈ {gpu_idle:.0f} W for CPU-only backends)', 'W'),
    ]:
        im = ax.imshow(mat, aspect='auto', cmap='YlOrRd',
                       vmin=np.nanmin(mat) * 0.95, vmax=np.nanmax(mat) * 1.02)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algo_labels, fontsize=9)
        ax.set_yticks(range(len(backends)))
        ax.set_yticklabels(backends, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        for bi in range(len(backends)):
            for ai in range(len(algos)):
                v = mat[bi, ai]
                if not np.isnan(v):
                    ax.text(ai, bi, f'{v:.0f}', ha='center', va='center',
                            fontsize=8.5, fontweight='bold',
                            color='white' if v > np.nanmax(mat) * 0.7 else 'black')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(unit, fontsize=9)

    fig.tight_layout()
    save(fig, os.path.join(out_dir, 'power_heatmap.png'))


def power_chart(rows, out_path):
    """Energy efficiency bar chart: GFLOP/s per total Watt (CPU+GPU) for all backends.
    Uses log scale — range spans ~3 orders of magnitude across algorithms.
    """
    algos    = [a for a in ALGO_ORDER if any(r['algo'] == a and r['status'] == 0 for r in rows)]
    backends = [b for b in BACKEND_ORDER if any(r['backend'] == b and r['status'] == 0 for r in rows)]
    if not algos or not backends:
        return

    gpu_only = [b for b in backends if b not in ('vulkan', 'hybrid')]
    idle_samples = [r['watts_gpu'] for r in rows
                    if r['status'] == 0 and r['backend'] in gpu_only and r['watts_gpu'] >= 0]
    gpu_idle = float(np.median(idle_samples)) if idle_samples else 0.0

    rapl_available = any(r['watts_cpu'] >= 0 for r in rows if r['status'] == 0)

    # GFLOPS / (cpu_w + gpu_w); when RAPL unavailable fall back to GPU watts only
    eff = collections.defaultdict(dict)
    for r in rows:
        if r['status'] != 0:
            continue
        gw = r['watts_gpu'] if r['watts_gpu'] >= 0 else gpu_idle
        if rapl_available and r['watts_cpu'] >= 0:
            tw = r['watts_cpu'] + gw
        else:
            tw = gw
        if tw > 0 and r['gflops'] > 0:
            eff[r['algo']][r['backend']] = r['gflops'] / tw

    if not any(eff.values()):
        return

    algos = [a for a in algos if a in eff]
    x     = np.arange(len(algos))
    width = 0.8 / len(backends)
    algo_labels = [a.replace('blackscholes', 'bs') for a in algos]

    fig, ax = plt.subplots(figsize=(max(12, len(algos) * 1.5), 6))

    for i, b in enumerate(backends):
        vals = [eff[a].get(b, 0) for a in algos]
        offset = (i - len(backends) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=b, color=backend_color(b), edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                lbl = f'{v:.1f}' if v < 10 else f'{v:.0f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.08,
                        lbl, ha='center', va='bottom', fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(algo_labels, fontsize=10)
    power_label = 'total CPU+GPU' if rapl_available else 'GPU only (CPU RAPL unavailable)'
    ax.set_ylabel(f'GFLOP/s per Watt  ({power_label})', fontsize=11)
    ax.set_title(f'Energy Efficiency: GFLOP/s per Watt  ({power_label})\n'
                 f'(GPU idle ≈ {gpu_idle:.0f} W counted for CPU-only backends)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    save(fig, out_path)


# Main

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading: {csv_path}")
    rows = load_csv(csv_path)
    if not rows:
        print("ERROR: No valid rows found in CSV")
        sys.exit(1)

    backends_found = sorted(set(r['backend'] for r in rows))
    algos_found = sorted(set(r['algo'] for r in rows))
    print(f"Backends: {backends_found}")
    print(f"Algorithms: {algos_found}")
    print(f"Output dir: {out_dir}")
    print()

    gf = pivot(rows, 'gflops')
    gb = pivot(rows, 'gbytes')
    tm = pivot(rows, 'calc_ms')

    grouped_bar(gf, 'GFLOP/s', 'Computational Throughput by Backend and Algorithm',
                os.path.join(out_dir, 'gflops_by_algo.png'), log_scale=True)

    grouped_bar(gb, 'GB/s', 'Memory Bandwidth by Backend and Algorithm',
                os.path.join(out_dir, 'gbytes_by_algo.png'), log_scale=True)

    grouped_bar(tm, 'Time per iteration (ms)', 'Execution Time per Iteration (lower = better)',
                os.path.join(out_dir, 'time_by_algo.png'), log_scale=True)

    if 'cpu_ref' in backends_found:
        speedup_chart(gf, 'cpu_ref', os.path.join(out_dir, 'speedup_vs_ref.png'))

    heatmap(rows, 'calc_ms', 'Execution Time Heatmap [ms] (log scale color)',
            os.path.join(out_dir, 'time_heatmap.png'))

    radar_chart(gf, 'GFLOP/s', os.path.join(out_dir, 'radar_gflops.png'))

    power_heatmap(rows, out_dir)
    power_chart(rows, os.path.join(out_dir, 'power_efficiency.png'))

    print("\nDone! Charts generated:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.png'):
            print(f"  {out_dir}/{f}")


if __name__ == '__main__':
    main()
