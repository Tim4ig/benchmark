#!/usr/bin/env python3
"""
plot_thesis.py - Publication-quality benchmark figures for bachelor thesis.

Usage:
    python3 scripts/plot_thesis.py results/latest.csv results/thesis_plots/

Generates:
    fig1_gflops_all.png       - Main GFLOPS comparison (all backends, all algos)
    fig2_speedup.png          - Speedup vs cpu_ref (log scale)
    fig3_categories.png       - 3-panel: memory-bound / compute-intensive / sorting
    fig4_hybrid.png           - Hybrid backend analysis (supported hybrid algos)
"""

import collections
import csv
import math
import matplotlib
import os
import sys

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# Style

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})

BACKENDS = ['cpu_ref', 'cpu_auto', 'cpu_avx512', 'cpu_mt', 'vulkan', 'hybrid']

BACKEND_LABEL = {
    'cpu_ref': 'CPU ref\n(-O0)',
    'cpu_auto': 'CPU auto\n(-O3 native)',
    'cpu_avx512': 'CPU AVX-512\n(intrinsics)',
    'cpu_mt': 'CPU MT\n(all cores)',
    'vulkan': 'GPU Vulkan\n(RDNA4)',
    'hybrid': 'Hybrid\n(CPU+GPU)',
}

BACKEND_COLOR = {
    'cpu_ref': '#e74c3c',
    'cpu_auto': '#e67e22',
    'cpu_avx512': '#27ae60',
    'cpu_mt': '#16a085',
    'vulkan': '#2980b9',
    'hybrid': '#8e44ad',
}

ALGO_LABEL = {
    'vecadd': 'VecAdd',
    'reduce': 'Reduce',
    'prefix': 'Prefix',
    'hist': 'Histogram',
    'conv2d': 'Conv2D',
    'spmv': 'SpMV',
    'matmul': 'MatMul',
    'blackscholes': 'Black-Scholes',
    'bsort': 'Bitonic Sort',
    'nbody': 'N-Body',
}

ALGO_ORDER = ['vecadd', 'reduce', 'prefix', 'hist', 'conv2d', 'spmv', 'matmul', 'blackscholes', 'bsort', 'nbody']


# Load data

def load(path):
    data = collections.defaultdict(dict)  # data[algo][backend] = {gflops, gbytes, ...}
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row['status']) != 0:
                continue
            data[row['algo']][row['backend']] = {
                'gflops': float(row['gflops']),
                'gbytes': float(row['gbytes']),
                'calc_ms': float(row['calc_ms']),
            }
    return data


def speedup(data, metric='gflops', ref='cpu_ref'):
    result = collections.defaultdict(dict)
    for algo in data:
        ref_val = data[algo].get(ref, {}).get(metric)
        if not ref_val:
            continue
        for b in data[algo]:
            v = data[algo][b].get(metric, 0)
            result[algo][b] = v / ref_val if ref_val > 0 else 0
    return result


# Bar helpers

def bar_group(ax, algos, backends, values, log=False, ylabel='', title='',
              bar_labels=True, label_fmt='{:.1f}'):
    n_algos = len(algos)
    n_backs = len(backends)
    x = np.arange(n_algos)
    width = 0.75 / n_backs

    for i, b in enumerate(backends):
        vals = [values.get(a, {}).get(b, 0) for a in algos]
        offset = (i - n_backs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.92,
                      label=BACKEND_LABEL[b].replace('\n', ' '),
                      color=BACKEND_COLOR[b], zorder=3,
                      edgecolor='white', linewidth=0.4)
        if bar_labels:
            for bar, v in zip(bars, vals):
                if v > 0:
                    txt = label_fmt.format(v)
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * (1.04 if not log else 1.3),
                            txt, ha='center', va='bottom',
                            fontsize=6.5, rotation=55, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABEL.get(a, a) for a in algos], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=8)
    if log:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(-0.6, n_algos - 0.4)


# ============================================================================
# Figure 1 - Main GFLOPS overview (all algorithms, all backends)
# ============================================================================

def fig1_gflops_all(data, out_dir):
    algos = [a for a in ALGO_ORDER if a in data]
    backends = [b for b in BACKENDS if any(b in data[a] for a in algos)]

    gf = {a: {b: data[a][b]['gflops'] for b in backends if b in data[a]} for a in algos}

    fig, ax = plt.subplots(figsize=(13, 5.5))
    bar_group(ax, algos, backends, gf, log=True,
              ylabel='GFLOP/s  (log scale)',
              title='Computational throughput - all backends and algorithms',
              label_fmt='{:.2g}')

    # legend
    handles = [mpatches.Patch(color=BACKEND_COLOR[b],
                              label=BACKEND_LABEL[b].replace('\n', ' '))
               for b in backends]
    ax.legend(handles=handles, loc='upper left', framealpha=0.85,
              ncol=len(backends), fontsize=9)

    # hardware note
    ax.text(0.99, 0.02,
            'CPU: AMD Ryzen 9 9950X (AVX-512)   GPU: AMD RX 9060 XT 16 GB (RDNA4, Vulkan)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.5, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.8))

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig1_gflops_all.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path}')


# ============================================================================
# Figure 2 - Speedup vs cpu_ref (log scale, only non-ref backends)
# ============================================================================

def fig2_speedup(data, out_dir):
    algos = [a for a in ALGO_ORDER if a in data]
    backends_plot = ['cpu_auto', 'cpu_avx512', 'cpu_mt', 'vulkan', 'hybrid']

    sp = speedup(data, 'gflops', 'cpu_ref')

    fig, ax = plt.subplots(figsize=(13, 5.5))
    n_algos = len(algos)
    n_backs = len(backends_plot)
    x = np.arange(n_algos)
    width = 0.75 / n_backs

    for i, b in enumerate(backends_plot):
        vals = [sp.get(a, {}).get(b, 0) for a in algos]
        offset = (i - n_backs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.92,
                      label=BACKEND_LABEL[b].replace('\n', ' '),
                      color=BACKEND_COLOR[b], zorder=3,
                      edgecolor='white', linewidth=0.4)
        for bar, v in zip(bars, vals):
            if v > 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.35,
                        f'{v:.1f}x', ha='center', va='bottom',
                        fontsize=6.5, rotation=55, color='#333333')

    ax.axhline(1.0, color='#e74c3c', linestyle='--', linewidth=1.2,
               label='cpu_ref  (1x baseline)', zorder=4)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 500])
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABEL.get(a, a) for a in algos], fontsize=9)
    ax.set_ylabel('Speedup vs cpu_ref  (log scale)')
    ax.set_title('Speedup relative to unoptimized baseline (cpu_ref)',
                 fontweight='bold', pad=8)
    ax.set_xlim(-0.6, n_algos - 0.4)

    handles = ([mpatches.Patch(color=BACKEND_COLOR[b],
                               label=BACKEND_LABEL[b].replace('\n', ' '))
                for b in backends_plot]
               + [plt.Line2D([0], [0], color='#e74c3c', linestyle='--',
                             linewidth=1.5, label='cpu_ref (1x)')])
    ax.legend(handles=handles, loc='upper left', framealpha=0.85, fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig2_speedup.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path}')


# ============================================================================
# Figure 3 - 3-panel by algorithm category
# ============================================================================

CATEGORIES = {
    'Memory-bound': ['vecadd', 'reduce', 'prefix', 'hist'],
    'Compute-intensive': ['matmul', 'blackscholes', 'nbody'],
    'Stencil / Sparse / Sort': ['conv2d', 'spmv', 'bsort'],
}


def fig3_categories(data, out_dir):
    backends_plot = ['cpu_ref', 'cpu_auto', 'cpu_avx512', 'cpu_mt', 'vulkan']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2),
                             gridspec_kw={'wspace': 0.35})

    for ax, (cat_name, algos) in zip(axes, CATEGORIES.items()):
        algos_present = [a for a in algos if a in data]
        gf = {a: {b: data[a][b]['gflops']
                  for b in backends_plot if b in data[a]}
              for a in algos_present}

        # determine log or linear
        vals_all = [v for a in gf for v in gf[a].values()]
        use_log = (max(vals_all) / (min(v for v in vals_all if v > 0) + 1e-9)) > 15

        bar_group(ax, algos_present, backends_plot, gf,
                  log=use_log,
                  ylabel='GFLOP/s' + (' (log)' if use_log else ''),
                  title=cat_name,
                  label_fmt='{:.2g}')

    # shared legend below
    handles = [mpatches.Patch(color=BACKEND_COLOR[b],
                              label=BACKEND_LABEL[b].replace('\n', ' '))
               for b in backends_plot]
    fig.legend(handles=handles, loc='lower center', ncol=len(backends_plot),
               bbox_to_anchor=(0.5, -0.04), fontsize=9, framealpha=0.85)

    fig.suptitle('Throughput by algorithm category', fontweight='bold',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig3_categories.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path}')


# ============================================================================
# Figure 4 - Hybrid backend deep-dive
# ============================================================================

def fig4_hybrid(data, out_dir):
    algos_hybrid = [a for a in ['matmul', 'nbody', 'blackscholes', 'reduce', 'vecadd']
                    if a in data and 'hybrid' in data[a]]
    if not algos_hybrid:
        return

    backends_plot = ['cpu_ref', 'cpu_auto', 'cpu_avx512', 'cpu_mt', 'vulkan', 'hybrid']

    fig, axes = plt.subplots(1, len(algos_hybrid),
                             figsize=(4.2 * len(algos_hybrid), 5.2),
                             gridspec_kw={'wspace': 0.38})
    if len(algos_hybrid) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos_hybrid):
        backs_present = [b for b in backends_plot if b in data[algo]]
        vals = [data[algo][b]['gflops'] for b in backs_present]
        colors = [BACKEND_COLOR[b] for b in backs_present]

        bars = ax.bar(range(len(backs_present)), vals, color=colors,
                      edgecolor='white', linewidth=0.5, zorder=3, width=0.65)

        # value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.04,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold', color='#222222')

        # highlight hybrid bar with bracket annotation
        if 'hybrid' in backs_present and 'vulkan' in backs_present:
            hi = backs_present.index('hybrid')
            vi = backs_present.index('vulkan')
            hybrid_v = data[algo]['hybrid']['gflops']
            vulkan_v = data[algo]['vulkan']['gflops']
            pct = hybrid_v / vulkan_v * 100
            ax.text(hi, vals[hi] * 1.18, f'{pct:.0f}% of GPU',
                    ha='center', va='bottom', fontsize=8,
                    color=BACKEND_COLOR['hybrid'], fontweight='bold')

        ax.set_xticks(range(len(backs_present)))
        ax.set_xticklabels([BACKEND_LABEL[b].replace('\n', '\n') for b in backs_present],
                           fontsize=8)
        ax.set_title(ALGO_LABEL.get(algo, algo), fontweight='bold', pad=8)
        ax.set_ylabel('GFLOP/s')

        # make y-axis start from 0 with headroom
        ax.set_ylim(0, max(vals) * 1.35)

    fig.suptitle('Hybrid (CPU + GPU) backend vs individual backends',
                 fontweight='bold', fontsize=13, y=1.01)

    handles = [mpatches.Patch(color=BACKEND_COLOR[b],
                              label=BACKEND_LABEL[b].replace('\n', ' '))
               for b in backends_plot if any(b in data[a] for a in algos_hybrid)]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               bbox_to_anchor=(0.5, -0.06), fontsize=9, framealpha=0.85)

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig4_hybrid.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path}')


# ============================================================================
# Figure 5 - Heatmap: speedup matrix (backends x algos)
# ============================================================================

def fig5_heatmap(data, out_dir):
    algos = [a for a in ALGO_ORDER if a in data]
    backends_plot = ['cpu_auto', 'cpu_avx512', 'cpu_mt', 'vulkan', 'hybrid']

    sp = speedup(data, 'gflops', 'cpu_ref')

    matrix = np.zeros((len(backends_plot), len(algos)))
    for bi, b in enumerate(backends_plot):
        for ai, a in enumerate(algos):
            matrix[bi, ai] = sp.get(a, {}).get(b, 0)

    fig, ax = plt.subplots(figsize=(12, 3.8))

    # Use log10 for color, but display actual values
    log_matrix = np.log10(np.where(matrix > 0, matrix, np.nan))
    im = ax.imshow(log_matrix, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=np.nanmax(log_matrix))

    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([ALGO_LABEL.get(a, a) for a in algos], fontsize=10)
    ax.set_yticks(range(len(backends_plot)))
    ax.set_yticklabels([BACKEND_LABEL[b].replace('\n', ' ') for b in backends_plot],
                       fontsize=10)
    ax.set_title('Speedup vs cpu_ref - heatmap (green = faster)',
                 fontweight='bold', pad=8)

    for bi in range(len(backends_plot)):
        for ai in range(len(algos)):
            v = matrix[bi, ai]
            if v > 0:
                txt = f'{v:.1f}x' if v < 100 else f'{v:.0f}x'
                brightness = log_matrix[bi, ai] / np.nanmax(log_matrix)
                color = 'white' if brightness > 0.75 else 'black'
                ax.text(ai, bi, txt, ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('log10(speedup)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig5_heatmap.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path}')


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'thesis_plots'
    os.makedirs(out_dir, exist_ok=True)

    data = load(csv_path)
    print(f'Loaded {sum(len(v) for v in data.values())} rows from {csv_path}')
    print(f'Output: {out_dir}/')
    print()

    fig1_gflops_all(data, out_dir)
    fig2_speedup(data, out_dir)
    fig3_categories(data, out_dir)
    fig4_hybrid(data, out_dir)
    fig5_heatmap(data, out_dir)

    print()
    print('Done.')


if __name__ == '__main__':
    main()
