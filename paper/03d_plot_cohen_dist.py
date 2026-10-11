"""
Step 3d: distribution of Cohen's d over the cells significant under hFDR-RESEL
(alpha_cell = 0.15), 1D vs 2D side by side, one curve per metric.

Run locally (not a qsub script), after 03b_stats_aggregate.sh:
    python 03d_plot_cohen_dist.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
STATS_DIR = '/path/to/stats'
OUT_FILE  = '/path/to/figures/result-cohen-dist-sig.png'

METRICS    = ['fa', 'md', 'rd', 'axd']
MODES      = ['1d', '2d']
S_LEN, R_LEN = 5, 5
TEST       = 'dxDementia'
STATS_FILE = 'hfdr_resel__stats_all_ac15.csv'

METRIC_DICT = {'fa': 'FA', 'md': 'MD', 'rd': 'RD', 'axd': 'AxD'}
MODE_DICT   = {'1d': '1D', '2d': '2D'}

# Load data
frames = []
for metric in METRICS:
    for mode in MODES:
        folder = f'{metric}__s{S_LEN}_r{R_LEN}_{mode}__{TEST}'
        csv = os.path.join(STATS_DIR, folder, STATS_FILE)
        if not os.path.exists(csv):
            print(f'Missing: {csv}')
            continue
        df = pd.read_csv(csv)
        df['metric'] = METRIC_DICT[metric]
        df['mode']   = MODE_DICT[mode]
        frames.append(df)

df_all = pd.concat(frames, ignore_index=True)
df_sig = df_all[df_all['sig_hfdr']].copy()
print(f'Total cells: {len(df_all):,}  |  Significant: {len(df_sig):,}')

df_sig['metric'] = pd.Categorical(df_sig['metric'], categories=['FA', 'MD', 'RD', 'AxD'], ordered=True)
df_sig['mode']   = pd.Categorical(df_sig['mode'],   categories=['1D', '2D'], ordered=True)

# Plot
g = sns.displot(
    data=df_sig,
    x='d_cohen',
    hue='metric',
    col='mode',
    kind='kde',
    common_norm=False,      # each curve integrates to 1 (compare shapes, not counts)
    facet_kws={'sharey': True},
    height=3.5,
    aspect=1.5,
)

g.set_titles("Dementia vs. CN ({col_name})")

for ax in g.axes.flat:
    if ax.has_data():
        ax.axvline(x=0.0, color='k', linestyle='--')

for ax in g.axes.flat:
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel("Cohen's d (sig.)", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.set_title(ax.get_title(), fontsize=18)

sns.move_legend(
    g,
    title=None,
    bbox_to_anchor=(0.5, 0.98), loc='lower center',
    ncol=4,
    fontsize=18,
    frameon=True,
)

g.figure.tight_layout(rect=[0, 0.07, 1, 1])
g.figure.savefig(OUT_FILE, bbox_inches='tight', dpi=300)
print(f'Saved: {OUT_FILE}')
