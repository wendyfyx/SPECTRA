"""
Step 2c: figures for the synthetic experiments.

Run locally (not a qsub script), after 02b_synth_aggregate.py.

Reads the aggregated CSVs written by 02b_synth_aggregate.py and saves:
  synth-block_{sensitivity,fdp}.pdf                    block-level (Stage 1) results
  synth-cell_{sensitivity,fdp}_{diffuse,localized}.pdf cell-level results
  gompertz-N{XX}.pdf (+ gompertz-N{XX}.csv)            sample size needed for XX% sensitivity

USAGE
    python 02c_synth_plot.py --data_dir /path/to/synthetic_eval/fa_2d --out_dir /path/to/figures/synth
    python 02c_synth_plot.py --data_dir /path/to/synthetic_eval/fa_1d --out_dir /path/to/figures/synth_1d
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from spectra.math.curve_fit import fit_curve, predict_x

LARGE_FONTSIZE, REG_FONTSIZE, SMALL_FONTSIZE = 24, 22, 20

sns.set_style('darkgrid')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _method_label(row, method_base):
    parts = ['BH-FDR'] if method_base == 'fdr' else ['hFDR']
    if row.get('is_adaptive', False):
        parts.append('Adaptive')
    if method_base == 'hfdr':
        parts.append(row['use_resel_block'].upper())
    return '-'.join(parts)


def load_agg_csv(data_dir, method, ac_vals=(0.25,), is_block=False):
    if method == 'hfdr':
        kind = 'hfdr_block' if is_block else 'hfdr'
        df = pd.concat([pd.read_csv(f'{data_dir}/synth_eval_{kind}_ac{ac}.csv') for ac in ac_vals])
    else:
        df = pd.read_csv(f'{data_dir}/synth_eval_{method}.csv')
    df = df.loc[df.is_weighted == False]
    df = df.dropna(axis='columns', how='all')
    df['method'] = df.apply(_method_label, axis=1, method_base=method)
    df['fdr'] = (df['FP'] / (df['TP'] + df['FP'])).fillna(0.0)
    return df


def rename_for_plot(df):
    df = df.copy()
    df['ac'] = df['ac'].fillna('none')
    return df.rename(columns={
        'method': 'Correction Method',
        'recall': 'Sensitivity',
        'fdr': 'FDP',
        'sample_size': 'Ns',
        'effect_size': r'$d_{max}$',
        'ac': r'$\alpha_2$',
        'n_bundles_affected': 'B',
    })


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def method_colors(g, df_plot):
    """Read method -> color from seaborn's legend before it is replaced."""
    colors = {}
    for handle, text in zip(g.legend.legend_handles, g.legend.texts):
        if hasattr(handle, 'get_color') and text.get_text() in df_plot['Correction Method'].values:
            colors[text.get_text()] = handle.get_color()
    for i, m in enumerate(df_plot['Correction Method'].unique()):
        colors.setdefault(m, sns.color_palette()[i])
    return colors


def finish_grid(g, handles, title, ncol, out_path):
    g.legend.remove()
    g.figure.legend(
        handles=handles, title=title,
        bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=ncol,
        fontsize=REG_FONTSIZE, title_fontsize=LARGE_FONTSIZE, frameon=False,
    )
    g.figure.tight_layout(rect=[0, 0.07, 1, 1])
    g.set_titles(col_template=r'$d_{{max}}$ = {col_name}', row_template='B = {row_name}', size=LARGE_FONTSIZE)
    for ax in g.axes.flat:
        ax.set_xlabel('Sample Size (Ns)', fontsize=REG_FONTSIZE)
        ax.set_ylabel(ax.get_ylabel(), fontsize=REG_FONTSIZE)
        ax.tick_params(axis='both', labelsize=SMALL_FONTSIZE)
    g.figure.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(g.figure)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_block(df_block, out_dir):
    df_plot = rename_for_plot(df_block)
    localities = df_plot['locality'].unique()
    locality_ls = dict(zip(localities, ['-', '--', ':', '-.']))

    for metric in ['Sensitivity', 'FDP']:
        g = sns.relplot(
            data=df_plot, x='Ns', y=metric,
            hue='Correction Method', style='locality', markers=False,
            col=r'$d_{max}$', row='B', kind='line', err_style='bars',
            height=3.6, aspect=1.1,
        )
        if metric == 'FDP':
            g.refline(y=0.05, color='k', linestyle='--')

        colors = method_colors(g, df_plot)
        handles = [
            mlines.Line2D([], [], color=colors[m], linestyle=locality_ls[loc], linewidth=2,
                          label=f'{m} ({loc.capitalize()})')
            for m in df_plot['Correction Method'].unique() for loc in localities
        ]
        finish_grid(g, handles, 'Method', len(handles), out_dir / f'synth-block_{metric.lower()}.pdf')


def plot_cell(df_cell, ac_vals, out_dir):
    # First alpha_2 solid, second dashed, third dotted
    seaborn_dashes, line_styles = ['', (2, 2), (1, 1)], ['-', '--', ':']
    dashes = {'none': '', **{ac: seaborn_dashes[i] for i, ac in enumerate(ac_vals)}}
    entries = [('BH-FDR', None, '-'), ('BH-FDR-Adaptive', None, '-')] + [
        (m, ac, line_styles[i]) for m in ['hFDR-RESEL', 'hFDR-BUNDLE'] for i, ac in enumerate(ac_vals)
    ]
    for metric in ['Sensitivity', 'FDP']:
        for locality in ['diffuse', 'localized']:
            df_plot = rename_for_plot(df_cell.loc[df_cell.locality == locality])
            g = sns.relplot(
                data=df_plot, x='Ns', y=metric,
                hue='Correction Method', style=r'$\alpha_2$',
                dashes=dashes, markers=False,
                col=r'$d_{max}$', row='B', kind='line', err_style='bars',
                height=3.6, aspect=1.1,
            )
            if metric == 'FDP':
                g.refline(y=0.05, color='k', linestyle='--')

            colors = method_colors(g, df_plot)
            handles = [
                mlines.Line2D([], [], color=colors.get(m, 'gray'), linestyle=ls, linewidth=2,
                              label=m if ac is None else f'{m} (α2={ac})')
                for m, ac, ls in entries
            ]
            finish_grid(g, handles, f'Method ({locality.capitalize()})', len(handles) // 2,
                        out_dir / f'synth-cell_{metric.lower()}_{locality}.pdf')


def plot_gompertz(df_cell, target_recall, out_dir):
    label = f'N{int(round(target_recall * 100))}'
    df = df_cell.copy()
    df['method_label'] = df.apply(
        lambda r: r['method'] if pd.isna(r['ac']) else f"{r['method']} (α₂={r['ac']})", axis=1)

    group_cols = ['method_label', 'effect_size', 'n_bundles_affected', 'locality']
    rows = []
    for vals, gdf in df.groupby(group_cols):
        # Gompertz with floor 0 and ceiling 1 fixed (the fit_curve defaults)
        report = fit_curve(gdf['sample_size'].values.astype(float),
                           gdf['recall'].values.astype(float), model='gompertz')
        if not report['converged']:
            print(f'Fit failed for {dict(zip(group_cols, vals))}')
            continue
        rows.append({**dict(zip(group_cols, vals)), label: predict_x(target_recall, report),
                     'r2': report['r2']})
    fit_df = pd.DataFrame(rows)
    fit_df[label] = np.ceil(fit_df[label])
    min_n, max_n = df['sample_size'].min(), df['sample_size'].max()
    fit_df.to_csv(out_dir / f'gompertz-{label}.csv', index=False)
    print(f'Gompertz fits: {len(fit_df)} groups, median r2 = {fit_df.r2.median():.3f}')

    # Summary ratios quoted in the text
    loc = fit_df.pivot_table(index=['method_label', 'effect_size', 'n_bundles_affected'],
                             columns='locality', values=label).reset_index()
    print('Diffuse / localized ratio by method:')
    print((loc['diffuse'] / loc['localized']).groupby(loc['method_label']).mean().round(3).to_string())
    ref = fit_df[fit_df.method_label == 'BH-FDR'].rename(columns={label: 'ref'})[
        ['effect_size', 'n_bundles_affected', 'locality', 'ref']]
    m = fit_df.merge(ref, on=['effect_size', 'n_bundles_affected', 'locality'])
    print('Ratio vs BH-FDR by method:')
    print((m[label] / m['ref']).groupby(m['method_label']).mean().round(3).to_string())

    df_plot = fit_df.rename(columns={'method_label': 'Method', 'effect_size': r'$d_{max}$',
                                     'n_bundles_affected': 'B', 'locality': 'Locality'})
    df_plot['Locality'] = df_plot['Locality'].str.capitalize()
    methods = sorted(df_plot['Method'].unique())
    localities = ['Diffuse', 'Localized']
    g = sns.catplot(df_plot, kind='bar', y='Method', x=label, hue='Locality',
                    order=methods, hue_order=localities,
                    col=r'$d_{max}$', row='B', height=4.5, aspect=1.5)
    annot = dict(xytext=(2, 0), textcoords='offset points', va='center', ha='left',
                 fontsize=SMALL_FONTSIZE * 0.85)
    for (b, d), ax in g.axes_dict.items():
        ax.tick_params(axis='both', labelsize=SMALL_FONTSIZE)
        ax.set_xlabel(ax.get_xlabel(), fontsize=SMALL_FONTSIZE)
        ax.set_ylabel('')
        ax.set_title(ax.get_title(), fontsize=REG_FONTSIZE)
        # '*' = extrapolated outside the simulated sample sizes
        for container in ax.containers:
            for bar in container:
                w = bar.get_width()
                if np.isfinite(w) and w > 0:
                    ax.annotate(f'{w:.0f}' + ('*' if not min_n <= w <= max_n else ''),
                                xy=(w, bar.get_y() + bar.get_height() / 2), **annot)
        # 'n.r.' = not reached (no bar is drawn); hue bars are dodged by -/+0.2
        panel = df_plot[(df_plot['B'] == b) & (df_plot[r'$d_{max}$'] == d) & df_plot[label].isna()]
        for _, r in panel.iterrows():
            y = methods.index(r['Method']) + (-0.2 if r['Locality'] == 'Diffuse' else 0.2)
            ax.annotate('n.r.', xy=(0, y), **annot)
        ax.set_xlim(right=df_plot[label].max() * 1.2)   # room for the value labels
    handles, labels = g.legend.legend_handles, [t.get_text() for t in g.legend.get_texts()]
    g.legend.remove()
    g.figure.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
                    ncol=2, fontsize=SMALL_FONTSIZE * 1.5, title='', frameon=False)
    g.figure.tight_layout(rect=[0, 0.07, 1, 1])
    out_path = out_dir / f'gompertz-{label}.pdf'
    g.figure.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(g.figure)
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic-experiment subfigures')
    parser.add_argument('--data_dir', required=True,
                        help='Folder with synth_eval_*.csv from 02b_synth_aggregate.py')
    parser.add_argument('--out_dir', required=True, help='Folder for the figures')
    parser.add_argument('--ac_vals', nargs='+', type=float, default=[0.05, 0.15],
                        help='alpha_2 values shown in the cell-level and Gompertz figures')
    parser.add_argument('--target_recall', type=float, default=0.7,
                        help='Sensitivity target for the Gompertz sample-size figure (0.7 -> N70)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 (block) results do not depend on alpha_2, so load a single value
    df_block = load_agg_csv(args.data_dir, 'hfdr', ac_vals=args.ac_vals[:1], is_block=True)
    df_cell = pd.concat([
        load_agg_csv(args.data_dir, 'fdr'),
        load_agg_csv(args.data_dir, 'hfdr', ac_vals=args.ac_vals),
    ]).reset_index(drop=True)

    plot_block(df_block, out_dir)
    plot_cell(df_cell, args.ac_vals, out_dir)
    plot_gompertz(df_cell, args.target_recall, out_dir)


if __name__ == '__main__':
    main()
