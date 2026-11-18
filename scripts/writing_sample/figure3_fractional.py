"""Fractional-intercept sensitivity (separate from Figure 3's affine curve), drawn from the canonical results file.

python scripts/writing_sample/figure3_fractional.py
Reads data/v4/writing_sample/results.csv (analyses figure3_fractional and figure3_plugin, trade date) and writes
data/v4/writing_sample/computation_2026-11-11/figure3/figure3_fractional_intercept.{pdf,png} plus the plotted data.
The marker is the covariance at c = lambda_hat (purchase forecast coefficient) with the JOINT household-bootstrap
interval; the shaded band is pointwise at fixed c and does not account for estimating lambda.
"""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'data/v4/writing_sample/results.csv'
OUT = ROOT / 'data/v4/writing_sample/computation_2026-11-11/figure3'


def curve(d, scale, analysis='figure3_fractional'):
    g = d[(d.analysis == analysis) & (d.clock == 'T') & (d.scale == scale) & d.term.str.startswith('curve')]
    q = g.pivot_table(index='lambda_value', columns='statistic', values='value').sort_index()
    return q


def main():
    d = pd.read_csv(RES)
    fr = curve(d, 'winsorized')
    grid = fr.index.to_numpy(float)
    on_grid = np.isclose(grid * 100, np.round(grid * 100), atol=1e-9)
    fr_g = fr[on_grid]
    a2 = d[(d.analysis == 'A2') & (d.clock == 'T') & (d.scale == 'winsorized') & (d.term == 'curve')].pivot_table(index='lambda_value', columns='statistic', values='value').sort_index()
    plug = d[(d.analysis == 'figure3_plugin') & (d.clock == 'T')]
    get = lambda s, sc: float(plug[(plug.statistic == s) & (plug.scale == sc)].value.iloc[0])
    lam = get('lambda_hat', 'raw')
    k_l, lo_l, hi_l = get('K_at_lambda_hat', 'winsorized'), get('joint_ci_low', 'winsorized'), get('joint_ci_high', 'winsorized')
    roots = d[(d.analysis == 'figure3_fractional') & (d.clock == 'T') & (d.scale == 'winsorized') & d.statistic.str.startswith('root_')].value.to_numpy(float)
    if len(fr_g) != 101:
        raise ValueError('fractional curve must have 101 grid points')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'axes.labelsize': 10.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.35, 3.85)); fig.subplots_adjust(left=.17, right=.98, bottom=.22, top=.96)
    x = fr_g.index.to_numpy(float)
    band = ax.fill_between(x, fr_g.ci_low, fr_g.ci_high, color='.86', linewidth=0, label='95% pointwise band at fixed c')
    line, = ax.plot(x, fr_g.estimate, color='.10', lw=1.8, label=r'Fractional intercept $Z(c)$')
    aff, = ax.plot(a2.index.to_numpy(float), a2.estimate, color='.45', lw=1.0, ls=(0, (4, 3)), label='Affine endpoint curve (Figure 3)')
    ax.axhline(0, color='.55', lw=.7, zorder=1)
    for r in roots:
        ax.scatter([r], [0], s=20, facecolor='white', edgecolor='.10', linewidth=.9, zorder=4)
        ax.annotate(f'{r:.3f}', xy=(r, 0), xytext=(6, -13), textcoords='offset points', fontsize=8.5)
    eb = ax.errorbar([lam], [k_l], yerr=[[k_l - lo_l], [hi_l - k_l]], fmt='s', ms=4.5, color='#b2182b', mfc='white', capsize=3, lw=1.1,
                     label=r'$c=\hat\lambda$ (purchase forecast), joint bootstrap 95%')
    ax.set(xlim=(-0.02, 1.0), ylim=(-4000, 8000), xlabel=r'Share $c$ of the fitted daily intercept in the benchmark', ylabel=r'Cross-period covariance (bp$^2$)')
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_xticklabels(['0\nFactor-only', '0.2', '0.4', '0.6', '0.8', '1\nEvent-study'])
    ax.get_xticklabels()[0].set_ha('left'); ax.get_xticklabels()[-1].set_ha('right')
    ax.set_yticks(np.arange(-4000, 8001, 2000)); ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.legend(handles=[line, band, aff, eb], loc='upper left', frameon=False, fontsize=8.2, handlelength=2.2)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'figure3_fractional_intercept.{ext}', dpi=200, metadata={'Creator': 'Congressional Trading replication code', 'CreationDate': None} if ext == 'pdf' else None)
    plt.close(fig)
    data = fr.reset_index().rename(columns={'lambda_value': 'c', 'estimate': 'covariance_bp2', 'ci_low': 'pointwise_low_bp2', 'ci_high': 'pointwise_high_bp2'})
    data.to_csv(OUT / 'figure3_fractional_plot_data.csv', index=False, float_format='%.17g', lineterminator='\n')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / 'figure3_fractional_figure_manifest.json').write_text(json.dumps(dict(results_sha256=sha(RES), script_sha256=sha(Path(__file__)),
        outputs={e: sha(OUT / f'figure3_fractional_intercept.{e}') for e in ('pdf', 'png')}, lambda_hat=lam, marker=dict(K=k_l, joint_low=lo_l, joint_high=hi_l),
        roots=roots.tolist(), note='separate sensitivity; the original affine Figure 3 is unchanged and carries no marker'), indent=2) + '\n')


if __name__ == '__main__':
    main()
