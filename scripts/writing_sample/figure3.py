"""Generate final-paper benchmark-sensitivity figures from verified stored results.

The main figure scales the fitted DAILY intercept before compounding. The
appendix figure preserves affine interpolation of separately winsorized endpoint
outcomes. Only presentation assets under paper/ are written.
"""
from pathlib import Path
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/v4/writing_sample/results.csv'
FIGURES = ROOT / 'paper/figures'
PRESENTATION = ROOT / 'paper/writing_sample_generated'
sys.path.insert(0, str(ROOT / 'paper/scripts'))
from palatino_plot_font import configure_palatino_font


def one(frame, statistic):
    values = frame.loc[frame.statistic == statistic, 'value']
    if len(values) != 1:
        raise ValueError(f'Expected one {statistic}, found {len(values)}')
    return float(values.iloc[0])


def curve(frame, analysis, prefix):
    rows = frame[(frame.analysis == analysis) & (frame.clock == 'T') &
                 (frame.scale == 'winsorized') & frame.term.str.startswith(prefix)]
    result = rows.pivot(index='lambda_value', columns='statistic', values='value').sort_index()
    grid = result.index.to_numpy(float)
    result = result[np.isclose(grid * 100, np.round(grid * 100), atol=1e-9)]
    if len(result) != 101 or not np.allclose(result.index[[0, -1]], [0, 1]):
        raise ValueError(f'{analysis} must have 101 grid points on [0,1]')
    return result


def save(fig, name):
    outputs = {}
    for ext in ('pdf', 'png'):
        path = FIGURES / f'{name}.{ext}'
        metadata = {'Creator': 'Congressional Trading final-paper generator',
                    'CreationDate': None, 'ModDate': None} if ext == 'pdf' else None
        fig.savefig(path, dpi=220, metadata=metadata)
        outputs[ext] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    return outputs


def main_figure(data):
    q = curve(data, 'figure3_fractional', 'curve;')
    plug = data[(data.analysis == 'figure3_plugin') & (data.clock == 'T')]
    point = one(plug[plug.scale == 'winsorized'], 'K_at_lambda_hat')
    low = one(plug[plug.scale == 'winsorized'], 'joint_ci_low')
    high = one(plug[plug.scale == 'winsorized'], 'joint_ci_high')
    lam = one(plug[plug.scale == 'raw'], 'lambda_hat')
    forecast = data[(data.analysis == 'A1_purchases') & (data.clock == 'T') &
                    (data.scale == 'raw') &
                    (data.term == 'sample=cohort;subset=all;regression=Y0_on_x')]
    lam_low, lam_high = one(forecast, 'ci_low'), one(forecast, 'ci_high')
    if not np.isclose(lam, one(forecast, 'estimate'), atol=1e-12):
        raise AssertionError('Forecast and plug-in coefficient disagree')
    roots = data[(data.analysis == 'figure3_fractional') & (data.clock == 'T') &
                 (data.scale == 'winsorized') & data.statistic.str.startswith('root_')]
    root = one(roots, 'root_1')
    x = q.index.to_numpy(float)
    fig, (ax, ruler) = plt.subplots(2, 1, figsize=(6.5, 4.35),
                                   gridspec_kw={'height_ratios': [3.45, 1]}, sharex=True)
    fig.subplots_adjust(left=.15, right=.98, top=.965, bottom=.17, hspace=.12)
    band = ax.fill_between(x, q.ci_low.to_numpy(), q.ci_high.to_numpy(),
                           color='.86', linewidth=0, label='Pointwise 95% band at fixed $c$')
    line, = ax.plot(x, q.estimate.to_numpy(), color='.1', lw=1.5,
                   label='Covariance estimate')
    ax.axhline(0, color='.55', lw=.7, zorder=1)
    ax.scatter([root], [0], s=19, facecolor='white', edgecolor='.15', lw=.8, zorder=5)
    ax.annotate(f'Zero crossing: {root:.3f}', (root, 0), xytext=(7, -15),
                textcoords='offset points', fontsize=8.2, ha='left')
    marker = ax.errorbar([lam], [point], yerr=[[point-low], [high-point]],
                         color='.1', marker='D', markerfacecolor='white', markersize=4.2,
                         linestyle='none', linewidth=1.05, capsize=3, zorder=6,
                         label=r'At $c=\hat\lambda$: joint 95% interval')
    ax.annotate(f'{point:,.0f}'.replace('-', '\u2212'), (lam, point), xytext=(9, -4),
                textcoords='offset points', ha='left', va='top', fontsize=8.5,
                bbox={'facecolor':'white', 'edgecolor':'none', 'pad':.6})
    ax.set(xlim=(-.17, 1.02), ylim=(-4000, 8000), ylabel=r'Covariance (bp$^2$)')
    ax.set_yticks(np.arange(-4000, 8001, 2000))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(handles=[line, band, marker], loc='upper left', frameon=False,
              fontsize=8.5, handlelength=2.3, labelspacing=.35, borderaxespad=.2)
    # A separate row distinguishes uncertainty in the forecast coefficient from the
    # covariance interval. Negative values are shown without extrapolating the curve.
    ruler.set_ylim(-.9, 1.0)
    ruler.set_yticks([])
    ruler.spines['left'].set_visible(False)
    ruler.axvline(0, color='.8', lw=.6)
    ruler.errorbar([lam], [0], xerr=[[lam-lam_low], [lam_high-lam]], color='.15',
                   marker='o', ms=4.2, capsize=3, linewidth=1.2, zorder=3)
    ruler.text(.32, .05, 'Forecast coefficient: '+rf'$\hat\lambda={lam:.3f}$'+'\n'
               +f'95% clustered interval [{lam_low:.3f}, {lam_high:.3f}]'.replace('-', '\u2212'),
               va='center', ha='left', fontsize=8.5)
    ticks = [-.1, 0, .2, .4, .6, .8, 1]
    ruler.set_xticks(ticks, ['\u22120.1', '0\nFactor-only', '0.2', '0.4', '0.6', '0.8', '1\nEvent-study'])
    ruler.get_xticklabels()[-1].set_ha('right')
    ruler.set_xlabel(r'Fraction $c$ of recent alpha carried forward', labelpad=6)
    result = save(fig, 'writing_sample_alpha_curve')
    return dict(outputs=result, root=root, forecast=dict(estimate=lam, low=lam_low, high=lam_high),
                covariance=dict(estimate=point, joint_low=low, joint_high=high),
                curve='Fitted daily intercept multiplied by c before benchmark compounding; period-specific winsorization',
                covariance_band='Pointwise 95% percentile band at fixed c',
                marker='Joint household bootstrap of lambda and covariance; 4,999 common selections',
                forecast_interval='Two-way household/entry-block clustered Student interval, df=37; separate lower row')


def affine_figure(data):
    q = curve(data, 'A2', 'curve')
    subset = data[(data.analysis == 'A2') & (data.clock == 'T') & (data.scale == 'winsorized')]
    root = one(subset, 'root_1')
    x = q.index.to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    fig.subplots_adjust(left=.15, right=.98, bottom=.22, top=.97)
    band = ax.fill_between(x, q.ci_low.to_numpy(), q.ci_high.to_numpy(), color='.86',
                           linewidth=0, label='95% pointwise bootstrap interval')
    line, = ax.plot(x, q.estimate.to_numpy(), color='.1', lw=1.5, label='Covariance estimate')
    ax.axhline(0, color='.55', lw=.7)
    ax.scatter([root], [0], s=19, facecolor='white', edgecolor='.1', lw=.8, zorder=5)
    ax.annotate(f'Zero crossing: {root:.3f}', (root, 0), xytext=(7, -14),
                textcoords='offset points', fontsize=8.5)
    ax.set(xlim=(0, 1), ylim=(-4000, 8000), xlabel=r'Fraction $c$ on the event-study endpoint',
           ylabel=r'Covariance (bp$^2$)')
    ax.set_xticks(np.linspace(0, 1, 6), ['0\nFactor-only', '0.2', '0.4', '0.6', '0.8', '1\nEvent-study'])
    ax.get_xticklabels()[0].set_ha('left')
    ax.get_xticklabels()[-1].set_ha('right')
    ax.set_yticks(np.arange(-4000, 8001, 2000))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.legend(handles=[line, band], loc='upper left', frameon=False, fontsize=8.5,
              handlelength=2.3, borderaxespad=.3)
    return dict(outputs=save(fig, 'writing_sample_alpha_affine'), root=root,
                curve='Exact affine interpolation of separately winsorized endpoints',
                band='Pointwise percentile interval, original 4,999 paired household resamples')


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    PRESENTATION.mkdir(parents=True, exist_ok=True)
    family, fonts = configure_palatino_font()
    plt.rcParams.update({'font.family': family, 'font.size': 9, 'axes.labelsize': 9.5,
                         'mathtext.fontset': 'custom', 'mathtext.rm': family,
                         'mathtext.it': family+':italic', 'mathtext.bf': family+':bold',
                         'pdf.fonttype':42, 'ps.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'xtick.direction':'out', 'ytick.direction':'out',
                         'axes.linewidth':.7})
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    data = pd.read_csv(SOURCE)
    outputs = {'main': main_figure(data), 'appendix_affine': affine_figure(data)}
    after = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if before != after:
        raise AssertionError('Presentation generation changed numerical source')
    manifest = dict(results_sha256=before, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    fonts=fonts, figures=outputs)
    (PRESENTATION/'figure3_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Generated main fractional-intercept and appendix affine sensitivity figures.')


if __name__ == '__main__':
    main()
