"""Consumers for A1 (purchases and the formation grid), A3, A4 and the spread-sample limit.

Inputs are exported from the existing licensed data (src/trading_on_disclosure_v4/writing_sample/exports.py); no
downloads occur. Each command writes its rows to ONE component file, data/v4/writing_sample/components/
consumer_<analysis>.csv (next to --results), and then rebuilds the canonical --results file from all components
(scripts/writing_sample/merge_results.py), so rerunning any producer never erases another analysis.
Use --synthetic-validation only for explicit synthetic tests.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/coherence'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from restricted_checks import require, finite, fit_scores
from aggregate_checks import moments
import merge_results

BASE = {'N': 7964, 'N_all': 24010, 'x': 49.50962667395324, 'Y0': -9.681031063265706, 'Y1': -63.218031430400494, 'Y1_se': 15.690833642102184}


def cgm_two_way(scores, household, block, k):
    """Two-way CR1 variance of a scalar contrast (household, block, household x block), never truncated.

    Each term carries G/(G-1) * (n-1)/(n-k). If the combined variance is not positive the SE is NaN and the
    largest one-way variance is reported separately as the correction."""
    scores = np.asarray(scores, float); n = len(scores)
    if n <= k: raise ValueError('Need observations > parameters')
    cl = pd.DataFrame({'h': np.asarray(household), 'b': np.asarray(block), 's': scores})
    parts, counts, factors = [], [], []
    for cols in (['h'], ['b'], ['h', 'b']):
        sums = cl.groupby(cols, dropna=False).s.sum(); G = len(sums)
        if G < 2: raise ValueError('Need at least two clusters in each partition')
        f = G / (G - 1) * (n - 1) / (n - k)
        parts.append(float(f * np.sum(sums.to_numpy() ** 2))); counts.append(G); factors.append(f)
    raw = parts[0] + parts[1] - parts[2]; valid = raw > 0
    return dict(variance_raw=raw, variance_valid=bool(valid), se=float(np.sqrt(raw)) if valid else np.nan,
                df=min(counts[:2]) - 1, household_clusters=counts[0], block_clusters=counts[1], household_block_cells=counts[2],
                variance_household=parts[0], variance_block=parts[1], variance_household_block=parts[2],
                factor_household=factors[0], factor_block=factors[1], factor_household_block=factors[2], k_parameters=k,
                se_correction_max_one_way=float(np.sqrt(max(parts[0], parts[1]))) if not valid else np.nan)


def inference(est, v, nulls=(0.0, 1.0)):
    se, df = v['se'], v['df']; out = dict(estimate=est, **v)
    if np.isfinite(se) and se > 0:
        c = t.ppf(.975, df); out.update(ci_low=est - c * se, ci_high=est + c * se, t_crit=c)
        for h in nulls: out[f'p_{"zero" if h == 0 else "one"}'] = 2 * t.sf(abs((est - h) / se), df)
    else:
        out.update(ci_low=np.nan, ci_high=np.nan, t_crit=np.nan, p_zero=np.nan, p_one=np.nan)
    return out


def fe_slope(d, ycol, xcol='x'):
    """OLS slope of y on x with entry-block fixed effects (within transformation); two-way CR1 with k = blocks + 1."""
    x = d[xcol].to_numpy(float); y = d[ycol].to_numpy(float)
    xd = x - d.groupby('entry_block')[xcol].transform('mean').to_numpy(float)
    yd = y - d.groupby('entry_block')[ycol].transform('mean').to_numpy(float)
    sxx = xd @ xd
    if sxx <= 1e-12: raise ValueError('No within-block regressor variation')
    slope = float(xd @ yd / sxx); res = yd - slope * xd; scores = xd * res / sxx
    blocks = int(d.entry_block.nunique())
    v = cgm_two_way(scores, d.household_index, d.entry_block, blocks + 1)
    out = inference(slope, v); out.update(observations=len(d), fixed_effects=blocks, within_sxx=float(sxx),
                                          singleton_blocks=int((d.groupby('entry_block').size() == 1).sum()))
    return out


def purchase_continuation(d, synthetic=False):
    cols = ['sample', 'event_id', 'household_index', 'entry_block', 'period', 'alpha20_bp', 'Y0_bp', 'Y1_bp', 'x_low', 'x_high', 'Y0_low', 'Y0_high', 'Y1_low', 'Y1_high']
    require(d, cols, ['sample', 'event_id']); finite(d, cols[5:])
    if not d.period.isin(['A', 'B']).all(): raise ValueError('Period must be A or B')
    if not synthetic:
        cohort = d[d['sample'] == 'cohort']
        if len(cohort) != BASE['N']: raise ValueError('Cohort must reproduce 7964 purchases')
        if len(d[d['sample'] == 'all_purchases']) != BASE['N_all']: raise ValueError('All-purchase sample must reproduce 24010 purchases')
        if not np.isclose(cohort.alpha20_bp.mean(), BASE['x'], atol=1e-7, rtol=1e-10): raise ValueError('Cohort alpha charge does not reproduce 49.5096 bp')
        if not np.isclose(cohort.Y0_bp.mean(), BASE['Y0'], atol=1e-7, rtol=1e-10): raise ValueError('Cohort factor-only mean does not reproduce baseline')
        if not np.isclose(cohort.Y1_bp.mean(), BASE['Y1'], atol=1e-7, rtol=1e-10): raise ValueError('Cohort event-study mean does not reproduce baseline')
    out = []
    for sample, g in d.groupby('sample'):
        for col in ['x', 'Y0', 'Y1']:
            for _, p in g.groupby('period'):
                if p[f'{col}_low'].nunique() != 1 or p[f'{col}_high'].nunique() != 1: raise ValueError('Cutoffs must be fixed within sample/period')
                if not (p[f'{col}_low'] < p[f'{col}_high']).all(): raise ValueError('Invalid cutoffs')
        for transform in ['winsorized', 'raw']:
            f = g.copy(); f['x'] = f.alpha20_bp; f['y0'] = f.Y0_bp; f['y1'] = f.Y1_bp
            if transform == 'winsorized':
                for target, base in [('x', 'x'), ('y0', 'Y0'), ('y1', 'Y1')]: f[target] = f[target].clip(lower=f[f'{base}_low'], upper=f[f'{base}_high'])
            clipped = {k: int((f[k] != g[src]).sum()) for k, src in [('x', 'alpha20_bp'), ('y0', 'Y0_bp'), ('y1', 'Y1_bp')]}
            choices = [('all', f)]
            if 'large_stock' in f and f.large_stock.notna().any():
                if not f.large_stock.dropna().isin([0, 1]).all(): raise ValueError('large_stock must equal 0 or 1')
                choices.append(('largest_third', f[f.large_stock == 1]))
            for size, s in choices:
                if s.empty: raise ValueError('Empty largest-third sample')
                r = s.copy(); r['remainder'] = r.y1 - r.y0 + r.x
                fits = {'Y0_on_x': fe_slope(s, 'y0'), 'Y1_on_x': fe_slope(s, 'y1'), 'remainder_on_x': fe_slope(r, 'remainder')}
                gap = fits['Y1_on_x']['estimate'] - (fits['Y0_on_x']['estimate'] - 1 + fits['remainder_on_x']['estimate'])
                if abs(gap) > 1e-9: raise AssertionError('Compounding/transformation slope identity failed')
                common = dict(sample=sample, scale=transform, subset=size, households=int(s.household_index.nunique()),
                              blocks=int(s.entry_block.nunique()), purchases_A=int((s.period == 'A').sum()), purchases_B=int((s.period == 'B').sum()),
                              clipped_x=clipped['x'] if size == 'all' else np.nan, clipped_Y0=clipped['y0'] if size == 'all' else np.nan,
                              clipped_Y1=clipped['y1'] if size == 'all' else np.nan, excluded_missing_size=int(f.large_stock.isna().sum()) if size != 'all' else 0)
                lam = fits['Y0_on_x']
                out.append(dict(**common, regression='Y0_on_x', **lam, event_study_slope=fits['Y1_on_x']['estimate'],
                                remainder_slope=fits['remainder_on_x']['estimate'], identity_gap=gap))
                for name in ('Y1_on_x', 'remainder_on_x'):
                    out.append(dict(**common, regression=name, **fits[name], event_study_slope=np.nan, remainder_slope=np.nan, identity_gap=np.nan))
    return pd.DataFrame(out)


def hac_summary(v, lags=3):
    v = np.asarray(v, float); n = len(v)
    if n < lags + 3: raise ValueError('Too few formation dates for NW(3)')
    e = v - v.mean(); meat = e @ e
    for lag in range(1, lags + 1): meat += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag])
    se = float(np.sqrt(meat) / n) if meat > 0 else np.nan; estimate = float(v.mean()); df = n - 1
    out = dict(estimate=estimate, se=se, df=df, nw_lags=lags, long_run_variance_raw=float(meat / n), formation_dates=n)
    if np.isfinite(se):
        c = t.ppf(.975, df); out.update(ci_low=estimate - c * se, ci_high=estimate + c * se, p_zero=2 * t.sf(abs(estimate / se), df), p_one=2 * t.sf(abs((estimate - 1) / se), df))
    else:
        out.update(ci_low=np.nan, ci_high=np.nan, p_zero=np.nan, p_one=np.nan)
    return out


def security_continuation(d, synthetic=False, per_date=None):
    cols = ['sample', 'security_id', 'formation_session', 'formation_date', 'fit_end_session', 'window_end_session', 'alpha20_bp', 'Y0_bp', 'Y1_bp', 'market_value']
    require(d, cols, ['sample', 'security_id', 'formation_session']); finite(d, ['formation_session', 'fit_end_session', 'window_end_session', 'alpha20_bp', 'Y0_bp', 'Y1_bp'])
    dates = pd.to_datetime(d.formation_date, errors='raise')
    if not dates.dt.year.between(2015, 2025).all(): raise ValueError('Formation dates must lie in 2015-2025')
    if not (d.fit_end_session == d.formation_session - 22).all() or not (d.window_end_session == d.formation_session + 20).all(): raise ValueError('Window convention differs from t-22 / t+20')
    mv = d.market_value.to_numpy(float)
    if (mv[np.isfinite(mv)] <= 0).any(): raise ValueError('Market values must be positive where present')
    rows = []
    for sample, g in d.groupby('sample'):
        sessions = np.sort(g.formation_session.unique())
        if not (np.diff(sessions) == 20).all(): raise ValueError('Formation grid must advance by exactly 20 trading days')
        for transform in ['winsorized', 'raw']:
            slopes = {'all': [], 'largest_third': []}; counts = {'all': 0, 'largest_third': 0}; size_ok = True
            for session, frame in g.groupby('formation_session', sort=True):
                frame = frame.copy(); frame['x'] = frame.alpha20_bp; frame['y'] = frame.Y0_bp
                if transform == 'winsorized':
                    for col in ['x', 'y']: frame[col] = frame[col].clip(*frame[col].quantile([.01, .99]).to_numpy())
                if 'large_stock' in frame:
                    large = frame[frame.large_stock == 1]
                else:
                    have = frame[np.isfinite(frame.market_value)]; large = have[have.market_value >= have.market_value.quantile(2 / 3)]
                for subset, sel in [('all', frame), ('largest_third', large)]:
                    X = np.column_stack([np.ones(len(sel)), sel.x])
                    if len(sel) < 3 or np.linalg.matrix_rank(X) < 2:
                        if subset == 'all': raise ValueError('Formation cross-section lacks variation')
                        size_ok = False; continue
                    coef = np.linalg.lstsq(X, sel.y.to_numpy(float), rcond=None)[0]; slopes[subset].append(coef[1]); counts[subset] += len(sel)
                    if per_date is not None:
                        per_date.append(dict(sample=sample, scale=transform, subset=subset, formation_session=int(session), formation_date=str(pd.Timestamp(frame.formation_date.iloc[0]).date()),
                                             securities=len(sel), intercept=coef[0], slope=coef[1], x_mean=float(sel.x.mean()), y_mean=float(sel.y.mean()),
                                             x_sd=float(sel.x.std(ddof=1)), x_low=float(frame.x.min()), x_high=float(frame.x.max())))
            for subset, values in slopes.items():
                if subset == 'largest_third' and not size_ok: continue
                if not values: continue
                rows.append(dict(sample=sample, scale=transform, subset=subset, observations=counts[subset], **hac_summary(values, 3)))
    return pd.DataFrame(rows)


def sales_signed(d, synthetic=False):
    """Exact covariance/decomposition from benchmark-paired household means (point values; uncertainty elsewhere)."""
    cols = ['clock', 'design', 'scale', 'household_index', 'n_A', 'n_B', 'Y1_A', 'Y1_B', 'Y0_A', 'Y0_B']
    require(d, cols, ['clock', 'design', 'scale', 'household_index']); finite(d, cols[3:])
    if not d.design.isin(['sales', 'signed_all']).all(): raise ValueError('Unknown sales/signed design')
    baseline = pd.read_csv(ROOT / 'data/v4/results.csv'); rows = []
    for (clock, design, scale), g in d.groupby(['clock', 'design', 'scale']):
        v = moments(g)[0]
        if not synthetic and scale == 'winsorized':
            old = baseline[(baseline.clock == clock) & (baseline.analysis == 'robustness') & (baseline.variant == design) & (baseline.statistic == 'tau2')]
            if len(old) != 1 or not np.isclose(v[0], old.value.iloc[0], atol=1e-7, rtol=1e-10): raise ValueError('Event-study covariance does not reproduce original sales/signed result')
            if len(g) != int(old.n_households.iloc[0]) or int(g.n_A.sum() + g.n_B.sum()) != int(old.n_events.iloc[0]): raise ValueError('Sales/signed counts differ from original sample')
        rows.append(dict(clock=clock, design=design, scale=scale, households=len(g), observations=int(g.n_A.sum() + g.n_B.sum()),
                         **dict(zip(['K11', 'K00', 'KDD', 'K0D', 'KD0', 'difference'], v)), identity_gap=v[0] - (v[1] + v[2] + v[3] + v[4])))
    return pd.DataFrame(rows)


def mean_standard_error(d, synthetic=False):
    """Intercept-only means with household x 20-session-block CR1 SEs; with alpha20_bp, also the charge, alpha and
    compounding components and their joint covariance (same estimator, same clusters)."""
    cols = ['event_id', 'household_index', 'entry_block', 'Y0_bp', 'Y1_bp']; require(d, cols, ['event_id']); finite(d, cols[1:])
    outcomes = {'Y0_bp': d.Y0_bp.to_numpy(float), 'Y1_bp': d.Y1_bp.to_numpy(float)}
    if 'alpha20_bp' in d:
        finite(d, ['alpha20_bp']); x = d.alpha20_bp.to_numpy(float)
        outcomes.update(charge_bp=outcomes['Y0_bp'] - outcomes['Y1_bp'], alpha20_bp=x, compounding_bp=outcomes['Y0_bp'] - outcomes['Y1_bp'] - x)
    rows = []; S = []
    for outcome, y in outcomes.items():
        estimate, scores = fit_scores(y, np.ones((len(d), 1)), np.ones(len(d)), [1]); S.append(scores)
        v = cgm_two_way(scores, d.household_index, d.entry_block, 1); r = inference(estimate, v, nulls=(0.0,))
        r.pop('p_one', None); rows.append(dict(outcome=outcome, observations=len(d), **r))
    frame = pd.DataFrame(rows)
    if not synthetic:
        r = frame[frame.outcome == 'Y1_bp'].iloc[0]
        if len(d) != BASE['N'] or not np.isclose(r.estimate, BASE['Y1'], atol=1e-7, rtol=1e-10) or not np.isclose(r.se, BASE['Y1_se'], atol=1e-7, rtol=1e-10):
            raise ValueError('Must reproduce original event-study mean and TWO-WAY SE before reporting A4')
    S = np.column_stack(S); n = len(d); cl = pd.DataFrame({'h': d.household_index.to_numpy(), 'b': d.entry_block.to_numpy()})
    V = np.zeros((S.shape[1], S.shape[1]))
    for sign, keys in ((1, ['h']), (1, ['b']), (-1, ['h', 'b'])):
        g = cl.groupby(keys).ngroup().to_numpy(); G = g.max() + 1
        sums = np.zeros((G, S.shape[1])); np.add.at(sums, g, S)
        V += sign * G / (G - 1) * sums.T @ sums
    names = list(outcomes)
    cov = pd.DataFrame(V, index=names, columns=names)
    frame.attrs['covariance'] = cov
    return frame


def spread_limit(d, synthetic=False, verify=True):
    cols = ['household_index', 'purchase_signals', 'sale_signals']; require(d, cols, ['household_index']); finite(d, cols)
    counts = d[['purchase_signals', 'sale_signals']].to_numpy(float)
    if (counts < 0).any() or (counts != np.floor(counts)).any() or (counts.sum(axis=1) < 1).any(): raise ValueError('Counts must be nonnegative integers on the union of active households')
    G = len(d); N = counts.sum(); Nbuy = counts[:, 0].sum(); Nsell = counts[:, 1].sum()
    if G < 2 or N <= 2 or Nbuy <= 0 or Nsell <= 0: raise ValueError('Need both directions and >=2 households')
    if not synthetic and (G != 325 or N != 39724): raise ValueError('Complete baseline spread sample must have 325 households and 39724 signals')
    w = counts[:, 0] / Nbuy; star = int(np.argmax(w)); q = G / (G - 1) * (N - 1) / (N - 2)
    sq_star = (1 - w[star]) ** 2; sq_other = float(np.sum(np.delete(w, star) ** 2))
    L = 1 / np.sqrt(q * (sq_star + sq_other))
    out = dict(households=G, signals=N, purchase_signals=Nbuy, sale_signals=Nsell, households_sales_only=int(((counts[:, 0] == 0)).sum()),
               recipient_household_index=d.household_index.iloc[star], recipient_purchase_signals=counts[star, 0], recipient_sale_signals=counts[star, 1],
               purchase_share=w[star], sum_purchase_shares_squared=float(np.sum(w ** 2)), one_minus_share_squared=sq_star,
               sum_other_shares_squared=sq_other, bracket=sq_star + sq_other, CR1_factor=q, limit=L)
    if verify:
        # t-statistic of the purchase-minus-sale spread with household CR1 as a gain to the recipient's purchases grows
        rng = np.random.default_rng(20261111); hh = np.repeat(np.arange(G), counts.sum(axis=1).astype(int))
        buy = np.concatenate([np.r_[np.ones(int(a)), np.zeros(int(b))] for a, b in counts]); base = rng.normal(0, 1, len(buy))
        X = np.column_stack([np.ones(len(buy)), buy])
        for gain in (1e2, 1e4, 1e6, 1e8):
            y = base + gain * ((hh == star) & (buy == 1))
            est, sc = fit_scores(y, X, np.ones(len(y)), [0, 1]); sums = pd.Series(sc).groupby(hh).sum().to_numpy()
            se = np.sqrt(G / (G - 1) * (N - 1) / (N - 2) * np.sum(sums ** 2)); out[f'verify_t_over_gain_ratio_{gain:.0e}'] = (est / se)
        out['verify_abs_gap_at_1e8'] = abs(out['verify_t_over_gain_ratio_1e+08'] - L)
    return pd.DataFrame([out])


UNITS = {'A1_purchases': {'estimate': 'slope', 'se': 'slope', 'ci_low': 'slope', 'ci_high': 'slope', 'variance_raw': 'slope^2'},
         'A1_securities': {'estimate': 'slope', 'se': 'slope', 'ci_low': 'slope', 'ci_high': 'slope'},
         'A3': {}, 'A4': {'estimate': 'bp', 'se': 'bp', 'ci_low': 'bp', 'ci_high': 'bp', 'variance_raw': 'bp2'},
         'spread_limit': {}}
COUNT_KEYS = ('observations', 'households', 'blocks', 'purchases_A', 'purchases_B', 'df', 'household_clusters', 'block_clusters', 'household_block_cells',
              'fixed_effects', 'singleton_blocks', 'k_parameters', 'formation_dates', 'signals', 'purchase_signals', 'sale_signals', 'clipped_x', 'clipped_Y0',
              'clipped_Y1', 'excluded_missing_size', 'households_sales_only', 'recipient_purchase_signals', 'recipient_sale_signals', 'nw_lags')
INFERENCE = {'A1_purchases': 'OLS with global 20-session entry-block fixed effects, equal purchase weights; two-way CR1 (household, block, minus household x block), k = blocks + 1, df = min(G_h, G_b) - 1; variance never truncated',
             'A1_securities': 'cross-sectional OLS per formation date; equal-date mean of slopes; Bartlett Newey-West, 3 lags; t(dates - 1)',
             'A3': 'point decomposition on benchmark-paired household means (w = sqrt(n_A n_B), activity log(1+n_A) in both periods)',
             'A4': 'intercept-only OLS; two-way CR1 household x 20-session block (k = 1); df = min(G_h, G_b) - 1',
             'spread_limit': 'limit of the household-CR1 t-statistic of the purchase-minus-sale spread under a gain to the largest purchase-share household'}


def unit_of(analysis, key):
    if key in COUNT_KEYS: return 'count'
    if key.startswith('p_') or key.startswith('variance_valid'): return 'p' if key.startswith('p_') else 'indicator'
    if analysis == 'A3': return 'bp2' if key in ('K11', 'K00', 'KDD', 'K0D', 'KD0', 'difference', 'identity_gap') else 'unitless'
    if analysis == 'A4' and key.startswith('variance'): return 'bp2'
    if analysis == 'A4' and key.startswith('factor'): return 'unitless'
    if analysis == 'spread_limit': return 'share' if 'share' in key else 'unitless'
    return UNITS.get(analysis, {}).get(key, 'unitless' if key.startswith(('factor', 't_crit', 'within_sxx', 'long_run')) else 'slope')


def tidy(frame, analysis, source, synthetic):
    identifiers = ['clock', 'design', 'sample', 'scale', 'subset', 'outcome', 'regression']; rows = []
    for _, r in frame.iterrows():
        term = ';'.join(f'{k}={r[k]}' for k in identifiers if k in frame and k not in ['clock', 'scale'])
        for key, value in r.items():
            if key in identifiers: continue
            try: v = float(value)
            except (TypeError, ValueError): continue
            rows.append(dict(analysis=analysis, clock=r.get('clock', 'T'), scale=r.get('scale', 'raw'), term=term, statistic=key, lambda_value=np.nan,
                             value=v, unit=unit_of(analysis, key), inference='SYNTHETIC VALIDATION ONLY' if synthetic else INFERENCE[analysis], source=source,
                             result_key=f'{analysis}/{r.get("clock", "T")}/{r.get("scale", "raw")}/{term}/{key}', sample=str(r.get('sample', r.get('design', ''))),
                             horizon='20', benchmark={'Y0_on_x': 'factor_only', 'Y1_on_x': 'event_study', 'remainder_on_x': 'remainder'}.get(r.get('regression', ''), ''),
                             weighting='equal purchase' if analysis in ('A1_purchases', 'A4') else ('equal date' if analysis == 'A1_securities' else ''),
                             controls='entry-block fixed effects' if analysis == 'A1_purchases' else ''))
    return pd.DataFrame(rows, columns=merge_results.RESULT_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('analysis', choices=['A1_purchases', 'A1_securities', 'A3', 'A4', 'spread_limit'])
    parser.add_argument('--input', required=True, type=Path); parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--synthetic-validation', action='store_true'); args = parser.parse_args()
    if not args.input.is_file(): parser.error('Required input is absent; no empirical statistic was computed')
    data = pd.read_csv(args.input); per_date = [] if args.analysis == 'A1_securities' else None
    if args.analysis == 'A1_securities': output = security_continuation(data, args.synthetic_validation, per_date)
    else: output = {'A1_purchases': purchase_continuation, 'A3': sales_signed, 'A4': mean_standard_error, 'spread_limit': spread_limit}[args.analysis](data, args.synthetic_validation)
    try: src = str(args.input.resolve().relative_to(ROOT))
    except ValueError: src = args.input.name
    fresh = tidy(output, args.analysis, src, args.synthetic_validation)
    if args.analysis == 'A4' and 'covariance' in output.attrs:
        cov = output.attrs['covariance']; extra = []
        for a in cov.index:
            for b in cov.columns:
                extra.append(dict(analysis='A4', clock='T', scale='raw', term=f'covariance;row={a};col={b}', statistic='joint_covariance', lambda_value=np.nan,
                                  value=float(cov.loc[a, b]), unit='bp2', inference='SYNTHETIC VALIDATION ONLY' if args.synthetic_validation else INFERENCE['A4'] + '; joint score covariance, same clusters and factors',
                                  source=src, result_key=f'A4/T/raw/covariance/{a}/{b}', sample='cohort', horizon='20', benchmark='', weighting='equal purchase', controls=''))
        fresh = pd.concat([fresh, pd.DataFrame(extra)], ignore_index=True)
    comp_dir = args.results.parent / 'components'; comp_dir.mkdir(parents=True, exist_ok=True)
    comp = comp_dir / f'consumer_{args.analysis}.csv'
    fresh.to_csv(comp, index=False, float_format='%.17g', lineterminator='\n')
    if per_date is not None:
        diag = args.results.parent / 'diagnostics'; diag.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_date).to_csv(diag / f'{args.analysis}_per_date.csv', index=False, float_format='%.17g', lineterminator='\n')
    merge_results.merge(args.results, comp_dir, require_all=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (comp_dir / f'consumer_{args.analysis}_manifest.json').write_text(json.dumps(dict(analysis=args.analysis, synthetic=args.synthetic_validation, input_sha256=sha(args.input),
        script_sha256=sha(Path(__file__)), component_sha256=sha(comp), input=src, component=comp.name), indent=2) + '\n')
    with pd.option_context('display.width', 250, 'display.max_columns', 40): print(output.to_string(index=False))


if __name__ == '__main__': main()
