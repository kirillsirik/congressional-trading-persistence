"""Rebuild the canonical results file of the continuation analysis from its component files.

python scripts/writing_sample/merge_results.py [--check]

Every producer (compute_available.py, each restricted consumer, and the computation round of 11 November 2026 in
trading_on_disclosure_v4.writing_sample) writes only its own file under data/v4/writing_sample/components/.
This script concatenates the components in sorted file order (each keeps its row order), fills the extended
identifier columns of the original ten-column rows deterministically, refuses duplicate keys, and writes
data/v4/writing_sample/results.csv. The output depends only on the component contents, so any producer can be
rerun in any order without erasing another analysis. --check also requires every registered group.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/v4/writing_sample'
COMPONENTS = OUT / 'components'
RESULT_COLUMNS = ['analysis', 'clock', 'scale', 'term', 'statistic', 'lambda_value', 'value', 'unit', 'inference', 'source',
                  'result_key', 'sample', 'horizon', 'benchmark', 'weighting', 'controls']
#: component file stem -> analysis groups it must contain (the registry the final build checks)
REQUIRED = {
    'available': ['A2', 'A5', 'A4_existing_mean', 'weight_check', 'existing_household_tests', 'A3_existing_event_study'],
    'consumer_A1_purchases': ['A1_purchases'], 'consumer_A1_securities': ['A1_securities'], 'consumer_A3': ['A3'],
    'consumer_A4': ['A4'], 'consumer_spread_limit': ['spread_limit'],
    'continuation_meta': ['continuation_meta'], 'placebo_levels': ['placebo_levels', 'placebo_continuation'],
    'robustness_both_benchmarks': ['S4_robustness', 'C14_duplicates', 'S4_sales_signed_decomposition', 'S4_overlap'],
    'figure3_fractional': ['figure3_fractional', 'figure3_plugin'], 'inference_extras': ['S2_slopes', 'S5_effective_counts', 'S5_intervals', 'S5_recurrence_regression', 'S5_sign_stability'],
    'benchmark_labels': ['S6_spread', 'S7_figure_G1', 'S7_labels'], 'factual_checks': ['factual_checks'],
}
#: deterministic identifiers for rows written in the original ten-column schema
LEGACY = {
    'A2': dict(sample='persistence_cohort', horizon='20', benchmark='affine_interpolation_Y0_plus_lambda_(Y1-Y0)', weighting='sqrt(n_A n_B)', controls='log(1+n_A), both periods'),
    'A5': dict(sample='persistence_cohort', horizon='20', benchmark='paired_event_study_factor_only', weighting='sqrt(n_A n_B)', controls='log(1+n_A), both periods'),
    'A4_existing_mean': dict(sample='persistence_cohort', horizon='20', benchmark='', weighting='equal purchase', controls=''),
    'weight_check': dict(sample='persistence_cohort', horizon='20', benchmark='', weighting='sqrt(n_A n_B)', controls=''),
    'existing_household_tests': dict(sample='full_n20', horizon='20', benchmark='event_study', weighting='', controls=''),
    'A3_existing_event_study': dict(sample='', horizon='20', benchmark='event_study', weighting='sqrt(n_A n_B)', controls='log(1+n_A), both periods'),
}


def registered_groups() -> dict:
    extra = COMPONENTS / 'registry.json'
    reg = dict(REQUIRED)
    if extra.exists():
        reg.update(json.loads(extra.read_text()))
    return reg


def _normalize(frame: pd.DataFrame, stem: str) -> pd.DataFrame:
    f = frame.copy()
    for c in RESULT_COLUMNS:
        if c not in f:
            f[c] = np.nan if c in ('lambda_value', 'value') else ''
    for c in ('sample', 'horizon', 'benchmark', 'weighting', 'controls'):
        f[c] = f[c].fillna('').astype(str).replace('nan', '')
    for a, ids in LEGACY.items():
        m = f['analysis'].eq(a)
        for c, v in ids.items():
            f.loc[m & f[c].eq(''), c] = v
    missing = f['result_key'].isna() | f['result_key'].astype(str).isin(['', 'nan'])
    lam = f['lambda_value'].map(lambda v: '' if pd.isna(v) else repr(float(v)))
    f.loc[missing, 'result_key'] = (f['analysis'] + '/' + f['clock'].astype(str) + '/' + f['scale'].astype(str) + '/' + f['term'].astype(str)
                                    + '/' + f['statistic'].astype(str) + np.where(lam.eq(''), '', '/lambda=' + lam))[missing]
    return f[RESULT_COLUMNS]


def merge(results: Path = OUT / 'results.csv', components: Path = COMPONENTS, require_all: bool = True) -> pd.DataFrame:
    files = sorted(p for p in components.glob('*.csv'))
    if not files:
        raise SystemExit(f'no component files under {components}')
    frames = []
    for p in files:
        f = pd.read_csv(p, keep_default_na=True)
        frames.append(_normalize(f, p.stem))
    merged = pd.concat(frames, ignore_index=True)
    if merged['result_key'].duplicated().any():
        raise SystemExit(f'duplicate result keys: {merged.loc[merged.result_key.duplicated(), "result_key"].head().tolist()}')
    five = merged[['analysis', 'clock', 'scale', 'term', 'statistic']].astype(str).assign(lam=merged['lambda_value'].astype(str))
    if five.duplicated().any():
        raise SystemExit('duplicate (analysis, clock, scale, term, statistic, lambda_value) rows')
    if require_all:
        have = {p.stem: set(pd.read_csv(p, usecols=['analysis'])['analysis']) for p in files}
        for stem, groups in registered_groups().items():
            if stem not in have:
                raise SystemExit(f'required component {stem}.csv is missing')
            lost = set(groups) - have[stem]
            if lost:
                raise SystemExit(f'component {stem}.csv lacks required groups {sorted(lost)}')
    results.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(results, index=False, float_format='%.17g', lineterminator='\n')
    digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (results.parent / 'results_merge_manifest.json').write_text(json.dumps(dict(results=results.name, results_sha256=hashlib.sha256(results.read_bytes()).hexdigest(),
        rows=int(len(merged)), components=digest, analyses=sorted(merged['analysis'].unique().tolist()), require_all=require_all), indent=2, sort_keys=True) + '\n')
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--check', action='store_true', help='require every registered group')
    a = ap.parse_args(); m = merge(require_all=a.check)
    print(f'{len(m)} rows from {m["analysis"].nunique()} analyses -> {OUT / "results.csv"}')


if __name__ == '__main__':
    main()
