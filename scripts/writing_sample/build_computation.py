"""One command for the continuation analysis of 11 November 2026.

  python scripts/writing_sample/build_computation.py --restricted   # every step, from the local restricted inputs
  python scripts/writing_sample/build_computation.py                # results file, figure and tests from stored components

The restricted run needs data/v4/restricted/ (forward cache, daily panel, event tables, placebo panels) and the
CRSP and factor files named in data/README.md. Without them only the last steps run, from the component files
already stored. The component factual_checks.csv (counts on the two households with unrecovered early histories)
is stored only: the check that produced it compares member identities with the cited papers and is not distributed.
Every command, its exit status and duration go to data/v4/writing_sample/computation_2026-11-11/logs/build_commands.log.
"""
from pathlib import Path
import argparse, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'data/v4/writing_sample/computation_2026-11-11/logs/build_commands.log'
R = 'data/v4/restricted/writing_sample'
RES = 'data/v4/writing_sample/results.csv'
CONSUMERS = [('A1_purchases', 'continuation_purchases'), ('A1_securities', 'continuation_security_grid'), ('A3', 'sales_signed_means'),
             ('A4', 'cohort_purchase_means'), ('spread_limit', 'spread_household_counts')]


def run(cmd: list[str]) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), MPLBACKEND='Agg')
    t0 = time.time()
    shown = ' '.join(c if ' ' not in c else f'"{c}"' for c in cmd).replace(sys.executable, 'python')
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, 'a') as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\texit={p.returncode}\t{time.time() - t0:.1f}s\t{shown}\n")
    if p.returncode:
        sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
        raise SystemExit(f'step failed: {shown}')
    print(f'ok ({time.time() - t0:.0f} s): {shown}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--restricted', action='store_true', help='rerun every computation from the restricted inputs')
    a = ap.parse_args()
    py = sys.executable
    m = lambda mod, *x: [py, '-m', f'trading_on_disclosure_v4.writing_sample.{mod}', *x]
    if a.restricted:
        for mod in ('fits', 'market_values', 'exports'):
            run(m(mod))
        run([py, 'scripts/writing_sample/compute_available.py'])
        for analysis, f in CONSUMERS:
            run([py, 'scripts/writing_sample/restricted_consumers.py', analysis, '--input', f'{R}/{f}.csv', '--results', RES])
        for mod in ('meta', 'placebo', 'robustness', 'fractional', 'inference_extras', 'spreads'):
            run(m(mod))
    run([py, 'scripts/writing_sample/merge_results.py', '--check'])
    run([py, 'scripts/writing_sample/figure3_fractional.py'])
    tests = ROOT / 'data/v4/writing_sample/computation_2026-11-11/logs/tests.log'
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), MPLBACKEND='Agg')
    p = subprocess.run([py, 'scripts/writing_sample/test_computations.py', '-v'], cwd=ROOT, env=env, capture_output=True, text=True)
    tests.write_text(p.stdout + p.stderr)
    with open(LOG, 'a') as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\texit={p.returncode}\t-\tpython scripts/writing_sample/test_computations.py -v\n")
    if p.returncode:
        raise SystemExit('tests failed; see ' + str(tests))
    print('ok: tests', flush=True)


if __name__ == '__main__':
    main()
