"""Placebo size of the existing paper's spread test (plan §5 clustering).

One both-direction block-shift placebo set on the public clock: every purchase
and sale signal of the de-duplicated sample (all anchor years 2014-2026; the
2014 and 2026 anchors form their own block groups so that the generator's
period-relative blocks are defined), L = 20, shifts +-40..250 sessions, the N
generator (`null.block_shift`, `null.run_null_panels`) untouched. Every row
stores two outcomes: `y_mkt_h20` (market-adjusted against the S&P 500 price
return, the paper's headline outcome `ar_E_h20`) and `y_h20` (FF5+MOM
factor-adjusted). On every draw and outcome the 20-day purchase-minus-sale
spread is estimated with day clusters (household x shifted anchor day) and
with block clusters (household x 20-trading-day block of the shifted anchor);
the summary reports rejection rates at nominal 5% and coverage on the
timing-only (N1) and centered (N2) panels, and the observed de-duplicated
spreads' placebo p-values (market-adjusted placebo for the headline spread,
factor-adjusted placebo for the CAPM and FF5+MOM spread rows). Panels:
`data/v4/restricted/n/panels/P/<set>_bothdir/{N1,N2}/`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.null import block_shift as bs
from trading_on_disclosure_v4.null import dup_rules as dr
from trading_on_disclosure_v4.null import run_null_panels as rnp
from trading_on_disclosure_v4.outcomes import event_filters as ef
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel

BP = 1e4
ALPHA = 0.05
SET_NAME = "A0_P_purchase+sale_ABX_L20_both_alllines"
SET_ROOT = rnp.PANELS_ROOT / "P" / f"{SET_NAME}_bothdir"
STATS_DIR = panel.REPO_ROOT / "data" / "v4" / "restricted" / "a"
SUMMARY_PATH = rerun.OUT_DIR / "spread_placebo_size.csv"
MKT_CHECK_PATH = rerun.OUT_DIR / "market_outcome_check.json"
PERIODS = {"A": ("2015-01-01", "2020-12-31"), "B": ("2021-01-01", "2025-12-31"),
           "X14": ("2014-01-01", "2014-12-31"), "X26": ("2026-01-01", "2026-12-31")}
FLAVOURS = ("N1", "N2")
CLUSTERS = ("day", "block20")
OUTCOMES = {"y_mkt_h20": "market_adjusted", "y_h20": "ff5mom_adjusted"}  # panel column -> label
# observed de-duplicated spreads placed against each placebo outcome (test ids of rerun.TESTS)
OBSERVED_TESTS = {"market_adjusted": ("TW_P1_E20",), "ff5mom_adjusted": ("TW_S2_CAPM", "TW_S2_FF5MOM")}
PANEL_KEYS = ["draw", "event_id", "household_id", "shifted_anchor_date"]


def make_config(block_length: int = 20) -> bs.ShiftConfig:
    return bs.ShiftConfig(test_name=SET_NAME, block_length=block_length, direction="both",
                          outcomes=tuple(OUTCOMES), periods=dict(PERIODS))


def relabel_periods(events: pd.DataFrame) -> pd.DataFrame:
    """Period X (2014 / 2026 anchors) -> X14 / X26 so every block group has a start date."""
    out = events.copy()
    year = pd.to_datetime(out["anchor_date"]).dt.year
    is_x = out["period"] == "X"
    out.loc[is_x, "period"] = np.where(year[is_x] == 2014, "X14", "X26")
    if not set(out["period"]).issubset(PERIODS):
        raise ValueError(f"unexpected periods {sorted(set(out['period']) - set(PERIODS))}")
    return out


def load_events(cfg: bs.ShiftConfig, dedup: bool = True) -> tuple[pd.DataFrame, dict]:
    """De-duplicated public-clock purchases and sales, all anchor years, block-assigned."""
    ev = pd.read_parquet(ef.EVENTS_DIR / "events_P.parquet")
    ev = ev[ev["direction"].isin(["purchase", "sale"])].reset_index(drop=True)
    ev, report = dr.filter_events(ev, "P", "all_lines" if dedup else "keepdups")
    ev = bs.assign_blocks(relabel_periods(ev), panel.v2_trading_calendar(), cfg)
    report["n_by_direction"] = ev["direction"].value_counts().to_dict()
    report["n_by_period"] = ev["period"].value_counts().to_dict()
    return ev, report


def load_grid(cfg: bs.ShiftConfig) -> bs.CacheGrid:
    cols = ["permno", "anchor_idx", "anchor_date", "reason_h20"] + [bs.Y_COLUMNS[o] for o in cfg.outcomes]
    return bs.CacheGrid(fc.load_cache(columns=cols), outcomes=cfg.outcomes)


def draw_spreads(g: pd.DataFrame, direction: pd.Series, calendar: pd.DatetimeIndex,
                 outcome: str = "y_mkt_h20") -> dict:
    """Day- and block-clustered spread statistics of one placebo draw for one outcome (bp)."""
    d = g.dropna(subset=[outcome])
    y = d[outcome].to_numpy(float) * BP
    purchase = (d["event_id"].map(direction) == "purchase").to_numpy(float)
    hh = d["household_id"].to_numpy()
    day = pd.DatetimeIndex(d["shifted_anchor_date"])
    out = {"outcome": OUTCOMES.get(outcome, outcome), "n": int(len(d)), "n_purchase": int(purchase.sum()),
           "purchase_mean": float(y[purchase == 1].mean()), "sale_mean": float(y[purchase == 0].mean())}
    for clusters in CLUSTERS:
        cl = day.to_numpy() if clusters == "day" else rerun.block_of(pd.Series(day), calendar)
        r = rerun.clustered_mean_or_spread(y, purchase, hh, cl, f"anchor_{clusters}")
        out.update({f"{clusters}_{k}": v for k, v in r.items() if k not in ("n", "psd_corrected")})
        out[f"{clusters}_psd_corrected"] = bool(r["psd_corrected"])
    return out


def stats_for_set(flavour: str, direction: pd.Series, calendar: pd.DatetimeIndex, root: Path = SET_ROOT,
                  out_dir: Path = STATS_DIR, log_path: Path | None = None,
                  outcomes: tuple[str, ...] = tuple(OUTCOMES)) -> pd.DataFrame:
    """Per-draw x outcome spread statistics for every stored batch of one flavour (resumable per batch)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for p in sorted((root / flavour).glob(f"{flavour.lower()}_batch_*.parquet")):
        target = out_dir / f"spreads_{flavour}_{p.stem.split('_')[-1]}.parquet"
        if target.exists():
            rows.append(pd.read_parquet(target))
            continue
        batch = pd.read_parquet(p, columns=[*PANEL_KEYS, *outcomes])
        recs = [{"draw": int(d), **draw_spreads(g, direction, calendar, o)}
                for d, g in batch.groupby("draw", sort=True) for o in outcomes]
        frame = pd.DataFrame(recs).assign(flavour=flavour)
        frame.to_parquet(target, index=False)
        rows.append(frame)
        fc._log(log_path, f"{flavour} {p.name}: {len(frame)} draw x outcome rows, elapsed {time.time() - t0:.1f}s")
    return pd.concat(rows, ignore_index=True)


def _size_row(g: pd.DataFrame, clusters: str, alpha: float) -> dict:
    est, se, p, dof = g[f"{clusters}_estimate"], g[f"{clusters}_se"], g[f"{clusters}_p"], g[f"{clusters}_dof"]
    crit = stats.t.ppf(1 - alpha / 2, dof)
    target = float(est.mean())
    covers_target = ((est - crit * se) <= target) & (target <= (est + crit * se))
    return {
        "clusters": clusters, "n_draws": int(len(g)),
        "reject_two_sided_5pct": float((p < alpha).mean()),
        "reject_one_sided_5pct_positive": float((g[f"{clusters}_t"] > stats.t.ppf(1 - alpha, dof)).mean()),
        "coverage_of_zero": float(1 - (p < alpha).mean()),
        "reject_two_sided_5pct_of_placebo_mean": float(1 - covers_target.mean()),
        "coverage_of_placebo_mean": float(covers_target.mean()),
        "placebo_mean_spread_bp": target, "placebo_sd_spread_bp": float(est.std(ddof=1)),
        "placebo_q025_spread_bp": float(est.quantile(0.025)), "placebo_q975_spread_bp": float(est.quantile(0.975)),
        "mean_se_bp": float(se.mean()), "sd_over_mean_se": float(est.std(ddof=1) / se.mean()),
        "mean_dof": float(dof.mean()), "mean_n": float(g["n"].mean()),
        "placebo_mean_purchase_bp": float(g["purchase_mean"].mean()),
        "placebo_mean_sale_bp": float(g["sale_mean"].mean()),
        "mc_se_of_rejection_rate": float(np.sqrt(alpha * (1 - alpha) / len(g))),
        "psd_corrected_share": float(g[f"{clusters}_psd_corrected"].mean()) if f"{clusters}_psd_corrected" in g else np.nan,
    }


def summarize(per_draw: pd.DataFrame, observed: dict | None = None, alpha: float = ALPHA) -> pd.DataFrame:
    """Rejection rates and coverage per outcome x flavour x clusters, then one `placebo_test` row per
    observed spread (``observed`` = {outcome label: {test_id: estimate_bp}}); estimates are
    cluster-free so each placement uses the per-draw day-cluster estimate."""
    from trading_on_disclosure_v4.inference.placebo import placebo_p_value

    if "outcome" not in per_draw.columns:
        per_draw = per_draw.assign(outcome="market_adjusted")
    rows = []
    for (outcome, flavour), g in per_draw.groupby(["outcome", "flavour"], sort=True):
        rows += [{"outcome": outcome, "flavour": flavour, **_size_row(g, c, alpha)} for c in CLUSTERS]
    placements = []
    for (outcome, flavour), g in per_draw.groupby(["outcome", "flavour"], sort=True):
        for test_id, est in (observed or {}).get(outcome, {}).items():
            pl = placebo_p_value(est, g["day_estimate"].to_numpy(float), "two-sided")
            placements.append({"outcome": outcome, "flavour": flavour, "clusters": "placebo_test",
                               "n_draws": pl["n_null"], "placebo_mean_spread_bp": pl["null_mean"],
                               "placebo_sd_spread_bp": pl["null_sd"],
                               "placebo_q025_spread_bp": float(np.quantile(g["day_estimate"], 0.025)),
                               "placebo_q975_spread_bp": float(np.quantile(g["day_estimate"], 0.975)),
                               "observed_test_id": test_id, "observed_spread_bp": est, "observed_placebo_p": pl["p"]})
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(placements)], ignore_index=True)


def observed_dedup_spreads(tables_path: Path = rerun.TABLES_PATH,
                           tests: dict[str, tuple[str, ...]] = OBSERVED_TESTS) -> dict:
    """{outcome label: {test_id: de-duplicated estimate, bp}} from the re-run tables."""
    t = pd.read_csv(tables_path)
    t = t[(t["sample"] == "dedup") & (t["clusters"] == "day")].set_index("test_id")
    return {label: {tid: float(t.loc[tid, "estimate"]) for tid in tids} for label, tids in tests.items()}


def check_market_outcome(signals: pd.DataFrame | None = None, n: int = 2000, seed: int = 20260921,
                         cache_path: Path = fc.CACHE_PATH) -> dict:
    """Max |bhar_mkt_h20 at (permno, follower_entry_date) - ar_E_h20| over ``n`` random non-mixed signals
    with a 20-day public-clock outcome: the cache's market-adjusted column is the paper's outcome."""
    signals = rerun.load_signals() if signals is None else signals
    sub = signals.dropna(subset=["ar_E_h20", "follower_entry_date"])
    samp = sub.sample(min(n, len(sub)), random_state=seed)
    cache = fc.load_cache(cache_path, columns=["permno", "anchor_date", "bhar_mkt_h20"])
    cache["anchor_date"] = pd.to_datetime(cache["anchor_date"])
    m = samp.merge(cache, left_on=["permno", "follower_entry_date"], right_on=["permno", "anchor_date"], how="left")
    diff = (m["bhar_mkt_h20"] - m["ar_E_h20"]).abs()
    return {"n_signals_compared": int(diff.notna().sum()), "n_sampled": int(len(samp)),
            "n_cache_missing": int(m["bhar_mkt_h20"].isna().sum()), "max_abs_diff": float(diff.max()),
            "seed": seed, "cache_column": "bhar_mkt_h20", "signal_column": "ar_E_h20"}


def build_parser():
    import argparse

    ap = argparse.ArgumentParser(description="both-direction placebo set and the spread test's size")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--time-only", type=int, default=0, metavar="N", help="time N draws in memory and exit")
    ap.add_argument("--stats-only", action="store_true", help="skip generation; summarize stored panels")
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--set-root", type=Path, default=SET_ROOT, help="panel directory of the set")
    return ap


def main(argv: list[str] | None = None) -> pd.DataFrame | dict:
    args = build_parser().parse_args(argv)
    cfg = make_config()
    events, report = load_events(cfg)
    calendar = panel.v2_trading_calendar()
    direction = events.set_index("event_id")["direction"]
    if args.time_only:
        grid = load_grid(cfg)
        est = rnp.time_draws(events, grid, cfg, args.time_only)
        t0 = time.time()
        p, _ = bs.shift_one_draw(events, grid, cfg, 0)
        for o in OUTCOMES:
            draw_spreads(p, direction, calendar, o)
        est["seconds_per_draw_stats"] = round(time.time() - t0, 3)
        est["extrapolated_seconds"] = round(args.draws * (est["seconds_per_draw"] + 2 * est["seconds_per_draw_stats"]), 1)
        print(json.dumps({"set": SET_NAME, **est, "events": report}, indent=2, default=str))
        return est
    n1_dir, n2_dir = args.set_root / "N1", args.set_root / "N2"
    if not args.stats_only:
        grid = load_grid(cfg)
        cache_sha = json.loads(fc.MANIFEST_PATH.read_text())["sha256"]
        gen = rnp.generate_n1(events, grid, cfg, args.draws, n1_dir, args.batch_size, cache_sha, args.log)
        summary, means = rnp.finalize_n1(n1_dir, cfg, events)
        summary["seconds"] = round(gen["seconds"], 2)
        extra = {"flavour": "N1", "clock": "P", "directions": "purchase,sale", "periods": list(PERIODS),
                 "batch_size": args.batch_size, "cache_sha256": cache_sha, "duplicate_filter": report}
        rnp.write_manifest(n1_dir, summary, extra)
        w = rnp.write_n2(n1_dir, n2_dir, means, cfg, args.log)
        rnp.write_manifest(n2_dir, summary, {**extra, "flavour": "N2", "centering_sha256": w["centering_sha256"],
                                             "n2_centering_check_max_abs": rnp.n2_centering_check(n2_dir, cfg)})
    per_draw = pd.concat([stats_for_set(f, direction, calendar, root=args.set_root, log_path=args.log)
                          for f in FLAVOURS], ignore_index=True)
    per_draw.to_parquet(STATS_DIR / "spread_per_draw.parquet", index=False)
    out = summarize(per_draw, observed_dedup_spreads())
    out.insert(0, "set", SET_NAME)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUMMARY_PATH, index=False)
    check = check_market_outcome()
    MKT_CHECK_PATH.write_text(json.dumps(check, indent=2))
    fc._log(args.log, f"market outcome check: {json.dumps(check)}")
    print(out.to_string())
    return out


if __name__ == "__main__":
    main()
