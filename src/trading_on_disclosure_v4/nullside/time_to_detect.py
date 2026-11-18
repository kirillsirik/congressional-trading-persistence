"""A6 time-to-detect table (plan §7.4, Figure 4), pre-tag: sigma pooled, d as a sensitivity range.

T = [(z_{1-alpha} + z_{power}) sigma sqrt(d) / theta_20]^2 / n (`estimators.ttd`). sigma = pooled
event-level SD of the 20-session FF5+MOM purchase outcome by clock (A u B), raw and winsorized at
the pooled 1/99 percentiles by clock x period (`data/v4/pooled/pooled_outcome_distribution.csv`);
the event-level join never groups by household. d in {1, 1.5, 2, 3}; theta annualized in
{2, 5, 10, 20}%; n in {10, 25, 50, 100, 250, 500}; plain and Bonferroni best-of-H.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import ttd
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel, pooled

BP = 1e4
D_GRID = (1.0, 1.5, 2.0, 3.0)
POOLED_CSV = pooled.OUT_PATH
DIAG = panel.REPO_ROOT / "data" / "v4" / "diagnostics"
OBSERVED_SPAN_YEARS = 11  # anchors 2015-01-01 .. 2025-12-31 (plan §2)


def r1_household_counts(path=DIAG / "eligible_households_by_rule_clock.csv", dup_filter: str = "none") -> dict:
    """R1-eligible household count per clock from D's diagnostics (counts only)."""
    t = pd.read_csv(path)
    t = t[(t["dup_filter"] == dup_filter) & (t["rule"] == "R1")]
    return {r.clock: int(r.n_households) for r in t.itertuples()}


def winsor_cutoffs(path=POOLED_CSV, horizon: str = "ff5mom_h20") -> pd.DataFrame:
    t = pd.read_csv(path)
    t = t[(t["horizon"] == horizon) & (t["direction"] == "purchase")]
    return t[["clock", "period", "n_ok", "sd", "winsor_lo", "winsor_hi"]].reset_index(drop=True)


def sigma_table(cache: pd.DataFrame | None = None, clocks=("T", "P"), cutoffs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pooled event-level SD (bp) by clock x {A, B, AB} x {raw, winsorized}; no household identity."""
    cut = winsor_cutoffs() if cutoffs is None else cutoffs
    if cache is None:
        cache = fc.load_cache(columns=["permno", "anchor_idx", "bhar_ff5mom_h20", "reason_h20"])
    calendar = panel.v2_trading_calendar()
    rows = []
    for clock in clocks:
        events, _ = pooled.load_events(clock)
        events["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(events["anchor_date"]))
        j = events.merge(cache, on=["permno", "anchor_idx"], how="left")[["period", "bhar_ff5mom_h20"]].dropna()
        j = j.merge(cut[cut["clock"] == clock][["period", "winsor_lo", "winsor_hi", "sd"]], on="period", how="left")
        j["y_w"] = j["bhar_ff5mom_h20"].clip(lower=j["winsor_lo"], upper=j["winsor_hi"])
        for period, g in list(j.groupby("period")) + [("AB", j)]:
            for variant, col in (("raw", "bhar_ff5mom_h20"), ("winsorized", "y_w")):
                rows.append({"clock": clock, "period": period, "sigma_variant": variant, "n_ok": int(len(g)),
                             "sigma_bp": float(g[col].std(ddof=1) * BP),
                             "csv_sd_bp": float(g["sd"].iloc[0] * BP) if period != "AB" else np.nan})
    return pd.DataFrame(rows)


def ttd_table(sigmas: pd.DataFrame, h_by_clock: dict, d_grid=D_GRID, h_extra=(330,),
              theta_pct=ttd.THETA_ANNUAL_GRID_PCT, n_grid=ttd.N_PER_YEAR_GRID) -> pd.DataFrame:
    """Tidy years-to-detect grid: clock x sigma variant x d x theta x n x {plain, bonferroni_H...}."""
    rows = []
    sig = sigmas[sigmas["period"] == "AB"]
    for r in sig.itertuples():
        adjustments = [("plain", 1, ttd.Z_ALPHA_05)]
        for h in (h_by_clock[r.clock], *h_extra):
            adjustments.append((f"bonferroni_H{h}", h, ttd.bonferroni_z(0.05, h)))
        for d, th, n, (adj, h, z) in itertools.product(d_grid, theta_pct, n_grid, adjustments):
            years = ttd.years_to_detect(r.sigma_bp, th * 100.0, n, d, z_alpha=z)
            rows.append({"clock": r.clock, "sigma_variant": r.sigma_variant, "sigma_bp": r.sigma_bp, "d": d,
                         "theta_annual_pct": th, "theta_20_bp": ttd.theta_20_from_annual(th * 100.0),
                         "n_per_year": n, "adjustment": adj, "H": h, "z_alpha": z, "z_power": ttd.Z_POWER_80,
                         "years": years, "exceeds_observed_span": bool(years > OBSERVED_SPAN_YEARS)})
    return pd.DataFrame(rows)


BIN_EDGES = {"1": (1, 1), "2": (2, 2), "3-5": (3, 5), "6-10": (6, 10), "11-20": (11, 20),
             "21-50": (21, 50), "51-100": (51, 100), "100+": (101, 1000)}


def add_bin_edges(table: pd.DataFrame) -> pd.DataFrame:
    """Numeric (lo, hi) edges for D's activity bins (the open top bin is capped at 1,000 for plotting)."""
    t = table.copy()
    t["lo"] = t["bin"].map(lambda b: BIN_EDGES[b][0])
    t["hi"] = t["bin"].map(lambda b: BIN_EDGES[b][1])
    return t


def activity_histogram(path=DIAG / "household_activity_distribution.csv") -> pd.DataFrame:
    """D's purchases-per-household-year histogram (counts only) with numeric bin edges for plotting."""
    return add_bin_edges(pd.read_csv(path))


def unit_test_value() -> float:
    """Plan §7.4 reference: sigma 900 bp, d = 1, theta 5% annualized, n = 50 -> ~63.6 years."""
    return ttd.years_to_detect(900.0, 500.0, 50, 1.0)
