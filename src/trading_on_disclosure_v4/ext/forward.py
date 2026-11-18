"""Plan section 7 (optional extension 3): forward prediction on the public clock. A household's score at
each year-end Y is the expanding mean of its winsorized FF5+MOM 20-session outcome over de-duplicated
purchases whose anchor and window end fall on or before 31 December Y; households with at least ten
scored purchases are eligible at Y, and their purchases anchored in Y+1 (raw outcome) are regressed on
the score with household-year weights and household x 20-day-block clusters, compared across score
terciles, counted per year, and placed against 2,000 within-year permutations of the scores."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.data import calendar as cal_mod
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.ext import common
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.inference.placebo import placebo_p_value
from trading_on_disclosure_v4.outcomes import event_filters as ef
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel as panel_mod
from trading_on_disclosure_v4.outcomes.seeds import seed_from

CLOCK = "P"
ANALYSIS = "forward_prediction"
YEAR_ENDS = tuple(range(2016, 2025))
MIN_SCORED = 10
N_PERMUTATIONS = 2000
SEED = seed_from("forward_prediction", "permutation")
PERIOD_B_START = pd.Timestamp("2021-01-01")
SE_LABEL = "household x 20-day anchor block (two-way CGM) SE in se_block"
OUT_CSV = common.OUT_DIR / "forward_prediction.csv"
BY_YEAR_CSV = common.OUT_DIR / "forward_prediction_by_year.csv"
RESTRICTED_DIR = common.RESTRICTED_EXT / "forward"
LOG = common.LOG_DIR / "forward.log"
PERM_STATS = ("slope", "slope_demeaned_within_year", "tercile_diff")
SAMPLE_NOTE = ("public clock, de-duplicated purchases; score at Y = mean winsorized outcome over purchases with anchor "
               "and 20-session window end on or before 31 Dec Y (the anchor is the first session after the public date, "
               "so the window-end condition implies the public-date condition); period-A cutoffs for anchors through "
               "2020-12-31, period-B after; eligible at >= 10 scored purchases; evaluation = raw outcome of the eligible "
               "households' purchases anchored in Y+1; Y = 2016..2024")


# ---------------------------------------------------------------- frame -> tables

def prepare(frame: pd.DataFrame, cutoffs: dict[str, tuple[float, float]], y: str = "y_h20") -> pd.DataFrame:
    """De-duplicated purchases with timestamps, the raw outcome `y`, and `y_wins` clipped at the cutoffs of
    the anchor's period (A through 2020-12-31, B after)."""
    f = frame
    if "direction" in f.columns:
        f = f.loc[f["direction"] == "purchase"]
    f = f.loc[~f["dup_all_lines"].astype(bool)].copy()
    f["household_id"] = f["household_id"].astype(str)
    f["anchor_date"] = pd.to_datetime(f["anchor_date"])
    f["window_end_date"] = pd.to_datetime(f["window_end_date"])
    if (f["window_end_date"] < f["anchor_date"]).any():
        raise ValueError("a window ends before its anchor")
    f["anchor_year"] = f["anchor_date"].dt.year.astype(int)
    f["y"] = pd.to_numeric(f[y], errors="coerce").astype(float)
    f["finite"] = np.isfinite(f["y"].to_numpy())
    f["cutoff_period"] = np.where(f["anchor_date"] < PERIOD_B_START, "A", "B")
    f["y_wins"] = sp.winsorize(f.assign(period=f["cutoff_period"]), cutoffs, y="y")["y"].to_numpy()
    cols = ["event_id", "household_id", "anchor_date", "anchor_year", "window_end_date", "y", "y_wins", "finite", "cutoff_period"]
    return f[cols].sort_values(["anchor_date", "household_id", "event_id"]).reset_index(drop=True)


def household_scores(f: pd.DataFrame, year_ends=YEAR_ENDS, min_scored: int = MIN_SCORED, cutoff_rule: str = "period") -> pd.DataFrame:
    """One row per (year_end, household) with at least one scored purchase: n_scored, score, eligible.

    cutoff_rule "period": the committed pooled cutoffs of the anchor's period (y_wins as prepared);
    "historical": at each Y the 1st/99th percentiles of the finite scored outcomes available at Y (addendum A)."""
    if cutoff_rule not in ("period", "historical"):
        raise ValueError(f"unknown cutoff rule {cutoff_rule!r}")
    parts, cuts = [], []
    finite = f["finite"].to_numpy()
    for year in year_ends:
        end = pd.Timestamp(year=year, month=12, day=31)
        m = finite & (f["anchor_date"] <= end).to_numpy() & (f["window_end_date"] <= end).to_numpy()
        sub = f.loc[m, ["household_id", "y", "y_wins"]]
        if cutoff_rule == "historical":
            lo, hi = (float(np.quantile(sub["y"], 0.01)), float(np.quantile(sub["y"], 0.99))) if len(sub) else (np.nan, np.nan)
            sub = sub.assign(y_wins=sub["y"].clip(lo, hi))
            cuts.append({"year_end": int(year), "n_scored_outcomes": int(len(sub)), "cutoff_low_bp": lo, "cutoff_high_bp": hi})
        g = sub.groupby("household_id")["y_wins"].agg(n_scored="size", score="mean").reset_index()
        g.insert(0, "year_end", int(year))
        parts.append(g)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["year_end", "household_id", "n_scored", "score"])
    out["n_scored"] = out["n_scored"].astype(int)
    out["eligible"] = out["n_scored"] >= min_scored
    out.attrs["cutoffs_by_year"] = cuts
    return out


def tercile_labels(scores: np.ndarray, order: np.ndarray) -> np.ndarray:
    """bottom / middle / top by rank of the score (ties by `order`); top and bottom hold floor(n/3) each."""
    n = len(scores)
    k = n // 3
    rank = np.empty(n, dtype=int)
    rank[np.lexsort((np.asarray(order), np.asarray(scores, dtype=float)))] = np.arange(n)
    lab = np.full(n, "middle", dtype=object)
    lab[rank < k] = "bottom"
    lab[rank >= n - k] = "top"
    return lab


def evaluation_tables(f: pd.DataFrame, year_ends=YEAR_ENDS, min_scored: int = MIN_SCORED, cutoff_rule: str = "period") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eligible household-years (score, tercile, next-year purchase counts) and the evaluation rows: the
    eligible households' purchases anchored in Y+1 with a finite raw outcome, weight 1/n_{h,Y+1}."""
    scores = household_scores(f, year_ends, min_scored, cutoff_rule)
    hy_parts, ev_parts = [], []
    for year in year_ends:
        h = scores.loc[(scores["year_end"] == year) & scores["eligible"]].sort_values("household_id").reset_index(drop=True)
        h["tercile"] = tercile_labels(h["score"].to_numpy(), np.arange(len(h)))
        nxt = f.loc[f["anchor_year"] == year + 1]
        n_any = nxt.groupby("household_id").size()
        n_fin = nxt.loc[nxt["finite"]].groupby("household_id").size()
        h["n_next"] = n_any.reindex(h["household_id"]).fillna(0).astype(int).to_numpy()
        h["n_next_finite"] = n_fin.reindex(h["household_id"]).fillna(0).astype(int).to_numpy()
        h["in_evaluation"] = h["n_next_finite"] > 0
        hy_parts.append(h)
        ev = nxt.loc[nxt["finite"] & nxt["household_id"].isin(h["household_id"]), ["event_id", "household_id", "anchor_date", "y"]]
        ev = ev.merge(h[["year_end", "household_id", "score", "tercile", "n_next_finite"]], on="household_id", how="left")
        ev["w"] = 1.0 / ev["n_next_finite"].astype(float)
        ev_parts.append(ev)
    hy_cols = ["year_end", "household_id", "n_scored", "score", "tercile", "n_next", "n_next_finite", "in_evaluation"]
    ev_cols = ["year_end", "household_id", "event_id", "anchor_date", "y", "score", "tercile", "n_next_finite", "w"]
    hy = pd.concat(hy_parts, ignore_index=True)[hy_cols] if hy_parts else pd.DataFrame(columns=hy_cols)
    ev = pd.concat(ev_parts, ignore_index=True)[ev_cols] if ev_parts else pd.DataFrame(columns=ev_cols)
    ev = ev.sort_values(["year_end", "household_id", "anchor_date", "event_id"]).reset_index(drop=True)
    hy.attrs["cutoffs_by_year"] = list(scores.attrs.get("cutoffs_by_year", []))
    return hy, ev


# ---------------------------------------------------------------- estimators

def point_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    den = float(np.sum(w * (x - xm) ** 2))
    return float(np.sum(w * (x - xm) * (y - ym)) / den) if den > 0 else np.nan


def point_tercile_diff(y: np.ndarray, w: np.ndarray, tercile: np.ndarray) -> float:
    top, bottom = tercile == "top", tercile == "bottom"
    if not top.any() or not bottom.any():
        return np.nan
    return float(np.average(y[top], weights=w[top]) - np.average(y[bottom], weights=w[bottom]))


def demean_within_year(x: np.ndarray, w: np.ndarray, year: np.ndarray) -> np.ndarray:
    """x minus its w-weighted mean over the evaluation rows of the same year-end."""
    out = np.asarray(x, dtype=float).copy()
    for yr in np.unique(year):
        m = year == yr
        out[m] -= np.average(out[m], weights=w[m])
    return out


def _clusters(ev: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"household_id": ev["household_id"].to_numpy(), "anchor_blk": ev["anchor_blk"].to_numpy()})


def clustered_slope(ev: pd.DataFrame, x: np.ndarray) -> cgm.ClusteredSlope | None:
    """WLS of the raw outcome on [1, x] with the household-year weights and the two-way block clusters."""
    if len(ev) < 3 or len(np.unique(x)) < 2:
        return None
    X = np.column_stack([np.ones(len(ev)), np.asarray(x, dtype=float)])
    return cgm.clustered_slope(ev["y"].to_numpy(), X, ev["w"].to_numpy(), _clusters(ev), coef_index=1)


def clustered_tercile_diff(ev: pd.DataFrame) -> cgm.ClusteredSlope | None:
    """Top minus bottom tercile: WLS of the outcome on [1, top] over the two terciles' rows (the same model as
    a top dummy plus a bottom dummy; the coefficient on top is their difference)."""
    sub = ev.loc[ev["tercile"].isin(["top", "bottom"])]
    top = (sub["tercile"].to_numpy() == "top").astype(float)
    if len(sub) < 3 or top.sum() == 0 or top.sum() == len(sub):
        return None
    X = np.column_stack([np.ones(len(sub)), top])
    return cgm.clustered_slope(sub["y"].to_numpy(), X, sub["w"].to_numpy(), _clusters(sub), coef_index=1)


def permutation_reference(hy: pd.DataFrame, ev: pd.DataFrame, n_draws: int = N_PERMUTATIONS, seed: int = SEED) -> pd.DataFrame:
    """Scores permuted across the eligible households within each year-end (every year-end once per draw);
    terciles re-formed on the permuted scores; one row of point estimates per draw."""
    rng = np.random.default_rng(seed)
    y, w, yr = ev["y"].to_numpy(dtype=float), ev["w"].to_numpy(dtype=float), ev["year_end"].to_numpy()
    plan = []
    for year in sorted(hy["year_end"].unique()):
        h = hy.loc[hy["year_end"] == year].sort_values("household_id")
        rows = np.flatnonzero(yr == year)
        idx = pd.Index(h["household_id"]).get_indexer(ev["household_id"].to_numpy()[rows])
        if (idx < 0).any():
            raise ValueError("evaluation row of a household that is not eligible in its year")
        plan.append((h["score"].to_numpy(dtype=float), rows, idx))
    out = np.full((n_draws, len(PERM_STATS)), np.nan)
    x = np.empty(len(ev))
    lab = np.empty(len(ev), dtype=object)
    for d in range(n_draws):
        for scores, rows, idx in plan:
            s = scores[rng.permutation(len(scores))]
            x[rows] = s[idx]
            lab[rows] = tercile_labels(s, np.arange(len(s)))[idx]
        out[d] = (point_slope(x, y, w), point_slope(demean_within_year(x, w, yr), y, w), point_tercile_diff(y, w, lab))
    return pd.DataFrame(out, columns=list(PERM_STATS))


# ---------------------------------------------------------------- rows

def per_year_table(hy: pd.DataFrame, ev: pd.DataFrame, year_ends=YEAR_ENDS) -> pd.DataFrame:
    """One row per year-end: eligibility and evaluation counts, tercile sizes, and the year's own estimates."""
    rows = []
    for year in year_ends:
        h = hy.loc[hy["year_end"] == year]
        e = ev.loc[ev["year_end"] == year]
        n_el = int(len(h))
        n_none = int((h["n_next"] == 0).sum())
        sl = clustered_slope(e, e["score"].to_numpy()) if len(e) else None
        td = clustered_tercile_diff(e) if len(e) else None
        rows.append({
            "year_end": int(year), "evaluation_year": int(year) + 1, "n_eligible": n_el, "n_no_purchase": n_none,
            "share_no_purchase": n_none / n_el if n_el else np.nan,
            "n_no_finite_outcome": int(((h["n_next"] > 0) & ~h["in_evaluation"]).sum()),
            "n_in_evaluation": int(h["in_evaluation"].sum()), "n_evaluation_purchases": int(len(e)),
            "n_bottom": int((h["tercile"] == "bottom").sum()), "n_middle": int((h["tercile"] == "middle").sum()),
            "n_top": int((h["tercile"] == "top").sum()),
            "n_bottom_in_evaluation": int(((h["tercile"] == "bottom") & h["in_evaluation"]).sum()),
            "n_top_in_evaluation": int(((h["tercile"] == "top") & h["in_evaluation"]).sum()),
            "n_bottom_purchases": int((e["tercile"] == "bottom").sum()), "n_top_purchases": int((e["tercile"] == "top").sum()),
            "score_mean_eligible": float(h["score"].mean()) if n_el else np.nan,
            "score_sd_eligible": float(h["score"].std(ddof=1)) if n_el > 1 else np.nan,
            "slope": sl.coef if sl else np.nan, "slope_se_block": sl.se if sl else np.nan, "slope_dof_block": sl.dof if sl else np.nan,
            "tercile_diff": td.coef if td else np.nan, "tercile_diff_se_block": td.se if td else np.nan,
            "tercile_diff_dof_block": td.dof if td else np.nan, "se_type": SE_LABEL,
        })
    return pd.DataFrame(rows)


def _stat_rows(variant: str, stat: str, fit: cgm.ClusteredSlope | None, unit: str, n_ev: int, n_hh: int, notes: str,
               perm: np.ndarray | None = None) -> list[dict]:
    """The estimate (block SE, permutation p when `perm` is given) with its t, p and dof rows."""
    if fit is None:
        return [rs.row(CLOCK, ANALYSIS, variant, stat, np.nan, unit, n_events=n_ev, n_households=n_hh, notes=f"not estimable; {notes}")]
    p_perm, note = np.nan, notes
    if perm is not None and np.isfinite(fit.coef) and np.isfinite(perm).any():
        pl = placebo_p_value(fit.coef, perm, "two-sided")
        p_perm = pl["p"]
        note = f"{notes}; permutation p two-sided about the permutation median, {pl['n_null']} draws"
    return [rs.row(CLOCK, ANALYSIS, variant, stat, fit.coef, unit, placebo_p=p_perm, se_block=fit.se, n_events=n_ev, n_households=n_hh, notes=note),
            rs.row(CLOCK, ANALYSIS, variant, f"{stat}_t_block", fit.t_stat, "t", n_events=n_ev, n_households=n_hh),
            rs.row(CLOCK, ANALYSIS, variant, f"{stat}_p_block", fit.p, "p", n_events=n_ev, n_households=n_hh, notes="two-sided, t with dof = min cluster count - 1"),
            rs.row(CLOCK, ANALYSIS, variant, f"{stat}_dof_block", fit.dof, "count", n_events=n_ev, n_households=n_hh)]


def result_rows(hy: pd.DataFrame, ev: pd.DataFrame, by_year: pd.DataFrame, perms: pd.DataFrame, seed: int) -> list[dict]:
    """Tidy rows: pooled estimates with permutation p, the permutation summaries, and every per-year row."""
    n_ev, n_hh = int(len(ev)), int(ev["household_id"].nunique())
    rows: list[dict] = []
    v = "pooled"
    sl = clustered_slope(ev, ev["score"].to_numpy())
    rows += _stat_rows(v, "slope", sl, "slope", n_ev, n_hh, f"{SAMPLE_NOTE}; next-year raw outcome on the score, weights 1/n_(h,Y+1); {SE_LABEL}",
                       perms["slope"].to_numpy())
    xd = demean_within_year(ev["score"].to_numpy(dtype=float), ev["w"].to_numpy(dtype=float), ev["year_end"].to_numpy())
    sd = clustered_slope(ev, xd)
    rows += _stat_rows(v, "slope_demeaned_within_year", sd, "slope", n_ev, n_hh,
                       f"score demeaned within year, descriptive (weighted mean over the year's evaluation household-years); {SE_LABEL}",
                       perms["slope_demeaned_within_year"].to_numpy())
    td = clustered_tercile_diff(ev)
    rows += _stat_rows(v, "tercile_diff", td, "bp", n_ev, n_hh,
                       f"top minus bottom score tercile (terciles of floor(n/3) households within each Y, ranks with ties by score then household order), "
                       f"household-year weighted; {SE_LABEL}", perms["tercile_diff"].to_numpy())
    for lab in ("top", "bottom"):
        sub = ev.loc[ev["tercile"] == lab]
        val = float(np.average(sub["y"], weights=sub["w"])) if len(sub) else np.nan
        rows.append(rs.row(CLOCK, ANALYSIS, v, f"{lab}_tercile_mean", val, "bp", n_events=int(len(sub)), n_households=int(sub["household_id"].nunique()),
                           notes="household-year weighted mean of the next-year raw outcome"))
    if sl is not None:
        rows.append(rs.row(CLOCK, ANALYSIS, v, "n_clusters_household", sl.cluster_counts.get("household_id", np.nan), "count", n_events=n_ev, n_households=n_hh))
        rows.append(rs.row(CLOCK, ANALYSIS, v, "n_clusters_block", sl.cluster_counts.get("anchor_blk", np.nan), "count", n_events=n_ev, n_households=n_hh))
    counts = (("n_household_years", int(hy["in_evaluation"].sum()), "eligible household-years with >= 1 evaluation purchase"),
              ("n_eligible_household_years", int(len(hy)), "eligible household-years over Y = 2016..2024"),
              ("n_no_purchase_household_years", int((hy["n_next"] == 0).sum()), "eligible household-years with no purchase in Y+1"),
              ("n_evaluation_purchases", n_ev, "purchases anchored in Y+1 with a finite raw outcome"),
              ("n_households_distinct", n_hh, "distinct households in the evaluation"))
    for stat, val, note in counts:
        rows.append(rs.row(CLOCK, ANALYSIS, v, stat, val, "count", n_events=n_ev, n_households=n_hh, notes=note))
    rows.append(rs.row(CLOCK, ANALYSIS, v, "share_no_purchase_household_years", (hy["n_next"] == 0).mean() if len(hy) else np.nan, "share",
                       n_events=n_ev, n_households=n_hh))
    rows.append(rs.row(CLOCK, ANALYSIS, v, "score_sd_eligible", float(hy["score"].std(ddof=1)) if len(hy) > 1 else np.nan, "bp",
                       n_events=n_ev, n_households=n_hh, notes="SD of the score over eligible household-years, pooled"))
    v = "permutation"
    for stat in PERM_STATS:
        unit = "bp" if stat == "tercile_diff" else "slope"
        vals = perms[stat].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        for suffix, val in (("mean", vals.mean()), ("sd", vals.std(ddof=1)), ("q025", np.quantile(vals, 0.025)), ("q975", np.quantile(vals, 0.975))):
            rows.append(rs.row(CLOCK, ANALYSIS, v, f"{stat}_{suffix}", float(val), unit, notes="scores permuted across the eligible households within each Y"))
    rows.append(rs.row(CLOCK, ANALYSIS, v, "n_draws", int(len(perms)), "count"))
    rows.append(rs.row(CLOCK, ANALYSIS, v, "seed", int(seed), "seed", notes="seed_from('forward_prediction', 'permutation')"))
    for r in by_year.itertuples(index=False):
        v = f"year_{r.year_end}"
        n_e, n_h = int(r.n_evaluation_purchases), int(r.n_in_evaluation)
        rows.append(rs.row(CLOCK, ANALYSIS, v, "slope", r.slope, "slope", se_block=r.slope_se_block, n_events=n_e, n_households=n_h,
                           notes=f"this year-end alone; {SE_LABEL}; dof {r.slope_dof_block}"))
        rows.append(rs.row(CLOCK, ANALYSIS, v, "tercile_diff", r.tercile_diff, "bp", se_block=r.tercile_diff_se_block, n_events=int(r.n_top_purchases + r.n_bottom_purchases),
                           n_households=int(r.n_top_in_evaluation + r.n_bottom_in_evaluation), notes=f"this year-end alone; {SE_LABEL}; dof {r.tercile_diff_dof_block}"))
        for stat in ("n_eligible", "n_no_purchase", "n_no_finite_outcome", "n_in_evaluation", "n_evaluation_purchases", "n_bottom", "n_middle", "n_top",
                     "n_bottom_in_evaluation", "n_top_in_evaluation"):
            rows.append(rs.row(CLOCK, ANALYSIS, v, stat, int(getattr(r, stat)), "count", n_events=n_e, n_households=n_h))
        rows.append(rs.row(CLOCK, ANALYSIS, v, "share_no_purchase", r.share_no_purchase, "share", n_events=n_e, n_households=n_h,
                           notes="eligible households with no de-duplicated purchase anchored in Y+1"))
    return rows


# ---------------------------------------------------------------- run

def run(frame: pd.DataFrame, calendar: pd.DatetimeIndex, cutoffs: dict[str, tuple[float, float]], year_ends=YEAR_ENDS,
        n_draws: int = N_PERMUTATIONS, seed: int = SEED, log=None, cutoff_rule: str = "period") -> dict:
    """Everything from an event frame: tables, permutations, tidy rows."""
    f = prepare(frame, cutoffs)
    hy, ev = evaluation_tables(f, year_ends, cutoff_rule=cutoff_rule)
    ev = rs.block_cluster(ev, calendar)
    common.log_line(log, f"{len(f)} de-duplicated purchases, {len(hy)} eligible household-years, {len(ev)} evaluation purchases")
    t0 = time.time()
    perms = permutation_reference(hy, ev, n_draws, seed)
    common.log_line(log, f"{n_draws} permutations in {time.time() - t0:.1f}s")
    by_year = per_year_table(hy, ev, year_ends)
    rows = result_rows(hy, ev, by_year, perms, seed)
    if cutoff_rule == "historical":
        for r in rows:
            r["notes"] = (r["notes"] + "; " if r["notes"] else "") + "score winsorized at the 1st/99th percentiles of the outcomes available at Y (addendum A)"
    return {"purchases": f, "household_years": hy, "evaluation": ev, "permutations": perms, "by_year": by_year, "rows": rows,
            "cutoffs_by_year": pd.DataFrame(hy.attrs.get("cutoffs_by_year", []))}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="plan section 7: forward prediction on the public clock")
    ap.add_argument("--draws", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--cutoffs", choices=["period", "historical"], default="period",
                    help="period: the committed pooled cutoffs (plan section 7); historical: percentiles of the outcomes available at Y (addendum A)")
    args = ap.parse_args(argv)
    suffix = "" if args.cutoffs == "period" else "_historical"
    out_csv, by_year_csv = OUT_CSV.with_name(f"forward_prediction{suffix}.csv"), BY_YEAR_CSV.with_name(f"forward_prediction{suffix}_by_year.csv")
    common.log_line(LOG, f"start forward prediction ({args.draws} permutations, seed {SEED}, cutoffs {args.cutoffs})")
    calendar = panel_mod.v2_trading_calendar()
    frame = evf.load_event_frame(CLOCK, calendar=calendar, directions=("purchase",), periods=("A", "B", "X"))
    cutoffs = evf.pooled_cutoffs(CLOCK)
    res = run(frame, calendar, cutoffs, n_draws=args.draws, log=LOG, cutoff_rule=args.cutoffs)
    RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    res["household_years"].to_parquet(RESTRICTED_DIR / f"household_year_scores{suffix}.parquet", index=False)
    res["evaluation"].to_parquet(RESTRICTED_DIR / f"evaluation_rows{suffix}.parquet", index=False)
    res["permutations"].to_parquet(RESTRICTED_DIR / f"permutation_stats{suffix}.parquet", index=False)
    common.write_rows(res["rows"], out_csv)
    res["by_year"].to_csv(by_year_csv, index=False, lineterminator="\n")
    if args.cutoffs == "historical":
        res["cutoffs_by_year"].to_csv(out_csv.with_name("forward_prediction_historical_cutoffs.csv"), index=False, lineterminator="\n")
    OUT, BY = out_csv, by_year_csv
    inputs = [common.file_entry(ef.EVENTS_DIR / f"events_{CLOCK}.parquet", rows=len(frame)), common.file_entry(ef.FLAGS_PATH),
              common.file_entry(fc.CACHE_PATH), common.file_entry(evf.POOLED_CSV), common.file_entry(cal_mod.benchmark_path(), rows=len(calendar))]
    common.write_manifest("forward" + suffix, inputs, extra={"outputs": [str(OUT.relative_to(common.REPO_ROOT)), str(BY.relative_to(common.REPO_ROOT))],
                                                             "n_draws": int(args.draws), "seed": int(SEED), "year_ends": list(YEAR_ENDS), "min_scored": MIN_SCORED,
                                                             "cutoff_rule": args.cutoffs})
    common.log_line(LOG, f"wrote {OUT.relative_to(common.REPO_ROOT)} ({len(res['rows'])} rows) and {BY.relative_to(common.REPO_ROOT)}")
    print(res["by_year"].to_string(index=False))
    return res


if __name__ == "__main__":
    main()
