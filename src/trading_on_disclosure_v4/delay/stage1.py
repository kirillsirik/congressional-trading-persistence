"""Filing-delay regression, stage 1: does the filing delay of a purchase predict its post-transaction abnormal return?

Exploratory, run before the plan commit. Sample: transaction-clock
purchase episodes (sales secondary) anchored in period A (2015-2020 by the plan's window rule) whose
official public date is OBSERVED and whose lag (public date - transaction date, calendar days, the
episode's earliest-disclosed line) lies in [1, 365]. Outcome: the 20- (and 10-) session FF5+MOM
buy-and-hold abnormal return read from N's forward cache on (permno, anchor_idx), in basis points.

Regressors: log(lag); 1{lag > 45}; lag bins (<=15 base, 16-30, 31-45, 46-90, >90); the log lag split
into its within-report and between-report parts (the mechanical within-report gradient: lines of one
report share a public date, so older trades have longer lags). Fixed effects are absorbed by
an exact within-projection (Frisch-Waugh) and are NEVER materialized, stored or returned; only the slope
coefficients, their two-way (household x anchor date) CGM standard errors, t, p and counts leave this
module. Key cut: lag >= 30, so the 20-session outcome window ends before disclosure.

Trigger (plan §7.6): |t| >= 2.5 on log(lag) in the household-FE purchase specification (h20, raw Y)
raises checkpoint 6; otherwise stage 2 is not run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from trading_on_disclosure_v4.data.events import _event_id
from trading_on_disclosure_v4.inference.cgm import wls_cluster_cov

REPO_ROOT = Path(__file__).resolve().parents[3]
EVENTS_T = REPO_ROOT / "data" / "v4" / "restricted" / "data" / "events_T.parquet"
CACHE_PATH = REPO_ROOT / "data" / "v4" / "restricted" / "n" / "forward_cache.parquet"
POOLED_PATH = REPO_ROOT / "data" / "v4" / "pooled" / "pooled_outcome_distribution.csv"
OUT_DIR = REPO_ROOT / "data" / "v4" / "delay"
RESTRICTED_DIR = REPO_ROOT / "data" / "v4" / "restricted" / "delay"
RESULTS_PATH = OUT_DIR / "stage1_results.csv"

LAG_RANGE = (1.0, 365.0)
KEY_CUT = 30.0
GT_CUT = 45.0
BIN_EDGES = (15.0, 30.0, 45.0, 90.0)
BIN_LABELS = ("le15", "16_30", "31_45", "46_90", "gt90")
HORIZONS = {"h20": ("bhar_ff5mom_h20", "reason_h20"), "h10": ("bhar_ff5mom_h10", "reason_h10")}
BP = 1e4
TRIGGER_ABS_T = 2.5
TRIGGER_SPEC = ("S1_loglag_hhfe_ymfe", "purchase", "h20", "raw", "log_lag")
CLUSTERS = ("household_id", "anchor_date")
OFFICIAL_OBSERVED = "OFFICIAL_OBSERVED"

RESULT_COLUMNS = [
    "spec_id", "direction", "horizon", "outcome", "sample", "fixed_effects", "regressor",
    "coef_bp", "se_bp", "t_stat", "p_two_sided", "ci_low_bp", "ci_high_bp", "dof",
    "n_events", "n_households", "n_dates", "n_yearmonths", "n_reports", "psd_corrected",
    "n_dropped_missing_outcome", "lag_min", "lag_max", "is_trigger_spec",
]
FORBIDDEN_OUTPUT_COLUMNS = ("household_id", "event_id", "document_uid", "permno", "transaction_uids")


# ---------------------------------------------------------------- sample selection (counts only)
def select_sample(
    events: pd.DataFrame, direction: str = "purchase", period: str = "A",
    lag_range: tuple[float, float] = LAG_RANGE,
) -> tuple[pd.DataFrame, dict]:
    """Period/direction/lag filter on D's T-clock event table; returns (rows, count report)."""
    if not events["clock"].eq("T").all():
        raise ValueError("A8 stage 1 runs on the transaction clock only")
    lo, hi = lag_range
    d = events["direction"].eq(direction)
    p = events["period"].eq(period)
    lag = events["lag_days"]
    observed = lag.notna()
    in_range = lag.between(lo, hi)
    keep = d & p & observed & in_range
    report = {
        "direction": direction, "period": period, "lag_range": [lo, hi],
        "n_events_T": int(len(events)), "n_direction": int(d.sum()), "n_direction_period": int((d & p).sum()),
        "n_lag_observed": int((d & p & observed).sum()),
        "n_lag_unobserved": int((d & p & ~observed).sum()),
        "n_lag_negative": int((d & p & observed & (lag < 0)).sum()),
        "n_lag_zero": int((d & p & observed & (lag == 0)).sum()),
        "n_lag_over_max": int((d & p & observed & (lag > hi)).sum()),
        "n_selected": int(keep.sum()),
        "n_selected_lag_ge_key_cut": int((keep & (lag >= KEY_CUT)).sum()),
        "n_households": int(events.loc[keep, "household_id"].nunique()),
        "n_anchor_dates": int(events.loc[keep, "anchor_date"].nunique()),
    }
    cols = ["event_id", "household_id", "permno", "anchor_date", "window_end_date", "direction",
            "anchor_year", "chamber", "record_source", "lag_days", "n_lines", "transaction_uids"]
    return events.loc[keep, cols].reset_index(drop=True), report


def _t_event_id(lines: pd.DataFrame) -> pd.Series:
    """Recompute D's T event_id from timed lines (same key, same hash: data.events._event_id)."""
    parts = pd.DataFrame({
        "household_id": lines["filer_uid"].astype(str),
        "permno": lines["permno"].astype("int64"),
        "trade_date_analysis": pd.to_datetime(lines["trade_date_analysis"]),
        "direction": lines["direction"].astype(str),
    })
    return _event_id("T:", parts)


# ---------------------------------------------------------------- report linkage (batch-filing caveat)
def observed_lines(timed: pd.DataFrame) -> pd.DataFrame:
    """Directional, PERMNO-matched timed lines with an OBSERVED official public date, plus their lag."""
    keep = timed["direction"].isin(["purchase", "sale"]) & timed["permno"].notna()
    keep &= timed["analysis_public_date_source"].eq(OFFICIAL_OBSERVED)
    lines = timed.loc[keep].copy()
    lines["permno"] = lines["permno"].astype("int64")
    lines["trade_date_analysis"] = pd.to_datetime(lines["trade_date_analysis"])
    lines["analysis_public_date"] = pd.to_datetime(lines["analysis_public_date"])
    lines["line_lag"] = (lines["analysis_public_date"] - lines["trade_date_analysis"]).dt.days.astype(float)
    lines["event_id"] = _t_event_id(lines)
    lines["document_uid"] = lines["document_uid"].astype(str)
    lines["document_format"] = lines["document_format"].astype(str)
    return lines.reset_index(drop=True)


def attach_report(sample: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Add the report (document) of the episode's earliest-disclosed line and report-level counts.

    ``report_n_lines`` counts every observed directional line of that report (all directions and
    dates: the batch-filing measure); ``report_lag_range`` is its max minus min line lag.
    """
    per_report = lines.groupby("document_uid").agg(
        report_n_lines=("line_lag", "size"), report_lag_min=("line_lag", "min"),
        report_lag_max=("line_lag", "max"), report_n_trade_dates=("trade_date_analysis", "nunique"),
    )
    per_report["report_lag_range"] = per_report["report_lag_max"] - per_report["report_lag_min"]
    first = (lines.sort_values(["event_id", "line_lag", "document_uid"])
             .drop_duplicates("event_id", keep="first")
             .set_index("event_id")[["document_uid", "document_format", "line_lag", "analysis_public_date"]])
    out = sample.merge(first, left_on="event_id", right_index=True, how="left", validate="1:1")
    if out["document_uid"].isna().any():
        raise ValueError("a selected episode has no observed official line")
    if not np.allclose(out["line_lag"], out["lag_days"]):
        raise ValueError("earliest-line lag disagrees with D's lag_days")
    out = out.drop(columns="line_lag").rename(columns={"analysis_public_date": "public_date"})
    return out.merge(per_report, left_on="document_uid", right_index=True, how="left")


# ---------------------------------------------------------------- outcomes
def attach_outcomes(sample: pd.DataFrame, cache: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Left-join the forward cache on (permno, anchor_idx); y_<h> in bp, NaN unless reason == ok."""
    out = sample.copy()
    out["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(out["anchor_date"]))
    if (out["anchor_idx"] < 0).any():
        raise ValueError("anchor_date off the trading calendar")
    cols = ["permno", "anchor_idx"] + [c for pair in HORIZONS.values() for c in pair]
    out = out.merge(cache[cols], on=["permno", "anchor_idx"], how="left", validate="m:1")
    for h, (ycol, rcol) in HORIZONS.items():
        out[rcol] = out[rcol].fillna("not_in_cache")
        out[f"y_{h}"] = np.where(out[rcol].eq("ok"), out[ycol] * BP, np.nan)
    return out.drop(columns=[c for pair in HORIZONS.values() for c in pair if c.startswith("bhar")])


def winsor_cutoffs(direction: str, horizon: str, pooled_path: Path = POOLED_PATH, sample: pd.DataFrame | None = None) -> tuple[float, float, str]:
    """(lo, hi, source) in bp. Purchases: the wave-1 pooled 1/99 cutoffs (T, A, ff5mom_<h>).

    Sales have no pooled row, so their cutoffs are the pooled 1st/99th percentiles of the sale
    sample itself (no household identity; allowed pre-tag) and the source says so.
    """
    if direction == "purchase":
        pooled = pd.read_csv(pooled_path)
        row = pooled[(pooled["clock"] == "T") & (pooled["period"] == "A") & (pooled["direction"] == "purchase")
                     & (pooled["horizon"] == f"ff5mom_{horizon}")]
        if len(row) != 1:
            raise ValueError(f"pooled cutoff row for {horizon} not unique")
        return float(row["winsor_lo"].iloc[0]) * BP, float(row["winsor_hi"].iloc[0]) * BP, pooled_path.name
    if sample is None:
        raise ValueError("sale cutoffs need the sample")
    y = sample[f"y_{horizon}"].dropna()
    return float(np.percentile(y, 1)), float(np.percentile(y, 99)), "sale_sample_pooled_p1_p99"


def winsorize(y: pd.Series, lo: float, hi: float) -> pd.Series:
    return y.clip(lower=lo, upper=hi)


# ---------------------------------------------------------------- regressors
def lag_bin(lag: pd.Series, edges: tuple[float, ...] = BIN_EDGES, labels: tuple[str, ...] = BIN_LABELS) -> pd.Series:
    """Categorical lag bin: <=15, 16-30, 31-45, 46-90, >90 (integer-day boundaries)."""
    return pd.cut(lag, bins=[-np.inf, *edges, np.inf], labels=list(labels), right=True).astype(str)


def add_regressors(df: pd.DataFrame) -> pd.DataFrame:
    """log(lag), 1{lag > 45}, bin dummies, transaction year-month, within/between-report log lag."""
    out = df.copy()
    lag = out["lag_days"].astype(float)
    if (lag < 1).any():
        raise ValueError("lag must be >= 1 before taking logs")
    out["log_lag"] = np.log(lag)
    out["lag_gt45"] = (lag > GT_CUT).astype(float)
    out["lag_bin"] = lag_bin(lag)
    for label in BIN_LABELS[1:]:
        out[f"bin_{label}"] = out["lag_bin"].eq(label).astype(float)
    out["year_month"] = pd.DatetimeIndex(out["anchor_date"]).strftime("%Y-%m")
    if "document_uid" in out.columns:
        mean_by_report = out.groupby("document_uid")["log_lag"].transform("mean")
        out["log_lag_between"] = mean_by_report
        out["log_lag_within"] = out["log_lag"] - mean_by_report
    return out


# ---------------------------------------------------------------- fixed effects (absorbed, never returned)
def _group_demean(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    n_g = np.bincount(g).astype(float)
    sums = np.zeros((len(n_g), X.shape[1]))
    np.add.at(sums, g, X)
    return X - (sums / n_g[:, None])[g]


def demean_by_groups(M: np.ndarray, groups: list[np.ndarray], rcond: float = 1e-10) -> np.ndarray:
    """Project the columns of ``M`` off one or more sets of group indicators, EXACTLY (Frisch-Waugh).

    ``groups`` are integer code arrays, the first (largest) dimension first. That dimension is
    demeaned in closed form; the remaining dimensions' dummy columns are demeaned the same way and
    the residual of a least-squares fit on them is returned (a rank-deficient dummy block is handled
    by ``lstsq``). The fitted group effects are local temporaries and are never returned.
    """
    X = np.array(M, dtype=float, copy=True)
    if X.ndim == 1:
        X = X[:, None]
    codes = [np.asarray(g, dtype=np.int64) for g in groups]
    X = _group_demean(X, codes[0])
    if len(codes) == 1:
        return X
    dummies = np.concatenate([np.eye(int(g.max()) + 1)[g] for g in codes[1:]], axis=1)
    dummies = _group_demean(dummies, codes[0])
    coef = np.linalg.lstsq(dummies, X, rcond=rcond)[0]
    return X - dummies @ coef


def _codes(values: pd.Series) -> np.ndarray:
    return pd.factorize(values, sort=True)[0].astype(np.int64)


@dataclass
class SpecResult:
    spec_id: str
    coefficients: pd.DataFrame
    counts: dict = field(default_factory=dict)


def fit_fe_regression(
    df: pd.DataFrame, y: str, regressors: list[str], fe_cols: list[str],
    clusters: tuple[str, str] = CLUSTERS, level: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    """OLS of ``y`` on ``regressors`` with ``fe_cols`` absorbed; two-way CGM SEs (v2 cgm3 via E's wrapper).

    Rows with a non-finite outcome are dropped first (counted). Returns one row per regressor
    (coefficient in the units of ``y``) and a count dict. No per-group quantity is returned.
    """
    finite = np.isfinite(df[y].to_numpy(dtype=float))
    d = df.loc[finite].reset_index(drop=True)
    groups = [_codes(d[c]) for c in fe_cols]
    Y = demean_by_groups(d[y].to_numpy(dtype=float), groups)[:, 0]
    X = demean_by_groups(d[regressors].to_numpy(dtype=float), groups)
    cl = pd.DataFrame({c: d[c].to_numpy() for c in clusters})
    res = wls_cluster_cov(Y, X, np.ones(len(d)), cl)
    dof = max(int(res.dof), 1)
    crit = float(stats.t.ppf(0.5 + level / 2, df=dof))
    se = np.sqrt(np.diag(res.cov))
    tstat = np.where(se > 0, res.beta / np.where(se > 0, se, 1.0), np.nan)
    table = pd.DataFrame({
        "regressor": regressors, "coef": res.beta, "se": se, "t_stat": tstat,
        "p_two_sided": 2 * stats.t.sf(np.abs(tstat), df=dof),
        "ci_low": res.beta - crit * se, "ci_high": res.beta + crit * se,
    })
    counts = {
        "dof": dof, "n_events": int(len(d)), "psd_corrected": bool(res.psd_corrected),
        "n_dropped_missing_outcome": int((~finite).sum()),
        "n_households": int(d["household_id"].nunique()), "n_dates": int(d["anchor_date"].nunique()),
        "n_yearmonths": int(d["year_month"].nunique()) if "year_month" in d else 0,
        "n_reports": int(d["document_uid"].nunique()) if "document_uid" in d else 0,
        "lag_min": float(d["lag_days"].min()), "lag_max": float(d["lag_days"].max()),
    }
    return table, counts


# ---------------------------------------------------------------- specifications
HH_YM = ["household_id", "year_month"]
BINS = [f"bin_{b}" for b in BIN_LABELS[1:]]
#: spec_id -> (regressors, fixed effects, sample restriction)
SPECS: dict[str, tuple[list[str], list[str], str]] = {
    "S1_loglag_hhfe_ymfe": (["log_lag"], HH_YM, "all"),
    "S2_gt45_hhfe_ymfe": (["lag_gt45"], HH_YM, "all"),
    "S3_bins_hhfe_ymfe": (BINS, HH_YM, "all"),
    "S4_loglag_ymfe_only": (["log_lag"], ["year_month"], "all"),
    "S5_loglag_hhfe_ymfe_lag_ge30": (["log_lag"], HH_YM, "lag_ge30"),
    "S6_gt45_hhfe_ymfe_lag_ge30": (["lag_gt45"], HH_YM, "lag_ge30"),
    "S7_loglag_within_between_hhfe_ymfe": (["log_lag_within", "log_lag_between"], HH_YM, "all"),
    "S8_loglag_reportfe_ymfe": (["log_lag"], ["document_uid", "year_month"], "all"),
    "S9_loglag_hhfe_ymfe_window_pre_disclosure": (["log_lag"], HH_YM, "window_pre_disclosure"),
}


def restrict(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    if sample == "all":
        return df
    if sample == "lag_ge30":
        return df[df["lag_days"] >= KEY_CUT]
    if sample == "window_pre_disclosure":  # exact version of the key cut: 20th session before the public date
        return df[df["window_end_date"] < df["public_date"]]
    raise ValueError(f"unknown sample restriction {sample!r}")


def run_specs(
    df: pd.DataFrame, direction: str, horizons: tuple[str, ...] = ("h20", "h10"),
    outcomes: tuple[str, ...] = ("raw", "winsor"), cutoffs: dict[str, tuple[float, float]] | None = None,
    specs: dict | None = None,
) -> pd.DataFrame:
    """Every spec x horizon x outcome variant for one direction; one row per coefficient."""
    specs = SPECS if specs is None else specs
    rows = []
    for h in horizons:
        for outcome in outcomes:
            ycol = f"y_{h}"
            frame = df
            if outcome == "winsor":
                lo, hi = cutoffs[h]
                frame = df.assign(**{ycol: winsorize(df[ycol], lo, hi)})
            for spec_id, (regs, fe, samp) in specs.items():
                sub = restrict(frame, samp)
                table, counts = fit_fe_regression(sub, ycol, regs, fe)
                for _, r in table.iterrows():
                    rows.append({
                        "spec_id": spec_id, "direction": direction, "horizon": h, "outcome": outcome,
                        "sample": samp, "fixed_effects": "+".join(fe), "regressor": r["regressor"],
                        "coef_bp": r["coef"], "se_bp": r["se"], "t_stat": r["t_stat"],
                        "p_two_sided": r["p_two_sided"], "ci_low_bp": r["ci_low"], "ci_high_bp": r["ci_high"],
                        **counts,
                        "is_trigger_spec": (spec_id, direction, h, outcome, r["regressor"]) == TRIGGER_SPEC,
                    })
    out = pd.DataFrame(rows)[RESULT_COLUMNS]
    assert_no_household_columns(out)
    return out


def assert_no_household_columns(table: pd.DataFrame) -> None:
    bad = [c for c in table.columns if c in FORBIDDEN_OUTPUT_COLUMNS]
    if bad:
        raise ValueError(f"per-unit identifier columns in an A8 output: {bad}")


def trigger_verdict(results: pd.DataFrame, threshold: float = TRIGGER_ABS_T) -> dict:
    """Checkpoint-6 rule: |t| >= 2.5 on log(lag), household-FE purchase spec, h20, raw Y."""
    row = results[results["is_trigger_spec"]]
    if len(row) != 1:
        raise ValueError("trigger specification must appear exactly once")
    t = float(row["t_stat"].iloc[0])
    return {"spec": list(TRIGGER_SPEC), "t_stat": t, "abs_t": abs(t), "threshold": threshold,
            "triggered": bool(abs(t) >= threshold), "coef_bp": float(row["coef_bp"].iloc[0]),
            "se_bp": float(row["se_bp"].iloc[0]), "n_events": int(row["n_events"].iloc[0]),
            "n_households": int(row["n_households"].iloc[0]), "n_dates": int(row["n_dates"].iloc[0])}


# ---------------------------------------------------------------- caveats (counts only)
HIST_EDGES = (0, 15, 30, 45, 60, 90, 120, 180, 365)


def lag_histogram(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Counts of episodes by lag bin, exact-day spikes around the 30/45-day rules, and lag quantiles."""
    lag = df["lag_days"].astype(float)
    cut = pd.cut(lag, bins=list(HIST_EDGES), right=True)
    rows = [{"kind": "bin", "bin": f"{int(iv.left) + 1}-{int(iv.right)}", "n_events": int(n),
             "value": float(n / len(df)), "value_is": "share"} for iv, n in cut.value_counts(sort=False).items()]
    for day in (28, 29, 30, 31, 44, 45, 46, 47):
        n = int((lag == day).sum())
        rows.append({"kind": "exact_day", "bin": str(day), "n_events": n, "value": n / len(df), "value_is": "share"})
    for q in (10, 25, 50, 75, 90):
        rows.append({"kind": "quantile", "bin": f"p{q}", "n_events": int(len(df)),
                     "value": float(np.percentile(lag, q)), "value_is": "lag_days"})
    return pd.DataFrame(rows).assign(direction=direction)


def report_caveats(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Batch filing, within-report lag dispersion, document format, within/between lag variance."""
    reports = df.drop_duplicates("document_uid")
    multi = reports["report_n_lines"] > 1
    rng = reports.loc[multi, "report_lag_range"]
    ll = df["log_lag"]
    var_total = float(ll.var(ddof=0))
    var_within = float(df["log_lag_within"].var(ddof=0)) if "log_lag_within" in df else np.nan
    rows = [
        ("n_events", len(df)), ("n_reports", len(reports)),
        ("share_reports_multi_transaction", float(multi.mean())),
        ("share_events_in_multi_transaction_reports", float((df["report_n_lines"] > 1).mean())),
        ("median_lines_per_report", float(reports["report_n_lines"].median())),
        ("p90_lines_per_report", float(reports["report_n_lines"].quantile(0.9))),
        ("median_within_report_lag_range_days_multi", float(rng.median()) if len(rng) else np.nan),
        ("p75_within_report_lag_range_days_multi", float(rng.quantile(0.75)) if len(rng) else np.nan),
        ("share_multi_reports_lag_range_ge30", float((rng >= 30).mean()) if len(rng) else np.nan),
        ("share_log_lag_variance_within_report", var_within / var_total if var_total > 0 else np.nan),
        ("n_events_lag_ge30", int((df["lag_days"] >= KEY_CUT).sum())),
        ("n_events_window_ends_before_public_date", int((df["window_end_date"] < df["public_date"]).sum())),
        ("n_events_lag_ge30_window_ends_before_public_date",
         int(((df["lag_days"] >= KEY_CUT) & (df["window_end_date"] < df["public_date"])).sum())),
        ("n_events_lag_gt45", int((df["lag_days"] > GT_CUT).sum())),
        ("n_households", df["household_id"].nunique()), ("n_anchor_dates", df["anchor_date"].nunique()),
    ]
    for fmt, n in df["document_format"].value_counts().items():
        rows.append((f"n_events_format_{fmt}", int(n)))
    for ch, n in df["chamber"].value_counts().items():
        rows.append((f"n_events_chamber_{ch}", int(n)))
    for (ch, fmt), n in df.groupby(["chamber", "document_format"]).size().items():
        rows.append((f"n_events_{ch}_{fmt}", int(n)))
    for label in BIN_LABELS:
        rows.append((f"n_events_bin_{label}", int(df["lag_bin"].eq(label).sum())))
    return pd.DataFrame(rows, columns=["metric", "value"]).assign(direction=direction)


# ---------------------------------------------------------------- pipeline
def load_inputs(cache_path: Path = CACHE_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Real-data inputs: D's T events, raw timed lines (no outcome columns), N's cache, the v2 calendar."""
    from trading_on_disclosure_v4.data.calendar import load_calendar
    from trading_on_disclosure_v4.data.load import load_timed_transactions

    events = pd.read_parquet(EVENTS_T)
    lines = load_timed_transactions()
    cols = ["permno", "anchor_idx"] + [c for pair in HORIZONS.values() for c in pair]
    cache = pd.read_parquet(cache_path, columns=cols)
    return events, lines, cache, load_calendar()


def build_frame(events: pd.DataFrame, timed: pd.DataFrame, cache: pd.DataFrame, calendar: pd.DatetimeIndex,
                direction: str) -> tuple[pd.DataFrame, dict]:
    """Selection -> report link (from raw timed lines) -> outcomes -> regressors, with the count report."""
    sample, report = select_sample(events, direction)
    df = add_regressors(attach_outcomes(attach_report(sample, observed_lines(timed)), cache, calendar))
    for h, (_, rcol) in HORIZONS.items():
        report[f"outcome_{h}_reasons"] = {str(k): int(v) for k, v in df[rcol].value_counts().items()}
    return df, report


def _log(path: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(path, "a") as fh:
        fh.write(line + "\n")


def main(argv: list[str] | None = None) -> dict:
    import argparse

    ap = argparse.ArgumentParser(description="A8 stage 1 (pre-tag, exploratory)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--log", type=Path, default=RESTRICTED_DIR / "logs" / "stage1.log")
    ap.add_argument("--directions", nargs="+", default=["purchase", "sale"])
    args = ap.parse_args(argv)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _log(args.log, "start A8 stage 1")
    events, lines, cache, calendar = load_inputs()
    _log(args.log, f"inputs loaded in {time.time() - t0:.1f}s (cache rows {len(cache)})")
    results, hists, caveats, reports, verdict = [], [], [], {}, None
    for direction in args.directions:
        t1 = time.time()
        df, reports[direction] = build_frame(events, lines, cache, calendar, direction)
        cutoffs, sources = {}, {}
        for h in HORIZONS:
            lo, hi, src = winsor_cutoffs(direction, h, sample=df)
            cutoffs[h], sources[h] = (lo, hi), {"lo_bp": lo, "hi_bp": hi, "source": src}
        reports[direction]["winsor_cutoffs_bp"] = sources
        res = run_specs(df, direction, cutoffs=cutoffs)
        results.append(res)
        hists.append(lag_histogram(df, direction))
        caveats.append(report_caveats(df, direction))
        if direction == "purchase":
            verdict = trigger_verdict(res)
        _log(args.log, f"{direction}: {reports[direction]['n_selected']} events, {len(res)} coefficient rows, {time.time() - t1:.1f}s")
    out = pd.concat(results, ignore_index=True)
    assert_no_household_columns(out)
    out.to_csv(args.out_dir / "stage1_results.csv", index=False, float_format="%.10g")
    pd.concat(hists, ignore_index=True).to_csv(args.out_dir / "stage1_lag_histogram.csv", index=False, float_format="%.6g")
    pd.concat(caveats, ignore_index=True).to_csv(args.out_dir / "stage1_caveats.csv", index=False, float_format="%.6g")
    manifest = {"selection": reports, "trigger": verdict, "runtime_seconds": round(time.time() - t0, 1),
                "n_result_rows": int(len(out))}
    (args.out_dir / "stage1_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    _log(args.log, f"done in {time.time() - t0:.1f}s; triggered={verdict['triggered'] if verdict else None}")
    return manifest


if __name__ == "__main__":
    main()
