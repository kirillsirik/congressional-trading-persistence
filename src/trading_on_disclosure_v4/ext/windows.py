"""Audit of incomplete return windows (extension plan section 4): outcome statuses with delisting and
endpoint censoring, partial-window returns, and the sensitivity of the spread, the persistence-sample
purchase mean and tau2 to the missing windows. Missing returns are never zero except in treatment (v)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trading_on_disclosure_v2.outcomes import factor_models as fm
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tests_t1_t4 as tt
from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.ext import common
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import event_filters as ef
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel as pm

H = 20
BP = 1e4
OUTCOMES = ("market_adjusted", "ff5mom")
T_OBSERVED = "(i) observed_only"
T_PARTIAL = "(ii) partial_window_imputed"
T_MEAN = "(iii) direction_mean_imputed"
T_LOW = "(iv) bound_low"
T_HIGH = "(iv) bound_high"
T_ZERO = "(v) zero_imputed_forbidden"
TREATMENTS = (T_OBSERVED, T_PARTIAL, T_MEAN, T_LOW, T_HIGH, T_ZERO)
QUANTILES = (0.05, 0.95)
PSEUDO_BLOCK = -1
PSEUDO_HOUSEHOLD = "pseudo_unresolved_eligible"
UNRESOLVED = "unresolved_eligible_security"
P_UNIVERSE = "dedup_nonmixed_with_entry_session"
T_UNIVERSE = "events_T_purchase_sale_periods_AB"
P_COLS = {"market_adjusted": ("ar_E_h20", "ar_E_h20_reason"), "ff5mom": ("car_ff5mom_h20", "car_ff5mom_h20_reason")}
T_COLS = {"market_adjusted": ("bhar_mkt_h20", "reason_mkt_h20"), "ff5mom": ("bhar_ff5mom_h20", "reason_h20")}
SENS_COLUMNS = ["estimand", "clock", "treatment", "estimate", "unit", "se_block", "n_used", "n_imputed",
                "retention_purchase", "retention_sale", "notes"]
AUDIT_CSV = common.OUT_DIR / "window_audit.csv"
PARTIAL_CSV = common.OUT_DIR / "partial_window_returns.csv"
SENSITIVITY_CSV = common.OUT_DIR / "missing_return_sensitivity.csv"
RETENTION_CSV = common.OUT_DIR / "retention_rates.csv"
PARTIAL_PARQUET = common.RESTRICTED_EXT / "windows" / "partial_outcomes.parquet"
LEDGER_COUNTS = common.OUT_DIR / "ledger_tier_d_classification.csv"
RESULTS_CSV = common.REPO_ROOT / "data" / "v4" / "results.csv"
LOG = common.LOG_DIR / "windows.log"



def window_bounds(anchor_idx: int, n_sessions: int, h: int = H) -> tuple[int, int]:
    """Calendar positions of the first and last session of the window anchor+1..anchor+h, clipped at the calendar end."""
    return anchor_idx + 1, min(anchor_idx + h, n_sessions - 1)


def delisting_in_window(anchor_idx: np.ndarray, delist_idx: np.ndarray, h: int = H) -> np.ndarray:
    """True when the security's delisting-return row sits in sessions anchor+1..anchor+h (NaN delist_idx -> False)."""
    a = np.asarray(anchor_idx, dtype=float)
    d = np.asarray(delist_idx, dtype=float)
    return (d >= a + 1) & (d <= a + h)


def endpoint_censoring(anchor_idx: np.ndarray, n_sessions: int, factor_end_idx: int, outcome: str, h: int = H) -> np.ndarray:
    """none / past_panel_end / past_factor_end (the factor label only on the factor outcome; the panel label wins)."""
    a = np.asarray(anchor_idx, dtype=np.int64)
    past_panel = a + h > n_sessions - 1
    past_factor = (a + h > factor_end_idx) & (outcome == "ff5mom")
    return np.where(past_panel, "past_panel_end", np.where(past_factor, "past_factor_end", "none"))


def partial_window(r: np.ndarray, ref: np.ndarray, lo: int, hi: int) -> dict:
    """Compound r and ref over the finite sessions lo..hi; gap pattern, both buy-and-hold returns and their difference."""
    out = {"n_window": int(max(hi - lo + 1, 0)), "n_observed": 0, "n_gap_leading": 0, "n_gap_interior": 0,
           "n_gap_trailing": 0, "bh_security": np.nan, "bh_reference": np.nan, "partial_outcome": np.nan}
    if hi < lo:
        return out
    wr, wb = np.asarray(r, dtype=float)[lo:hi + 1], np.asarray(ref, dtype=float)[lo:hi + 1]
    obs = np.isfinite(wr) & np.isfinite(wb)
    n = int(obs.sum())
    if n == 0:
        out["n_gap_leading"] = int(len(obs))
        return out
    first = int(np.argmax(obs))
    last = int(len(obs) - 1 - np.argmax(obs[::-1]))
    bh_r = float(np.prod(1.0 + wr[obs]) - 1.0)
    bh_b = float(np.prod(1.0 + wb[obs]) - 1.0)
    out.update(n_observed=n, n_gap_leading=first, n_gap_trailing=int(len(obs) - 1 - last),
               n_gap_interior=int((~obs[first:last + 1]).sum()), bh_security=bh_r, bh_reference=bh_b,
               partial_outcome=bh_r - bh_b)
    return out


def expected_return_grid(beta: np.ndarray, F: np.ndarray, rf: np.ndarray) -> np.ndarray:
    """rf + alpha + beta'f on the calendar grid (NaN where a factor or rf is missing)."""
    return np.asarray(rf, dtype=float) + float(beta[0]) + np.asarray(F, dtype=float) @ np.asarray(beta[1:], dtype=float)



def retention_rates(frame: pd.DataFrame, y: str, by: tuple[str, ...] = ("direction", "period")) -> pd.DataFrame:
    """n_signals, n_observed (finite y) and retention by `by`, with an `all` margin on every key."""
    f = frame.assign(_obs=np.isfinite(pd.to_numeric(frame[y], errors="coerce").to_numpy(dtype=float)))
    keys = list(by)
    rows = []
    for subset in [keys[:k] for k in range(len(keys), -1, -1)]:
        groups = f.groupby(subset, sort=True) if subset else [((), f)]
        for key_vals, g in groups:
            vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
            row = {key: "all" for key in keys}
            row.update(dict(zip(subset, vals, strict=False)))
            n, obs = int(len(g)), int(g["_obs"].sum())
            rows.append({**row, "n_signals": n, "n_observed": obs, "retention": obs / n if n else np.nan})
    return pd.DataFrame(rows)


def impute(y: np.ndarray, direction: np.ndarray, treatment: str, partial: np.ndarray | None = None,
           quantiles: tuple[float, float] = QUANTILES) -> tuple[np.ndarray, int, str]:
    """Outcome vector under one treatment (NaN where still missing), the number imputed, and the fill rule used."""
    y = np.asarray(y, dtype=float).copy()
    direction = np.asarray(direction)
    missing = ~np.isfinite(y)
    if treatment == T_OBSERVED:
        return y, 0, "observed windows only"
    if treatment == T_PARTIAL:
        p = np.full(len(y), np.nan) if partial is None else np.asarray(partial, dtype=float)
        fill = missing & np.isfinite(p)
        y[fill] = p[fill]
        return y, int(fill.sum()), "incomplete windows with a partial-window outcome imputed at that outcome"
    if treatment not in TREATMENTS:
        raise ValueError(f"unknown treatment {treatment!r}")
    lo_q, hi_q = quantiles
    fills, parts = {}, []
    for d in ("purchase", "sale"):
        v = y[~missing & (direction == d)]
        if not len(v):
            continue
        if treatment == T_MEAN:
            fills[d], rule = float(v.mean()), "observed direction mean"
        elif treatment == T_LOW:
            q = lo_q if d == "purchase" else hi_q
            fills[d], rule = float(np.quantile(v, q)), f"q{q:g} of observed {d}s"
        elif treatment == T_HIGH:
            q = hi_q if d == "purchase" else lo_q
            fills[d], rule = float(np.quantile(v, q)), f"q{q:g} of observed {d}s"
        else:
            fills[d], rule = 0.0, "zero (the treatment the brief forbids; printed for distance only)"
        parts.append(f"missing {d}s at {rule} = {fills[d]:.2f} bp")
    n_imputed = 0
    for d, val in fills.items():
        m = missing & (direction == d)
        y[m] = val
        n_imputed += int(m.sum())
    return y, n_imputed, "; ".join(parts)


def spread_estimate(frame: pd.DataFrame, y: np.ndarray) -> dict:
    """Block-clustered purchase-minus-sale spread (household x 20-day block of the row's own anchor) on the finite rows."""
    ok = np.isfinite(np.asarray(y, dtype=float))
    f = frame.loc[ok]
    return rerun.clustered_mean_or_spread(np.asarray(y, dtype=float)[ok], (f["direction"].to_numpy() == "purchase").astype(float),
                                          f["household"].to_numpy(), f["blk"].to_numpy(), "follower_entry_date_blk20")


def purchase_mean_estimate(frame: pd.DataFrame, y: np.ndarray) -> dict:
    """T1 clustered mean (household x block) of the finite rows of a purchase frame with household/period/blk columns."""
    ok = np.isfinite(np.asarray(y, dtype=float))
    f = frame.loc[ok]
    p = pd.DataFrame({"household_id": f["household"].to_numpy(), "period": f["period"].to_numpy(), "y_h20": np.asarray(y, dtype=float)[ok],
                      "anchor_blk": f["blk"].to_numpy()})
    return tt.t1_clustered_mean(p, anchor="anchor_blk")


def sens_row(estimand: str, clock: str, treatment: str, estimate, unit: str, se, n_used, n_imputed, r_p, r_s, notes: str) -> dict:
    return {"estimand": estimand, "clock": clock, "treatment": treatment, "estimate": estimate, "unit": unit, "se_block": se,
            "n_used": n_used, "n_imputed": n_imputed, "retention_purchase": r_p, "retention_sale": r_s, "notes": notes}


def spread_treatments(frame: pd.DataFrame, estimand: str = "public_spread_dedup", clock: str = "P") -> list[dict]:
    """The five treatments of the spread on a frame with direction/household/blk/y (bp, NaN = missing)/y_partial."""
    obs = np.isfinite(frame["y"].to_numpy(dtype=float))
    dirn = frame["direction"].to_numpy()
    r_p = float(obs[dirn == "purchase"].mean()) if (dirn == "purchase").any() else np.nan
    r_s = float(obs[dirn == "sale"].mean()) if (dirn == "sale").any() else np.nan
    rows = []
    for t in TREATMENTS:
        yi, n_imp, rule = impute(frame["y"].to_numpy(dtype=float), dirn, t, frame["y_partial"].to_numpy(dtype=float))
        e = spread_estimate(frame, yi)
        notes = f"{rule}; {e['n_households']} households, {e['n_dates_or_blocks']} blocks; imputed rows cluster in their own anchor's block"
        if t == T_ZERO:
            notes = "FORBIDDEN treatment (missing returns are never zero); " + notes
        rows.append(sens_row(estimand, clock, t, e["estimate"], "bp", e["se"], e["n"], n_imp, r_p, r_s, notes))
    return rows


def purchase_mean_treatments(frame: pd.DataFrame, estimand: str, clock: str = "T", extra_notes: str = "") -> list[dict]:
    """The five treatments of the raw purchase mean on a purchase frame (household/period/blk/y/y_partial)."""
    obs = np.isfinite(frame["y"].to_numpy(dtype=float))
    r_p = float(obs.mean())
    rows = []
    for t in TREATMENTS:
        yi, n_imp, rule = impute(frame["y"].to_numpy(dtype=float), frame["direction"].to_numpy(), t, frame["y_partial"].to_numpy(dtype=float))
        e = purchase_mean_estimate(frame, yi)
        notes = f"raw outcome; {rule}; {e['n_households']} households, {e['n_anchor_dates']} blocks; imputed rows cluster in their own anchor's block"
        if t == T_ZERO:
            notes = "FORBIDDEN treatment (missing returns are never zero); " + notes
        rows.append(sens_row(estimand, clock, t, e["estimate"], "bp", e["se"], e["n_events"], n_imp, r_p, np.nan, f"{notes}; {extra_notes}".rstrip("; ")))
    return rows


def targets_note(pers: pd.DataFrame) -> str:
    """Count-only description of the imputation targets (non-finite purchases) by status and partial availability."""
    miss = pers.loc[~np.isfinite(pers["y"].to_numpy(dtype=float))]
    by_status = "; ".join(f"{k} {v}" for k, v in miss["status"].value_counts().sort_index().items())
    return f"{len(miss)} targets: {by_status}; {int(np.isfinite(miss['y_partial'].to_numpy(dtype=float)).sum())} with a partial window"


def record_level_bound(spread_frame: pd.DataFrame, n_unresolved_eligible_purchase: int, n_unresolved_eligible_sale: int,
                       estimand: str = "public_spread_dedup", clock: str = "P", quantiles: tuple[float, float] = QUANTILES) -> list[dict]:
    """Bound (iv) with the unresolved-eligible records as extra missing rows (one pseudo-household and pseudo-block)."""
    obs = spread_frame.loc[np.isfinite(spread_frame["y"].to_numpy(dtype=float)), ["direction", "household", "blk", "y"]]
    n_p, n_s = int((obs["direction"] == "purchase").sum()), int((obs["direction"] == "sale").sum())
    n_up, n_us = int(n_unresolved_eligible_purchase), int(n_unresolved_eligible_sale)
    extra = pd.DataFrame({"direction": ["purchase"] * n_up + ["sale"] * n_us, "household": PSEUDO_HOUSEHOLD, "blk": PSEUDO_BLOCK, "y": np.nan})
    frame = pd.concat([obs, extra], ignore_index=True)
    frame["y_partial"] = np.nan
    r_p, r_s = n_p / (n_p + n_up) if n_p + n_up else np.nan, n_s / (n_s + n_us) if n_s + n_us else np.nan
    rows = []
    for t in (T_LOW, T_HIGH):
        yi, n_imp, rule = impute(frame["y"].to_numpy(dtype=float), frame["direction"].to_numpy(), t, None, quantiles)
        e = spread_estimate(frame, yi)
        notes = (f"record-level counterpart: {n_up} purchase and {n_us} sale unresolved-eligible records enter as extra rows ({rule}); "
                 f"retention = retained signals / (retained + unresolved eligible) by direction; the extra rows share one pseudo-household "
                 f"and one pseudo-block so the two-way clustering is defined (the SE is mechanical, not an inference); window failures not added")
        rows.append(sens_row(estimand, clock, f"{t}_record_level", e["estimate"], "bp", e["se"], e["n"], n_imp, r_p, r_s, notes))
    return rows


def awaiting_record_level_rows(path: Path, estimand: str = "public_spread_dedup", clock: str = "P") -> list[dict]:
    """Placeholder rows when the ledger's tier-D classification is not on disk yet."""
    note = f"awaiting {path.name}"
    return [sens_row(estimand, clock, f"{t}_record_level", np.nan, "bp", np.nan, np.nan, np.nan, np.nan, np.nan, note) for t in (T_LOW, T_HIGH)]


def read_record_level_counts(path: Path) -> dict[str, int] | None:
    """{direction: unresolved-eligible count} from the ledger's tier-D classification; None when the file is absent."""
    if not Path(path).exists():
        return None
    t = pd.read_csv(path)
    if "level" in t.columns:
        t = t.loc[t["level"] == "record"]
    t = t.loc[t["category"] == UNRESOLVED]
    ncol = next((c for c in ("n", "count", "n_records", "n_lines") if c in t.columns), None)
    g = t.groupby("direction")[ncol].sum() if ncol else t.groupby("direction").size()
    return {d: int(g.get(d, 0)) for d in ("purchase", "sale")}



def delisting_positions(daily_path: Path, calendar: pd.DatetimeIndex) -> pd.Series:
    """Calendar position of each PERMNO's first delisting-return row (`is_delisting_return_row`)."""
    d = pq.read_table(daily_path, columns=["permno", "date"], filters=[("is_delisting_return_row", "=", True)]).to_pandas()
    d["idx"] = calendar.get_indexer(pd.DatetimeIndex(d["date"]))
    if (d["idx"] < 0).any():
        raise ValueError(f"{int((d['idx'] < 0).sum())} delisting rows are not dated on a session")
    return d.sort_values(["permno", "idx"]).drop_duplicates("permno").set_index("permno")["idx"]


class Market:
    """Calendar grids of the benchmark and factors, delisting positions, and per-PERMNO return series loaded on demand."""

    def __init__(self, calendar: pd.DatetimeIndex, bench: pd.Series, factors: pd.DataFrame, daily_path: Path):
        self.calendar, self.n = calendar, len(calendar)
        self.bench = pm.align_to_calendar(bench, calendar)
        self.factors = factors
        self.F = factors[list(pm.FACTOR_COLS)].reindex(calendar).to_numpy(dtype=float)
        self.rf = factors["rf"].reindex(calendar).to_numpy(dtype=float)
        self.factor_end_idx = int(calendar.searchsorted(factors.index.max(), side="right") - 1)
        self.daily_path = daily_path
        self.delist = delisting_positions(daily_path, calendar)
        self.panel_permnos = set(pq.read_table(daily_path, columns=["permno"]).column("permno").unique().to_pylist())
        self._series: dict[int, pd.Series] = {}
        self._grid: dict[int, np.ndarray] = {}

    def load(self, permnos) -> None:
        """Read total_ret of the given PERMNOs from the local daily panel (only those not loaded yet)."""
        need = sorted(set(int(p) for p in permnos) - set(self._series))
        if not need:
            return
        d = pq.read_table(self.daily_path, columns=["permno", "date", "total_ret"], filters=[("permno", "in", need)]).to_pandas()
        for p, g in d.groupby("permno"):
            self._series[int(p)] = g.set_index("date")["total_ret"]
        for p in need:
            self._series.setdefault(p, pd.Series(dtype=float))

    def series(self, permno: int) -> pd.Series:
        self.load([permno])
        return self._series[int(permno)]

    def grid(self, permno: int) -> np.ndarray:
        if int(permno) not in self._grid:
            self._grid[int(permno)] = pm.align_to_calendar(self.series(permno), self.calendar)
        return self._grid[int(permno)]

    def loadings(self, permno: int, anchor_date) -> np.ndarray | None:
        """The v2 FF5+MOM loadings for one event (252-session window ending 21 sessions before the anchor)."""
        return fm._estimate_loadings(self.series(permno), self.factors, pd.Timestamp(anchor_date), self.calendar, pm.FACTOR_COLS)



def public_universe(calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """The de-duplicated non-mixed signals with an entry session, with both outcomes (bp) and statuses; count report."""
    signals = rerun.load_signals()
    dedup, report = rerun.dedup_filter(signals, rerun.duplicate_signal_uids())
    events = pd.read_parquet(ef.EVENTS_DIR / "events_P.parquet", columns=["event_id", "signal_uid", "period"])
    u = dedup.loc[dedup["follower_entry_date"].notna()].merge(events, on="signal_uid", how="left", validate="1:1")
    if u["period"].isna().any():
        raise ValueError(f"{int(u['period'].isna().sum())} signals with an entry session are not in events_P.parquet")
    out = pd.DataFrame({"clock": "P", "universe": P_UNIVERSE, "event_id": u["event_id"].to_numpy(), "signal_uid": u["signal_uid"].to_numpy(),
                        "household": u[rerun.HOUSEHOLD].astype(str).to_numpy(), "permno": u["permno"].astype(int).to_numpy(),
                        "direction": u["direction"].to_numpy(), "period": u["period"].to_numpy(), "anchor_date": u["follower_entry_date"].to_numpy()})
    for outcome, (ycol, rcol) in P_COLS.items():
        out[f"y_{outcome}"] = u[ycol].to_numpy(dtype=float) * BP
        out[f"status_{outcome}"] = u[rcol].to_numpy()
    out["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(out["anchor_date"]))
    if (out["anchor_idx"] < 0).any():
        raise ValueError("public-clock anchors off the calendar")
    report.update(n_signals=int(len(signals)), n_no_entry=int(dedup["follower_entry_date"].isna().sum()), n_universe=int(len(out)))
    return out.reset_index(drop=True), report


def transaction_universe(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """events_T purchases and sales of periods A and B joined to the forward cache's outcomes (bp) and reason codes."""
    meta = evf.event_metadata("T")
    meta["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(meta["anchor_date"]))
    if (meta["anchor_idx"] < 0).any():
        raise ValueError("transaction-clock anchors off the calendar")
    cols = ["permno", "anchor_idx", *[c for pair in T_COLS.values() for c in pair]]
    cache = pq.read_table(fc.CACHE_PATH, columns=cols).to_pandas(strings_to_categorical=True)
    cache = cache.loc[cache["permno"].isin(meta["permno"].unique())]
    t = meta.merge(cache, on=["permno", "anchor_idx"], how="left", validate="m:1")
    out = pd.DataFrame({"clock": "T", "universe": T_UNIVERSE, "event_id": t["event_id"].to_numpy(), "signal_uid": None,
                        "household": t["household_id"].astype(str).to_numpy(), "permno": t["permno"].astype(int).to_numpy(),
                        "direction": t["direction"].to_numpy(), "period": t["period"].to_numpy(), "anchor_date": t["anchor_date"].to_numpy(),
                        "anchor_idx": t["anchor_idx"].to_numpy()})
    for outcome, (ycol, rcol) in T_COLS.items():
        out[f"y_{outcome}"] = t[ycol].to_numpy(dtype=float) * BP
        out[f"status_{outcome}"] = t[rcol].astype(object).where(t[rcol].notna(), "not_in_cache").to_numpy()
    out["dup_all_lines"] = t["dup_all_lines"].to_numpy(dtype=bool)
    return out.reset_index(drop=True)


def attach_flags(frame: pd.DataFrame, market: Market) -> pd.DataFrame:
    """Add the 20-day block, the delisting-in-window flag and the per-outcome endpoint-censoring labels."""
    out = frame.copy()
    out["blk"] = out["anchor_idx"].to_numpy() // rerun.BLOCK_L
    didx = market.delist.reindex(out["permno"]).to_numpy(dtype=float)
    flag = delisting_in_window(out["anchor_idx"].to_numpy(), didx)
    in_panel = out["permno"].isin(market.panel_permnos).to_numpy()
    out["delisting_in_window"] = np.where(in_panel, np.where(flag, "True", "False"), "NA")
    for outcome in OUTCOMES:
        out[f"endpoint_{outcome}"] = endpoint_censoring(out["anchor_idx"].to_numpy(), market.n, market.factor_end_idx, outcome)
    return out



def audit_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Tidy counts by clock x universe x outcome x direction x period x status x delisting x endpoint."""
    parts = []
    for outcome in OUTCOMES:
        g = frame.groupby(["clock", "universe", "direction", "period", f"status_{outcome}", "delisting_in_window", f"endpoint_{outcome}"], sort=True).size()
        p = g.reset_index(name="n").rename(columns={f"status_{outcome}": "status", f"endpoint_{outcome}": "endpoint_censoring"})
        p.insert(2, "outcome", outcome)
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def compute_partials(frame: pd.DataFrame, market: Market, outcome: str) -> pd.DataFrame:
    """Partial-window outcomes (market-adjusted or FF5+MOM) over the observed sessions of every row of `frame`."""
    market.load(frame["permno"].unique())
    rows = []
    for e in frame.itertuples(index=False):
        r = market.grid(e.permno)
        lo, hi = window_bounds(int(e.anchor_idx), market.n)
        if outcome == "market_adjusted":
            ref, have = market.bench, True
        else:
            beta = market.loadings(e.permno, e.anchor_date)
            have = beta is not None
            ref = expected_return_grid(beta, market.F, market.rf) if have else np.full(market.n, np.nan)
        pw = partial_window(r, ref, lo, hi)
        rows.append({"clock": e.clock, "outcome": outcome, "event_id": e.event_id, "signal_uid": e.signal_uid, "direction": e.direction,
                     "period": e.period, "permno": int(e.permno), "anchor_date": e.anchor_date, "status": getattr(e, f"status_{outcome}"),
                     "delisting_in_window": e.delisting_in_window, "endpoint_censoring": getattr(e, f"endpoint_{outcome}"),
                     "loadings_available": have, **pw, "partial_outcome_bp": pw["partial_outcome"] * BP})
    return pd.DataFrame(rows)


def incomplete_rows(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    return frame.loc[frame[f"status_{outcome}"] != "ok"]


def partial_summary(partials: pd.DataFrame) -> pd.DataFrame:
    """mean / median / sd (bp) and sessions observed of the partial outcomes by direction x status x delisting, with margins."""
    rows = []
    keys = ["direction", "status", "delisting_in_window"]
    subsets = [keys, keys[:2], [keys[0], keys[2]], keys[1:], keys[:1], [keys[1]], [keys[2]], []]
    for (clock, outcome), p in partials.groupby(["clock", "outcome"], sort=True):
        for subset in subsets:
            groups = p.groupby(subset, sort=True) if subset else [((), p)]
            for key_vals, g in groups:
                vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
                row = {"clock": clock, "outcome": outcome, **{kk: "all" for kk in keys}}
                row.update(dict(zip(subset, vals, strict=False)))
                w = g.loc[np.isfinite(g["partial_outcome_bp"].to_numpy(dtype=float))]
                row.update(n_windows=int(len(g)), n_with_partial=int(len(w)),
                           sessions_observed_mean=float(w["n_observed"].mean()) if len(w) else np.nan,
                           mean_bp=float(w["partial_outcome_bp"].mean()) if len(w) else np.nan,
                           median_bp=float(w["partial_outcome_bp"].median()) if len(w) else np.nan,
                           sd_bp=float(w["partial_outcome_bp"].std(ddof=1)) if len(w) > 1 else np.nan)
                rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def check_complete_windows(frame: pd.DataFrame, market: Market, outcome: str, n: int, seed: int) -> dict:
    """Max |partial over the full window - released outcome| (bp) on a random sample of complete windows."""
    ok = frame.loc[frame[f"status_{outcome}"] == "ok"]
    sample = ok.sample(n=min(n, len(ok)), random_state=seed)
    p = compute_partials(sample, market, outcome)
    diff = np.abs(p["partial_outcome_bp"].to_numpy(dtype=float) - sample[f"y_{outcome}"].to_numpy(dtype=float))
    return {"n_checked": int(len(sample)), "max_abs_diff_bp": float(np.nanmax(diff)), "n_not_finite": int((~np.isfinite(diff)).sum()),
            "all_20_sessions": bool((p["n_observed"] == H).all())}


def retention_table(p_frame: pd.DataFrame, t_frame: pd.DataFrame, persistence: pd.DataFrame | None) -> pd.DataFrame:
    """Retention by clock x universe x outcome x direction x period (observed = finite released outcome)."""
    parts = []
    for frame in (p_frame, t_frame):
        for outcome in OUTCOMES:
            r = retention_rates(frame, f"y_{outcome}")
            r.insert(0, "outcome", outcome)
            r.insert(0, "universe", frame["universe"].iloc[0])
            r.insert(0, "clock", frame["clock"].iloc[0])
            parts.append(r)
    if persistence is not None:
        r = retention_rates(persistence.assign(y_ff5mom=persistence["y"]), "y_ff5mom")
        r.insert(0, "outcome", "ff5mom")
        r.insert(0, "universe", persistence["universe"].iloc[0])
        r.insert(0, "clock", "T")
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def spread_frame(p_frame: pd.DataFrame, partials: pd.DataFrame) -> pd.DataFrame:
    """Public-clock rows for the spread: y = released market-adjusted outcome (bp), y_partial = partial-window outcome (bp)."""
    pm_rows = partials.loc[(partials["clock"] == "P") & (partials["outcome"] == "market_adjusted"), ["signal_uid", "partial_outcome_bp"]]
    f = p_frame.merge(pm_rows.rename(columns={"partial_outcome_bp": "y_partial"}), on="signal_uid", how="left", validate="1:1")
    f["y"] = np.where(f["status_market_adjusted"] == "ok", f["y_market_adjusted"], np.nan)
    return f[["signal_uid", "direction", "period", "household", "blk", "status_market_adjusted", "y", "y_partial"]].rename(columns={"status_market_adjusted": "status"})


def persistence_frame(ctx: rs.Context, calendar: pd.DatetimeIndex, t_frame: pd.DataFrame, partials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, sp.Spec]:
    """The headline panel's events (raw y) plus the persistence households' non-finite purchases as imputation targets."""
    spec = sp.headline_spec("T")
    panel = ctx.panel(spec)
    households = set(panel["household_id"].astype(str))
    fr = ctx.frames["T"]
    allp = fr.loc[(fr["direction"] == "purchase") & fr["period"].isin(["A", "B"]) & ~fr["dup_all_lines"].astype(bool)
                  & fr["household_id"].astype(str).isin(households), ["event_id", "household_id", "period", "anchor_date", "y_h20"]].copy()
    allp["y_h20"] = pd.to_numeric(allp["y_h20"], errors="coerce").to_numpy(dtype=float)
    in_panel = allp["event_id"].isin(set(panel["event_id"]))
    if int(in_panel.sum()) != len(panel) or (in_panel & ~np.isfinite(allp["y_h20"])).any():
        raise ValueError("the headline panel is not the finite-outcome subset of the persistence households' purchases")
    pf = partials.loc[(partials["clock"] == "T") & (partials["outcome"] == "ff5mom"), ["event_id", "partial_outcome_bp"]]
    f = allp.merge(pf.rename(columns={"partial_outcome_bp": "y_partial"}), on="event_id", how="left", validate="1:1")
    f = f.merge(t_frame[["event_id", "status_ff5mom"]], on="event_id", how="left", validate="1:1")
    f["blk"] = rerun.block_of(f["anchor_date"], calendar)
    out = pd.DataFrame({"event_id": f["event_id"], "household": f["household_id"].astype(str), "period": f["period"], "anchor_date": f["anchor_date"],
                        "direction": "purchase", "blk": f["blk"], "y": f["y_h20"], "y_partial": f["y_partial"], "status": f["status_ff5mom"],
                        "universe": f"persistence_{len(households)}_households_purchases"})
    return out.reset_index(drop=True), panel, spec


def tau2_treatments(ctx: rs.Context, calendar: pd.DatetimeIndex, panel: pd.DataFrame, pers: pd.DataFrame, spec: sp.Spec, extra_notes: str = "") -> list[dict]:
    """tau2 (event-share weights, block clusters) observed and with the partial-window FF5+MOM rows appended (winsorized)."""
    cutoffs = ctx.cutoffs_for(spec)
    r_p = float(np.isfinite(pers["y"].to_numpy(dtype=float)).mean())
    rows = []
    pb = rs.block_cluster(panel, calendar)
    r = cgm.tau2_i1(pb, spec.weights, anchor="anchor_blk")
    n_used = int(r["n_events_a"] + r["n_events_b"])
    rows.append(sens_row("T_tau2_headline", "T", T_OBSERVED, r["tau2_hat"], "bp2", r["var_wx"] * r["se_b"], n_used, 0, r_p, np.nan,
                         f"winsorized headline panel; {r['n_households']} households; events A {r['n_events_a']}, B {r['n_events_b']}; {r['n_anchor_dates']} blocks; {extra_notes}".rstrip("; ")))
    tgt = pers.loc[~np.isfinite(pers["y"].to_numpy(dtype=float)) & np.isfinite(pers["y_partial"].to_numpy(dtype=float))]
    add = pd.DataFrame({"draw": 0, "event_id": tgt["event_id"].to_numpy(), "household_id": tgt["household"].to_numpy(), "clock": "T",
                        "period": tgt["period"].to_numpy(), "y_h20": tgt["y_partial"].to_numpy(dtype=float), "anchor_date": tgt["anchor_date"].to_numpy()})
    add = sp.winsorize(add, cutoffs)
    p2 = pd.concat([panel[["draw", "event_id", "household_id", "clock", "period", "y_h20", "anchor_date"]], add], ignore_index=True)
    r2 = cgm.tau2_i1(rs.block_cluster(p2, calendar), spec.weights, anchor="anchor_blk")
    rows.append(sens_row("T_tau2_headline", "T", T_PARTIAL, r2["tau2_hat"], "bp2", r2["var_wx"] * r2["se_b"], int(r2["n_events_a"] + r2["n_events_b"]),
                         int(len(add)), r_p, np.nan, f"partial-window FF5+MOM outcomes winsorized at the pooled cutoffs; {r2['n_households']} households; "
                         f"events A {r2['n_events_a']}, B {r2['n_events_b']}; {r2['n_anchor_dates']} blocks"))
    for t in (T_MEAN, T_LOW, T_HIGH, T_ZERO):
        rows.append(sens_row("T_tau2_headline", "T", t, np.nan, "bp2", np.nan, np.nan, np.nan, r_p, np.nan,
                             "not computed: the plan applies only the partial-window imputation to tau2 (bounds do not translate to a covariance)"))
    return rows



def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="audit of incomplete return windows (extension plan section 4)")
    ap.add_argument("--record-level-counts", type=Path, default=LEDGER_COUNTS)
    ap.add_argument("--check-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args(argv)
    t0 = time.time()

    def log(msg: str) -> None:
        common.log_line(LOG, msg)

    log("windows: start")
    calendar = pm.v2_trading_calendar()
    bench, factors = pm.load_benchmark(), pm.load_factors()
    market = Market(calendar, bench, factors, pm.daily_parquet_path())
    log(f"calendar {len(calendar)} sessions {calendar[0].date()}..{calendar[-1].date()}; factor end {factors.index.max().date()} "
        f"(idx {market.factor_end_idx}); {len(market.delist)} delisting rows")
    p_frame, report = public_universe(calendar)
    p_frame = attach_flags(p_frame, market)
    log(f"public universe {report}")
    t_frame = attach_flags(transaction_universe(calendar), market)
    log(f"transaction universe {len(t_frame)} events; not_in_cache {int((t_frame['status_ff5mom'] == 'not_in_cache').sum())}")
    both = pd.concat([p_frame, t_frame], ignore_index=True)
    audit = audit_table(both)
    _write(audit, AUDIT_CSV)
    log(f"audit rows {len(audit)} -> {AUDIT_CSV.relative_to(common.REPO_ROOT)}")
    partials = pd.concat([compute_partials(incomplete_rows(f, o), market, o) for f in (p_frame, t_frame) for o in OUTCOMES], ignore_index=True)
    PARTIAL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    partials.to_parquet(PARTIAL_PARQUET, index=False)
    summary = partial_summary(partials)
    _write(summary, PARTIAL_CSV)
    log(f"partial outcomes {len(partials)} rows ({int(np.isfinite(partials['partial_outcome_bp']).sum())} with >= 1 observed session) -> parquet; summary {len(summary)} rows")
    checks = {}
    for f, label in ((p_frame, "P"), (t_frame, "T")):
        for o in OUTCOMES:
            checks[f"{label}/{o}"] = check_complete_windows(f, market, o, args.check_n, args.seed)
            log(f"complete-window check {label}/{o}: {checks[f'{label}/{o}']}")
    ctx, _ = common.load_headline_context(clocks=("T",), with_placebo=False)
    pers, panel, spec = persistence_frame(ctx, calendar, t_frame, partials)
    retention = retention_table(p_frame, t_frame, pers)
    _write(retention, RETENTION_CSV)
    sf = spread_frame(p_frame, partials)
    rows = spread_treatments(sf)
    on_file = pd.read_csv(rerun.TABLES_PATH)
    ref = on_file.loc[(on_file["test_id"] == rerun.PRIMARY_TEST_ID) & (on_file["sample"] == "dedup") & (on_file["clusters"] == "block20")].iloc[0]
    d_est, d_se = abs(rows[0]["estimate"] - ref["estimate"]), abs(rows[0]["se_block"] - ref["se"])
    log(f"spread (i) vs rerun_tables.csv: |d estimate| {d_est:.3e} bp, |d se| {d_se:.3e} bp, n {rows[0]['n_used']} vs {int(ref['n'])}")
    if d_est > 1e-6 or d_se > 1e-6 or rows[0]["n_used"] != int(ref["n"]):
        raise AssertionError("the observed-only spread does not reproduce data/v4/existing/rerun_tables.csv")
    note = targets_note(pers)
    log(f"persistence sample: {note}")
    rows += purchase_mean_treatments(pers, "T_purchase_mean_persistence", extra_notes=note)
    rows += tau2_treatments(ctx, calendar, panel, pers, spec, extra_notes=note)
    res = pd.read_csv(RESULTS_CSV)
    tau_ref = res.loc[(res["clock"] == "T") & (res["analysis"] == "tau") & (res["variant"] == "event_share") & (res["statistic"] == "tau2")].iloc[0]
    tau_row = next(r for r in rows if r["estimand"] == "T_tau2_headline" and r["treatment"] == T_OBSERVED)
    log(f"tau2 (i) vs results.csv: |d| {abs(tau_row['estimate'] - tau_ref['value']):.3e} bp2, |d se| {abs(tau_row['se_block'] - tau_ref['se_block']):.3e}")
    if abs(tau_row["estimate"] - tau_ref["value"]) > 1e-6:
        raise AssertionError("the observed-only tau2 does not reproduce data/v4/results.csv")
    counts = read_record_level_counts(args.record_level_counts)
    if counts is None:
        rows += awaiting_record_level_rows(args.record_level_counts)
        log(f"record-level counts absent ({args.record_level_counts}); placeholder rows written")
    else:
        rows += record_level_bound(sf, counts["purchase"], counts["sale"])
        log(f"record-level counts {counts} from {args.record_level_counts}")
    sens = pd.DataFrame(rows)[SENS_COLUMNS]
    for c in ("n_used", "n_imputed"):
        sens[c] = pd.to_numeric(sens[c], errors="raise").astype("Int64")
    _write(sens, SENSITIVITY_CSV)
    entries = [common.file_entry(rerun.SIGNALS_PATH, report["n_signals"]), common.file_entry(ef.EVENTS_DIR / "events_P.parquet"),
               common.file_entry(ef.EVENTS_DIR / "events_T.parquet"), common.file_entry(ef.FLAGS_PATH), common.file_entry(fc.MANIFEST_PATH),
               common.file_entry(fc.CACHE_PATH), common.file_entry(pm.daily_parquet_path()),
               common.file_entry(pm.processed_dir() / "benchmark_daily_2014plus.parquet"),
               common.file_entry(pm.project_root() / "data" / "raw" / "factors" / "factors_daily.csv"),
               common.file_entry(rerun.TABLES_PATH), common.file_entry(RESULTS_CSV), common.file_entry(evf.POOLED_CSV)]
    if counts is not None:
        entries.append(common.file_entry(args.record_level_counts))
    extra = {"outputs": [str(p.relative_to(common.REPO_ROOT)) for p in (AUDIT_CSV, PARTIAL_CSV, SENSITIVITY_CSV, RETENTION_CSV, PARTIAL_PARQUET)],
             "complete_window_checks": checks, "universe_counts": {"P": report, "T_events": int(len(t_frame))},
             "record_level_counts": counts, "seconds": round(time.time() - t0, 1)}
    common.write_manifest("windows", entries, extra)
    log(f"windows: done in {time.time() - t0:.1f}s")
    return {"audit": audit, "partials": partials, "summary": summary, "retention": retention, "sensitivity": sens, "checks": checks}


if __name__ == "__main__":
    main()
