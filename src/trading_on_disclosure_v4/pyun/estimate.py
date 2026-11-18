"""Outcomes and clustered quantities for one specification state of the bridge.

A STATE fixes: the record (all / official / dedup lines), the unit (line / event), Pyun's pairing rule,
the entry convention per clock (pyun / ours), the horizon, the benchmark (vw / sp / ff5mom) and the
period. For each state six quantities are computed on purchases (and sales where a spread is asked):
``T_mean`` and ``P_mean`` (mean abnormal return from the transaction and the public date), ``TP_diff``
(the paired difference on the same purchases), ``TP_diff_all`` (the same pooled over purchases and
sales, the registered ``TW_C1_PAIRED`` form), ``P_spread`` and ``T_spread`` (purchase minus sale).
Every quantity carries three standard errors: household x filing month (Pyun's Table 3), household x
20-session block (ours) and household x anchor day. Estimates in bp.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.existing.rerun import clustered_mean_or_spread
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.pyun import frames as fr
from trading_on_disclosure_v4.pyun.returns import ReturnGrid

BP = 1e4
BLOCK_L = 20
QUANTITIES = ("T_mean", "P_mean", "TP_diff", "TP_diff_all", "P_spread", "T_spread")
CLUSTER_SCHEMES = ("pyun", "block", "day")
FF5_COLS = {10: "bhar_ff5mom_h10", 20: "bhar_ff5mom_h20", 60: "bhar_ff5mom_h60"}


@dataclass(frozen=True)
class State:
    record: str = "all"  # all | official | dedup
    unit: str = "line"  # line | event
    pairing: bool = True  # Pyun's rule: both dates inside the period, lag 1-365 days
    entry_T: str = "pyun"  # pyun | ours
    entry_P: str = "pyun"
    h: int = 30
    bench: str = "vw"  # raw | vw | sp | ff5mom
    period: str = "2015-2025"  # 2015-2025 | all | plan_AB
    winsor: bool = False  # 1/99 clip of the outcome over the sample (sensitivity only)

    def with_(self, **kw) -> State:
        return replace(self, **kw)


class Inputs:
    """Everything a state needs: lines, signals, episodes, the return grid and the FF5+MOM cache.
    Without arguments every input is loaded from disk; tests pass toy frames."""

    def __init__(self, calendar: pd.DatetimeIndex | None = None, lines: pd.DataFrame | None = None,
                 signals: pd.DataFrame | None = None, episodes: pd.DataFrame | None = None,
                 grid: ReturnGrid | None = None, cache: pd.DataFrame | None = None):
        from trading_on_disclosure_v4.data.calendar import load_calendar

        self.calendar = load_calendar() if calendar is None else calendar
        self.lines = fr.load_lines(calendar=self.calendar) if lines is None else lines
        self.signals = fr.load_signal_events(calendar=self.calendar) if signals is None else signals
        self.episodes = fr.load_episode_events(calendar=self.calendar) if episodes is None else episodes
        if "period_ab" not in self.signals.columns:
            self.signals["period_ab"] = fr.signal_period_ab(self.signals, self.calendar)
        if "month_pyun" not in self.episodes.columns:
            self.episodes["month_pyun"] = episode_filing_month(self.episodes, self.lines)
        if grid is None:
            permnos = np.unique(np.concatenate([self.lines["permno"].unique(), self.signals["permno"].unique(),
                                                self.episodes["permno"].unique()]))
            grid = ReturnGrid.load(permnos)
        self.grid = grid
        self.cache = fc.load_cache(columns=["permno", "anchor_idx", *FF5_COLS.values()]) if cache is None else cache

    def outcome(self, permno: np.ndarray, anchor_idx: np.ndarray, h: int, bench: str) -> np.ndarray:
        """Abnormal return (bp) at the given anchors; NaN when the window is incomplete."""
        if bench == "ff5mom":
            if h not in FF5_COLS:
                raise ValueError(f"FF5+MOM outcomes exist for horizons {sorted(FF5_COLS)}, not {h}")
            key = pd.DataFrame({"permno": np.asarray(permno, dtype=np.int64), "anchor_idx": np.asarray(anchor_idx, dtype=np.int32)})
            y = key.merge(self.cache[["permno", "anchor_idx", FF5_COLS[h]]], on=["permno", "anchor_idx"], how="left")[FF5_COLS[h]]
            return y.to_numpy(dtype=float) * BP
        return self.grid.bhar(permno, anchor_idx, h, bench) * BP


def episode_filing_month(episodes: pd.DataFrame, lines: pd.DataFrame) -> pd.Series:
    """Month of the earliest public date among an episode's lines (Pyun's filing-month cluster on the T unit)."""
    ex = episodes[["event_id", "transaction_uids"]].assign(_u=episodes["transaction_uids"].astype(str).str.split(";")).explode("_u")
    pub = lines.set_index("transaction_uid")["analysis_public_date"]
    ex["_d"] = pub.reindex(ex["_u"].to_numpy()).to_numpy()
    first = ex.groupby("event_id")["_d"].min()
    return first.reindex(episodes["event_id"]).dt.to_period("M").astype(str).to_numpy()


# ---------------------------------------------------------------- frames per state

def line_frame(inp: Inputs, st: State) -> pd.DataFrame:
    """Pyun's unit under the state: paired lines of the record with both anchors."""
    t = fr.select_record(inp.lines, st.record)
    if st.period == "2015-2025":
        t = fr.paired_lines(t)
    elif st.period == "all":
        t = t.loc[t["lag_days"].between(*fr.LAG_RANGE)]
    else:
        raise ValueError("the plan's A/B periods are defined on events, not lines")
    out = pd.DataFrame({
        "household_id": t[fr.HOUSEHOLD].astype(str).to_numpy(), "permno": t["permno"].to_numpy(),
        "direction": t["direction"].to_numpy(), "month_pyun": t["filing_month_pyun"].to_numpy(),
        "a_T": t[f"a_T_{st.entry_T}"].to_numpy(), "a_P": t[f"a_P_{st.entry_P}"].to_numpy()})
    out["ok_T"] = out["ok_P"] = True
    return out.reset_index(drop=True)


def _period_mask_events(anchor_idx: np.ndarray, period_ab: np.ndarray | None, st: State, calendar: pd.DatetimeIndex) -> np.ndarray:
    if st.period == "all":
        return np.ones(len(anchor_idx), dtype=bool)
    if st.period == "2015-2025":
        d = calendar[np.clip(anchor_idx, 0, len(calendar) - 1)]
        return (anchor_idx >= 0) & (d >= fr.PERIOD[0]) & (d <= fr.PERIOD[1])
    if st.period == "plan_AB":
        return np.isin(period_ab, ["A", "B"])
    raise ValueError(f"unknown period {st.period!r}")


def signal_frame(inp: Inputs, st: State) -> pd.DataFrame:
    """The P unit (non-mixed signals) under the state, with the signal's own transaction anchor."""
    if st.entry_T != "ours" or st.entry_P != "ours":
        raise ValueError("event units carry our entry conventions only")
    s = inp.signals
    keep = np.ones(len(s), dtype=bool)
    if st.record == "official":
        keep &= (s["record_source"] == "official").to_numpy()
    elif st.record == "dedup":
        keep &= ~s["dup_all_lines"].to_numpy()
    if st.pairing:
        uids = set(fr.paired_lines(fr.select_record(inp.lines, st.record))["transaction_uid"])
        keep &= fr.events_with_lines(s, uids, "parent_transaction_uids").to_numpy()
    ok_P = _period_mask_events(s["a_P_ours"].to_numpy(), s["period_ab"].to_numpy(), st, inp.calendar)
    ok_T = _period_mask_events(s["a_T_ours"].to_numpy(), s["period_ab"].to_numpy(), st, inp.calendar)
    keep &= ok_P | ok_T
    sub = s.loc[keep]
    return pd.DataFrame({"household_id": sub["household_id"].to_numpy(), "permno": sub["permno"].to_numpy(),
                         "direction": sub["direction"].to_numpy(), "month_pyun": sub["filing_month_pyun"].to_numpy(),
                         "a_T": sub["a_T_ours"].to_numpy(), "a_P": sub["a_P_ours"].to_numpy(),
                         "ok_T": ok_T[keep], "ok_P": ok_P[keep]}).reset_index(drop=True)


def episode_frame(inp: Inputs, st: State) -> pd.DataFrame:
    """The T unit (D's episodes) under the state; the plan drops no episode for duplicates."""
    if st.entry_T != "ours":
        raise ValueError("event units carry our entry conventions only")
    e = inp.episodes
    keep = np.ones(len(e), dtype=bool)
    if st.record == "official":
        keep &= (e["record_source"] == "official").to_numpy()
    if st.pairing:
        uids = set(fr.paired_lines(fr.select_record(inp.lines, st.record))["transaction_uid"])
        keep &= fr.events_with_lines(e, uids, "transaction_uids").to_numpy()
    keep &= _period_mask_events(e["a_T_ours"].to_numpy(), e["period"].to_numpy(), st, inp.calendar)
    sub = e.loc[keep]
    return pd.DataFrame({"household_id": sub["household_id"].astype(str).to_numpy(), "permno": sub["permno"].to_numpy(),
                         "direction": sub["direction"].to_numpy(), "month_pyun": sub["month_pyun"].to_numpy(),
                         "a_T": sub["a_T_ours"].to_numpy(), "a_P": np.full(len(sub), -1),
                         "ok_T": True, "ok_P": False}).reset_index(drop=True)


# ---------------------------------------------------------------- quantities

def _clip(y: np.ndarray, on: bool) -> np.ndarray:
    if not on:
        return y
    ok = np.isfinite(y)
    lo, hi = np.quantile(y[ok], [0.01, 0.99])
    return np.where(ok, np.clip(y, lo, hi), y)


def _stat(y: np.ndarray, purchase: np.ndarray | None, household: np.ndarray, month: np.ndarray, anchor: np.ndarray) -> dict:
    """Estimate with the three SEs; n, households; all on the finite rows of y."""
    ok = np.isfinite(y)
    y, household, month, anchor = y[ok], household[ok], month[ok], anchor[ok]
    purchase = None if purchase is None else purchase[ok]
    out = {"estimate": np.nan, "n": int(len(y)), "n_households": int(len(np.unique(household))) if len(y) else 0}
    if len(y) < 3 or (purchase is not None and (purchase.sum() == 0 or purchase.sum() == len(y))):
        for c in CLUSTER_SCHEMES:
            out[f"se_{c}"] = np.nan
            out[f"n_clusters_{c}"] = 0
        return out
    for scheme, cl in (("pyun", month), ("block", anchor // BLOCK_L), ("day", anchor)):
        r = clustered_mean_or_spread(y, purchase, household, cl, f"cl_{scheme}")
        out["estimate"] = r["estimate"]
        out[f"se_{scheme}"] = r["se"]
        out[f"n_clusters_{scheme}"] = r["n_dates_or_blocks"]
        out[f"dof_{scheme}"] = r["dof"]
    return out


def quantities(inp: Inputs, st: State, frame: pd.DataFrame, which=QUANTITIES) -> dict[str, dict]:
    """The requested quantities on one unit frame (columns household_id, permno, direction, month_pyun, a_T, a_P,
    ok_T, ok_P); a row enters a clock's quantities only when its anchor on that clock is inside the period."""
    hh, month = frame["household_id"].to_numpy(), frame["month_pyun"].to_numpy()
    buy = (frame["direction"].to_numpy() == "purchase")
    aT, aP = frame["a_T"].to_numpy(dtype=int), frame["a_P"].to_numpy(dtype=int)
    p = frame["permno"].to_numpy()
    okT, okP = frame["ok_T"].to_numpy(dtype=bool), frame["ok_P"].to_numpy(dtype=bool)
    yT = _clip(inp.outcome(p, aT, st.h, st.bench), st.winsor) if (aT >= 0).any() else np.full(len(p), np.nan)
    yP = _clip(inp.outcome(p, aP, st.h, st.bench), st.winsor) if (aP >= 0).any() else np.full(len(p), np.nan)
    yT, yP = np.where(okT, yT, np.nan), np.where(okP, yP, np.nan)
    out = {}
    if "T_mean" in which:
        out["T_mean"] = _stat(yT[buy], None, hh[buy], month[buy], aT[buy])
    if "P_mean" in which:
        out["P_mean"] = _stat(yP[buy], None, hh[buy], month[buy], aP[buy])
    if "TP_diff" in which:
        out["TP_diff"] = _stat((yT - yP)[buy], None, hh[buy], month[buy], aP[buy])
    if "TP_diff_all" in which:
        out["TP_diff_all"] = _stat(yT - yP, None, hh, month, aP)
    if "P_spread" in which:
        out["P_spread"] = _stat(yP, buy.astype(float), hh, month, aP)
    if "T_spread" in which:
        out["T_spread"] = _stat(yT, buy.astype(float), hh, month, aT)
    return out


def state_quantities(inp: Inputs, st: State) -> dict[str, dict]:
    """All six quantities of a state: on lines, or on episodes (T_mean) and signals (the rest)."""
    if st.unit == "line":
        return quantities(inp, st, line_frame(inp, st))
    if st.unit != "event":
        raise ValueError(f"unknown unit {st.unit!r}")
    out = quantities(inp, st, episode_frame(inp, st), which=("T_mean",))
    out.update(quantities(inp, st, signal_frame(inp, st), which=tuple(q for q in QUANTITIES if q != "T_mean")))
    return out
