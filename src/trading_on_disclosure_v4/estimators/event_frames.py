"""Event frames: the household-keyed join of D's event tables to the forward cache, and the same
join for N's stored placebo draws.

`load_event_frame(clock)` returns one row per event (purchases and sales, periods A, B and X) with the
outcome columns of `estimators.specs` in bp and `sigma_event` = estimation-window residual SD x sqrt(20)
in bp. `placebo_event_frames(clock, flavour)` yields (draw, frame) for N's block-shift draws
(`flavour` "N1" = timing-only, "N2" = centered) joined to the same event metadata by `event_id`.
`pooled_cutoffs` reads the committed pooled table (purchases) for the winsorization step.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..outcomes import event_filters as ef
from ..outcomes import forward_cache as fc
from ..outcomes import panel as panel_mod

REPO_ROOT = panel_mod.REPO_ROOT
POOLED_CSV = REPO_ROOT / "data" / "v4" / "pooled" / "pooled_outcome_distribution.csv"
AUDIT_ROWS = REPO_ROOT / "data" / "v4" / "restricted" / "data" / "duplicate_audit_rows.parquet"
PANELS_ROOT = panel_mod.RESTRICTED_N / "panels"
BP = 1e4
SIGMA_HORIZON = 20
#: duplicates label of the committed pooled table per clock (T: 0 flagged episodes, so keepdups == default)
POOLED_DUP_LABEL = {"T": "keepdups", "P": "nodup"}
#: production placebo sets: default duplicate treatment and the keep-duplicates sensitivity
PLACEBO_SETS = {("T", "default"): "A0_T_purchase_AB_L20_both_keepdups", ("T", "keepdups"): "A0_T_purchase_AB_L20_both_keepdups",
                ("P", "default"): "A0_P_purchase_AB_L20_both_alllines", ("P", "keepdups"): "A0_P_purchase_AB_L20_both_keepdups_w3"}
FLAVOURS = {"N1": "timing-only placebo", "N2": "centered placebo"}
CACHE_Y = {"y_h20": "bhar_ff5mom_h20", "y_h10": "bhar_ff5mom_h10", "y_h60": "bhar_ff5mom_h60",
           "y_noalpha_h20": "bhar_ff5mom_noalpha_h20", "y_mkt_h20": "bhar_mkt_h20"}
META_COLUMNS = ["event_id", "household_id", "clock", "period", "anchor_date", "anchor_year", "window_end_date",
                "record_source", "permno", "direction"]


def pooled_cutoffs(clock: str, outcome: str = "ff5mom_h20", path: Path = POOLED_CSV,
                   direction: str = "purchase") -> dict[str, tuple[float, float]]:
    """{period: (p1, p99)} in bp for clock x outcome from the committed pooled table (plan §3)."""
    t = pd.read_csv(path)
    t = t[(t["clock"] == clock) & (t["direction"] == direction) & (t["horizon"] == outcome)
          & (t["duplicates"] == POOLED_DUP_LABEL[clock])]
    if len(t) != 2 or set(t["period"]) != {"A", "B"}:
        raise ValueError(f"expected pooled rows for periods A and B of {clock}/{outcome}, found {len(t)}")
    return {str(r.period): (float(r.winsor_lo) * BP, float(r.winsor_hi) * BP) for r in t.itertuples()}


def sample_cutoffs(y: np.ndarray, period: np.ndarray) -> dict[str, tuple[float, float]]:
    """{period: (p1, p99)} of the finite outcomes, per period (used where no committed table exists)."""
    out = {}
    for p in ("A", "B"):
        v = np.asarray(y, dtype=float)[np.asarray(period) == p]
        v = v[np.isfinite(v)]
        if len(v):
            out[p] = (float(np.quantile(v, 0.01)), float(np.quantile(v, 0.99)))
    return out


def blind_vendor_households(n: int = 2, audit_rows: Path = AUDIT_ROWS) -> frozenset[str]:
    """The `n` households of the 'without parsed official filing' class (`data.coverage`) with the most
    vendor lines in the duplicate audit. Returned for in-memory
    masks only: never print, log or store the result."""
    from ..data import coverage as cv
    from ..data import load

    classes = cv.classify_households(load.load_matched_transactions())
    without = classes.index[classes == cv.CLASS_WITHOUT]
    lines = pd.read_parquet(audit_rows, columns=["filer_uid"]).groupby("filer_uid").size()
    top = lines.reindex(without).fillna(0).sort_values(ascending=False, kind="stable")
    return frozenset(str(h) for h in top.index[:n])


def _dup_flags(clock: str) -> pd.DataFrame:
    f = pd.read_parquet(ef.FLAGS_PATH, columns=["event_id", "clock", "dup_primary", "dup_primary_all_lines"])
    f = f.loc[f["clock"] == clock, ["event_id", "dup_primary", "dup_primary_all_lines"]]
    return f.rename(columns={"dup_primary": "dup_any_line", "dup_primary_all_lines": "dup_all_lines"})


def event_metadata(clock: str, directions=("purchase", "sale"), periods=("A", "B")) -> pd.DataFrame:
    """D's events (no outcome) with both duplicate flags; the metadata every frame shares."""
    ev = pd.read_parquet(ef.EVENTS_DIR / f"events_{clock}.parquet", columns=META_COLUMNS)
    ev = ev[ev["direction"].isin(directions) & ev["period"].isin(periods)]
    out = ev.merge(_dup_flags(clock), on="event_id", how="left")
    out["dup_any_line"] = out["dup_any_line"].fillna(False).astype(bool)
    out["dup_all_lines"] = out["dup_all_lines"].fillna(False).astype(bool)
    return out.reset_index(drop=True)


def load_event_frame(clock: str, calendar: pd.DatetimeIndex | None = None, directions=("purchase", "sale"),
                     periods=("A", "B")) -> pd.DataFrame:
    """Actual event frame: metadata + outcomes at the event's own anchor (bp)."""
    meta = event_metadata(clock, directions=directions, periods=periods)
    calendar = panel_mod.v2_trading_calendar() if calendar is None else calendar
    cache = fc.load_cache(columns=["permno", "anchor_idx", "resid_sd", *CACHE_Y.values()])
    meta["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(meta["anchor_date"]))
    if (meta["anchor_idx"] < 0).any():
        raise ValueError(f"{int((meta['anchor_idx'] < 0).sum())} anchors are not calendar sessions")
    out = meta.merge(cache, on=["permno", "anchor_idx"], how="left")
    for y, col in CACHE_Y.items():
        out[y] = out[col].astype(float) * BP
    out["sigma_event"] = out["resid_sd"].astype(float) * np.sqrt(SIGMA_HORIZON) * BP
    return out.drop(columns=[*CACHE_Y.values(), "resid_sd", "anchor_idx"])


def placebo_set_dir(clock: str, flavour: str, dup: str = "default", root: Path = PANELS_ROOT) -> Path:
    if flavour not in FLAVOURS:
        raise ValueError(f"unknown placebo flavour {flavour!r}; expected one of {sorted(FLAVOURS)}")
    return root / clock / flavour / PLACEBO_SETS[(clock, "keepdups" if dup == "keepdups" else "default")]


def placebo_event_frames(clock: str, flavour: str, meta: pd.DataFrame | None = None, dup: str = "default",
                         first: int = 0, n: int = 2000, root: Path = PANELS_ROOT,
                         batches: bool = False) -> Iterator[tuple[int, pd.DataFrame]]:
    """(draw, event frame) for N's stored draws first..first+n-1: shifted outcomes (bp), shifted anchor as
    `anchor_date`, original `anchor_year` / `window_end_date` / `record_source` / `permno` / `direction`
    from the metadata. With `batches=True` yields (batch_index, frame of every draw in the file) instead."""
    meta = event_metadata(clock) if meta is None else meta
    meta = meta.drop(columns=["anchor_date", "clock", "period", "household_id"])
    d = placebo_set_dir(clock, flavour, dup, root)
    prefix = flavour.lower()
    last = first + n - 1
    for i, p in enumerate(sorted(d.glob(f"{prefix}_batch_*.parquet"))):
        avail = pq.read_schema(p).names
        cols = ["draw", "event_id", "household_id", "period", "shifted_anchor_date"]
        cols += [c for c in CACHE_Y if c in avail] + [c for c in ("sigma_event", "resid_sd") if c in avail]
        batch = pd.read_parquet(p, columns=cols)
        batch = batch[(batch["draw"] >= first) & (batch["draw"] <= last)]
        if batch.empty:
            continue
        f = batch.rename(columns={"shifted_anchor_date": "anchor_date"})
        for c in ("event_id", "household_id", "period"):
            f[c] = f[c].astype(str)
        for y in CACHE_Y:
            if y in f.columns:
                f[y] = f[y].astype(float) * BP
        if "sigma_event" in f.columns:
            f["sigma_event"] = f["sigma_event"].astype(float) * BP
        elif "resid_sd" in f.columns:
            f["sigma_event"] = f["resid_sd"].astype(float) * np.sqrt(SIGMA_HORIZON) * BP
            f = f.drop(columns="resid_sd")
        f["clock"] = clock
        f = f.merge(meta, on="event_id", how="left").reset_index(drop=True)
        if batches:
            yield i, f
            continue
        for draw, g in f.groupby("draw", sort=True):
            yield int(draw), g.reset_index(drop=True)


def placebo_draw_frame(clock: str, draw: int, flavour: str = "N2", meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """One placebo draw as an event frame (the dry-run substitute for the actual frame)."""
    for d, f in placebo_event_frames(clock, flavour, meta, first=draw, n=1):
        if d == draw:
            return f
    raise ValueError(f"placebo draw {draw} of {clock}/{flavour} not on disk")
