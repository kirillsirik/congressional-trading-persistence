"""The original event frames with the extended fit cache attached, the original specifications, and the pooled
cutoffs of the new factor-only horizons.

New event-frame columns (bp): `y_noalpha_h10`, `y_noalpha_h60` (factor-only outcomes from the same loadings as the
event-study outcomes `y_h10`, `y_h60`), `x_alpha20` (20 * fitted daily alpha) and the fitted coefficients. The new
outcomes are registered in `estimators.specs.OUTCOMES` at run time only; no original module is edited.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.ext import common as ec
from trading_on_disclosure_v4.outcomes import pooled
from trading_on_disclosure_v4.writing_sample import common as cm
from trading_on_disclosure_v4.writing_sample import fits

NEW_OUTCOMES = {"ff5mom_noalpha_h10": "y_noalpha_h10", "ff5mom_noalpha_h60": "y_noalpha_h60"}
#: event-study outcome -> factor-only outcome with the same horizon and loadings
FACTOR_ONLY = {"ff5mom_h20": "ff5mom_noalpha_h20", "ff5mom_h10": "ff5mom_noalpha_h10", "ff5mom_h60": "ff5mom_noalpha_h60"}
FIT_COLS = ["alpha_daily", *fits.BETA_COLS, "n_est", "y0_h10", "y0_h60"]
POOLED_NEW = cm.OUT / "pooled_cutoffs_new_horizons.csv"


def register_outcomes() -> None:
    for k, v in NEW_OUTCOMES.items():
        sp.OUTCOMES[k] = v


def extend_frame(frame: pd.DataFrame, calendar: pd.DatetimeIndex, ext: pd.DataFrame) -> pd.DataFrame:
    """Attach alpha, loadings and the factor-only 10/60-session outcomes to an event frame by (permno, anchor)."""
    f = frame.copy()
    f["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(f["anchor_date"]))
    if (f["anchor_idx"] < 0).any():
        raise ValueError("anchor not on the calendar")
    f = f.merge(ext, on=["permno", "anchor_idx"], how="left", validate="many_to_one")
    f["y_noalpha_h10"] = f.pop("y0_h10").astype(float) * cm.BP
    f["y_noalpha_h60"] = f.pop("y0_h60").astype(float) * cm.BP
    f["x_alpha20"] = cm.H * f["alpha_daily"].astype(float) * cm.BP
    f["entry_block"] = f["anchor_idx"] // rs.BLOCK_L
    return f


def new_pooled_cutoffs(ext: pd.DataFrame, calendar: pd.DatetimeIndex, log) -> pd.DataFrame:
    """p1/p99 of the factor-only 10/60-session outcomes over the pooled purchase population of the committed table
    (same events, duplicate rule, percentile routine and %.8g storage). The routine is first checked against
    every committed horizon."""
    committed = pd.read_csv(evf.POOLED_CSV)
    rows = []
    cache_cols = {"ff5mom_h10": "y1_h10", "ff5mom_h20": "y1_h20", "ff5mom_h60": "y1_h60", "ff5mom_noalpha_h20": "y0_h20",
                  "ff5mom_noalpha_h10": "y0_h10", "ff5mom_noalpha_h60": "y0_h60"}
    for clock in ("T", "P"):
        events, _ = pooled.load_events(clock)
        ev = events.copy()
        ev["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(ev["anchor_date"]))
        joined = ev.merge(ext[["permno", "anchor_idx", *cache_cols.values()]], on=["permno", "anchor_idx"], how="left")
        for period, g in joined.groupby("period", sort=True):
            for outcome, col in cache_cols.items():
                ok = g[col].dropna()
                lo, hi = (float(f"{np.percentile(ok, q):.8g}") for q in (1, 99))
                rec = {"clock": clock, "period": period, "direction": "purchase", "horizon": outcome, "units": "decimal_return",
                       "duplicates": evf.POOLED_DUP_LABEL[clock], "n_events": int(len(g)), "n_ok": int(len(ok)), "winsor_lo": lo, "winsor_hi": hi}
                c = committed[(committed["clock"] == clock) & (committed["period"] == period) & (committed["horizon"] == outcome)]
                if len(c):
                    same = (lo == float(c["winsor_lo"].iloc[0])) and (hi == float(c["winsor_hi"].iloc[0])) and int(c["n_ok"].iloc[0]) == len(ok)
                    if not same:
                        raise AssertionError(f"pooled cutoff routine does not reproduce {clock}/{period}/{outcome}")
                    rec["reproduces_committed"] = True
                else:
                    rec["reproduces_committed"] = np.nan
                rows.append(rec)
    out = pd.DataFrame(rows)
    log(f"pooled cutoffs: routine reproduces all {int(out['reproduces_committed'].eq(True).sum())} committed rows exactly; "
        f"new rows {int(out['reproduces_committed'].isna().sum())}")
    return out


def cutoff_function(new: pd.DataFrame):
    """(clock, outcome) -> {period: (lo, hi)} in bp: the committed table, or the new rows for the new horizons."""
    def f(clock: str, outcome: str) -> dict[str, tuple[float, float]]:
        if outcome in NEW_OUTCOMES:
            t = new[(new["clock"] == clock) & (new["horizon"] == outcome)]
            if len(t) != 2:
                raise ValueError(f"no new cutoffs for {clock}/{outcome}")
            return {str(r.period): (float(r.winsor_lo) * cm.BP, float(r.winsor_hi) * cm.BP) for r in t.itertuples()}
        return evf.pooled_cutoffs(clock, outcome)
    return f


def load(log, clocks=("T", "P")):
    """(context with extended frames, calendar, extended cache cells, new pooled cutoffs)."""
    register_outcomes()
    ctx, calendar = ec.load_headline_context(tuple(clocks), with_placebo=False)
    ext = fits.load(["permno", "anchor_idx", *FIT_COLS, "y1_h10", "y1_h20", "y1_h60", "y0_h20", "reason_h20"])
    new = new_pooled_cutoffs(ext, calendar, log)
    cm.write_csv(new, POOLED_NEW)
    ext_ev = ext[["permno", "anchor_idx", *FIT_COLS]]
    for c in clocks:
        ctx.frames[c] = extend_frame(ctx.frames[c], calendar, ext_ev)
    ctx.cutoffs = cutoff_function(new)
    ctx._cutoffs.clear()
    return ctx, calendar, ext, new


def factor_only(spec: sp.Spec) -> sp.Spec:
    """The same specification with the factor-only outcome of the same horizon (name suffixed)."""
    return replace(spec, outcome=FACTOR_ONLY[spec.outcome], name=f"{spec.name}#factor_only")


def cohort_spec(clock: str) -> sp.Spec:
    """The persistence cohort's purchases with raw outcomes (the A4 / Table 4 sample)."""
    return replace(sp.headline_spec(clock), winsorized=False)


def all_purchases_spec(clock: str) -> sp.Spec:
    return sp.analysis_specs(clock)["purchase_mean"]


def attach(panel: pd.DataFrame, frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Columns of the event frame attached to a panel by event id (one row per event)."""
    f = frame.drop_duplicates("event_id").set_index("event_id")
    out = panel.copy()
    for c in cols:
        out[c] = f.loc[out["event_id"], c].to_numpy()
    return out
