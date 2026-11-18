"""Three-subperiod covariance matrix: is the two-period covariance special?

The plan's estimand is the w-weighted covariance of two period means. This module cuts the same anchors
into THREE consecutive subperiods of equal calendar length and computes the 3x3 w-weighted covariance
(and correlation) matrix of the subperiod mean purchase outcomes, over the households active in all
three. Each element uses the pairwise event-share weight w_h^{ij} = sqrt(n_hi x n_hj), so every
off-diagonal element is the plan's own estimator applied to that pair of subperiods and the diagonal is
the w-weighted variance of that subperiod's means.

Boundaries (half-open, 44 calendar months each, covering the anchors 2015-2025 of the sample):
2015-01-01 / 2018-09-01 / 2022-05-01 / 2026-01-01. The 20-session completion rule is kept as elsewhere in
the paper: a purchase counts in a subperiod only when its 20-trading-day window also ends inside that
subperiod, so no outcome straddles a boundary. The rule is applied to the frame's OWN anchor date, so a
placebo draw is labelled by its shifted anchor exactly as the actual frame is labelled by its anchor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: half-open subperiod boundaries; three intervals of 44 calendar months
SUBPERIOD_BOUNDS = ("2015-01-01", "2018-09-01", "2022-05-01", "2026-01-01")
#: plain subperiod names, in order
SUBPERIODS = ("first", "second", "third")
#: pairs whose covariance is reported (the off-diagonal elements)
SUBPERIOD_PAIRS = (("first", "second"), ("first", "third"), ("second", "third"))
HORIZON = 20


def subperiod_labels(anchor_date, calendar: pd.DatetimeIndex, bounds=SUBPERIOD_BOUNDS, horizon: int = HORIZON) -> np.ndarray:
    """Subperiod name per event ("" when the anchor is outside every subperiod or its 20-session window
    does not complete inside the subperiod holding the anchor)."""
    cal = pd.DatetimeIndex(calendar).to_numpy(dtype="datetime64[ns]")
    ad = pd.DatetimeIndex(anchor_date).to_numpy(dtype="datetime64[ns]")
    pos = pd.DatetimeIndex(calendar).get_indexer(pd.DatetimeIndex(anchor_date))
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} anchors are not calendar sessions")
    end_pos = pos + horizon
    complete = end_pos < len(cal)
    end = np.where(complete, cal[np.clip(end_pos, 0, len(cal) - 1)], np.datetime64("NaT", "ns"))
    out = np.full(len(ad), "", dtype=object)
    for i, name in enumerate(SUBPERIODS):
        lo, hi = np.datetime64(bounds[i], "ns"), np.datetime64(bounds[i + 1], "ns")
        out[(ad >= lo) & (ad < hi) & complete & (end < hi)] = name
    return out


def subperiod_table(panel: pd.DataFrame, calendar: pd.DatetimeIndex, y: str = "y_h20",
                    household: str = "household_id", anchor: str = "anchor_date",
                    bounds=SUBPERIOD_BOUNDS, horizon: int = HORIZON) -> tuple[pd.DataFrame, dict]:
    """(table of households active in ALL THREE subperiods, counts). The table has `n_<name>` and
    `mean_<name>` per subperiod; the counts dict carries the events kept and dropped and the household
    totals. Household ids stay the index and are never written out."""
    lab = subperiod_labels(panel[anchor], calendar, bounds, horizon)
    p = panel.assign(_sub=lab)
    kept = p.loc[p["_sub"] != ""]
    g = kept.groupby([household, "_sub"], sort=True)[y].agg(["count", "mean"]).unstack("_sub")
    have = [s for s in SUBPERIODS if ("count", s) in g.columns]
    if len(have) < len(SUBPERIODS):
        raise ValueError(f"only {len(have)} of {len(SUBPERIODS)} subperiods have purchases")
    g = g.dropna()
    table = pd.DataFrame({f"n_{s}": g[("count", s)] for s in SUBPERIODS} | {f"mean_{s}": g[("mean", s)] for s in SUBPERIODS})
    counts = {"n_events_kept": int(len(kept)), "n_events_dropped": int(len(panel) - len(kept)),
              "n_households_all_three": int(len(table)), "n_households_panel": int(panel[household].nunique())}
    for s in SUBPERIODS:
        counts[f"n_events_{s}"] = int(table[f"n_{s}"].sum()) if len(table) else 0
    return table, counts


def weighted_covariance(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """w-weighted covariance of two household vectors (the plan's estimand for one pair of periods)."""
    a, b, w = (np.asarray(v, dtype=float) for v in (a, b, w))
    sw = float(np.sum(w))
    return float(np.sum(w * (a - float(np.sum(w * a)) / sw) * (b - float(np.sum(w * b)) / sw)) / sw)


def covariance_matrix(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(3x3 covariance, 3x3 correlation) with pairwise weights sqrt(n_hi x n_hj) in bp^2."""
    k = len(SUBPERIODS)
    cov = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            w = np.sqrt(table[f"n_{SUBPERIODS[i]}"].to_numpy(dtype=float) * table[f"n_{SUBPERIODS[j]}"].to_numpy(dtype=float))
            cov[i, j] = weighted_covariance(table[f"mean_{SUBPERIODS[i]}"].to_numpy(), table[f"mean_{SUBPERIODS[j]}"].to_numpy(), w)
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(d, d)
    return cov, corr


def subperiod_statistics(panel: pd.DataFrame, calendar: pd.DatetimeIndex, y: str = "y_h20",
                         household: str = "household_id", anchor: str = "anchor_date") -> dict[str, float]:
    """Every element of the matrix as a flat dict: `cov_<i>_<j>`, `var_<i>`, `corr_<i>_<j>`, plus counts.
    The same function runs on the actual panel and on every placebo draw."""
    table, counts = subperiod_table(panel, calendar, y=y, household=household, anchor=anchor)
    cov, corr = covariance_matrix(table)
    out: dict[str, float] = {f"var_{s}": float(cov[i, i]) for i, s in enumerate(SUBPERIODS)}
    for a, b in SUBPERIOD_PAIRS:
        i, j = SUBPERIODS.index(a), SUBPERIODS.index(b)
        out[f"cov_{a}_{b}"] = float(cov[i, j])
        out[f"corr_{a}_{b}"] = float(corr[i, j])
    out.update({k: float(v) for k, v in counts.items()})
    return out


#: the statistics whose placebo distribution the subperiod analysis needs
SUBPERIOD_STATS = tuple([f"cov_{a}_{b}" for a, b in SUBPERIOD_PAIRS] + [f"var_{s}" for s in SUBPERIODS])
