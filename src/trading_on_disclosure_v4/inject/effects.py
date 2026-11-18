"""Effect injection into a centered (N2) placebo panel (analysis plan §6, A2).

Three alternatives, each applied independently per draw using the panel's own
household/event structure and returning the injected panel plus a truth table:
dense (theta_h ~ N(0, tau^2) on every event of household h), sparse households
(a fraction pi_H of households, drawn uniformly or proportional to activity,
gets theta_h = mu_H), sparse trades (each event informed w.p. pi_T, or w.p.
pi_h ~ Gamma(shape 1, mean pi_T) per household, with effect mu_T).
Units follow the outcome column (decimal returns; 1 bp = 1e-4). Seeds:
seed_from(name, draw, kind, panel_id) with panel_id = '<clocks>:<periods>' of the
panel (or the caller's `panel_id`), so T and P panels injected under the same
name receive different effect streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.outcomes.seeds import seed_from

HH = "household_id"


@dataclass
class Injection:
    panel: pd.DataFrame
    truth: pd.DataFrame
    params: dict


def _per_draw(panel: pd.DataFrame):
    return panel.groupby("draw", sort=True).indices.items()


def panel_id_of(panel: pd.DataFrame) -> str:
    """Seed component identifying the panel: clock(s) and period set, e.g. 'T:AB'."""
    clocks = "".join(sorted(map(str, panel["clock"].unique()))) if "clock" in panel else "?"
    periods = "".join(sorted(map(str, panel["period"].unique()))) if "period" in panel else "?"
    return f"{clocks}:{periods}"


def _rng(name: str, draw: int, kind: str, panel_id: str) -> np.random.Generator:
    """Seed = seed_from(name, draw, kind, panel_id): T and P panels under one name differ."""
    return np.random.default_rng(seed_from(name, draw, kind, panel_id))


def _households(panel: pd.DataFrame, ix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique household ids, inverse index per row and event counts, for one draw."""
    hh, inv, counts = np.unique(panel[HH].to_numpy()[ix], return_inverse=True, return_counts=True)
    return hh, inv, counts


def inject_dense(panel: pd.DataFrame, tau: float, name: str, y_col: str = "y_h20",
                 panel_id: str | None = None) -> Injection:
    """theta_h ~ N(0, tau^2) added to every event of household h (fresh per draw)."""
    pid = panel_id_of(panel) if panel_id is None else panel_id
    out = panel.copy()
    y = out[y_col].to_numpy(float).copy()
    rows = []
    for draw, ix in _per_draw(out):
        hh, inv, _ = _households(out, ix)
        theta = _rng(name, draw, "dense", pid).normal(0.0, tau, size=len(hh))
        y[ix] += theta[inv]
        rows.append(pd.DataFrame({"draw": draw, HH: hh, "theta": theta}))
    out[y_col] = y
    return Injection(out, pd.concat(rows, ignore_index=True),
                     {"kind": "dense", "tau": tau, "y_col": y_col, "name": name, "panel_id": pid})


def inject_sparse_households(
    panel: pd.DataFrame, pi_h: float, mu_h: float, name: str,
    selection: str = "uniform", y_col: str = "y_h20", panel_id: str | None = None,
) -> Injection:
    """round(pi_H * H) informed households get theta_h = mu_H; the rest 0.

    ``selection``: 'uniform' (equal probability) or 'activity' (probability
    proportional to the household's event count in that draw), without replacement.
    """
    if selection not in ("uniform", "activity"):
        raise ValueError(f"unknown selection {selection!r}")
    if not 0.0 <= pi_h <= 1.0:
        raise ValueError("pi_h must lie in [0, 1]")
    pid = panel_id_of(panel) if panel_id is None else panel_id
    out = panel.copy()
    y = out[y_col].to_numpy(float).copy()
    rows = []
    for draw, ix in _per_draw(out):
        hh, inv, counts = _households(out, ix)
        m = int(round(pi_h * len(hh)))
        p = counts / counts.sum() if selection == "activity" else None
        chosen = _rng(name, draw, "sparse_h", pid).choice(len(hh), size=m, replace=False, p=p)
        theta = np.zeros(len(hh))
        theta[chosen] = mu_h
        y[ix] += theta[inv]
        rows.append(pd.DataFrame({"draw": draw, HH: hh, "theta": theta, "n_events": counts}))
    out[y_col] = y
    return Injection(out, pd.concat(rows, ignore_index=True),
                     {"kind": "sparse_households", "pi_h": pi_h, "mu_h": mu_h,
                      "selection": selection, "y_col": y_col, "name": name, "panel_id": pid})


def inject_sparse_trades(
    panel: pd.DataFrame, pi_t: float, mu_t: float, name: str,
    household_specific: bool = False, y_col: str = "y_h20", panel_id: str | None = None,
) -> Injection:
    """Each event informed w.p. pi_T (common) or pi_h ~ Gamma(1, pi_T) capped at 1
    (household-specific, mean pi_T, CV 1); informed events get +mu_T."""
    if not 0.0 <= pi_t <= 1.0:
        raise ValueError("pi_t must lie in [0, 1]")
    pid = panel_id_of(panel) if panel_id is None else panel_id
    out = panel.copy()
    y = out[y_col].to_numpy(float).copy()
    informed = np.zeros(len(out), dtype=bool)
    rows = []
    for draw, ix in _per_draw(out):
        rng = _rng(name, draw, "sparse_t", pid)
        hh, inv, counts = _households(out, ix)
        if household_specific:
            prob_h = np.minimum(rng.gamma(shape=1.0, scale=pi_t, size=len(hh)), 1.0)
        else:
            prob_h = np.full(len(hh), pi_t)
        flag = rng.random(len(ix)) < prob_h[inv]
        informed[ix] = flag
        y[ix] += np.where(flag, mu_t, 0.0)
        rows.append(pd.DataFrame({"draw": draw, HH: hh, "pi_h": prob_h, "n_events": counts}))
    out[y_col] = y
    truth = out[["draw", "event_id", HH]].copy()
    truth["informed"] = informed
    truth["effect"] = np.where(informed, mu_t, 0.0)
    return Injection(out, truth, {"kind": "sparse_trades", "pi_t": pi_t, "mu_t": mu_t,
                                  "household_specific": household_specific, "y_col": y_col,
                                  "name": name, "panel_id": pid,
                                  "household_pi": pd.concat(rows, ignore_index=True)})
