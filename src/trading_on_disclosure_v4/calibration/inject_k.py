"""Sparse-household alternative re-parametrized in k informed households drawn from a test's own
analysis set (plan §11 row "Sparse-household alternative"; referee calibration_wave2 MAJOR-4).

`base` names the household set the k informed households are drawn from: "R1" (the T2 set: R1-eligible
households present in the panel) or "n20" (the T3 set: households with >= `min_n` kept events in the
panel); "all" reproduces N's share-based population for comparison. Selection is uniform or
activity-proportional (probability proportional to the household's event count in the draw), without
replacement. Seeds: seed_from(name, draw, "sparse_k", panel_id) — N's convention (`inject/effects.py`).
Operates on shifted placebo (N2) panels only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import winners
from trading_on_disclosure_v4.inject.effects import HH, panel_id_of
from trading_on_disclosure_v4.outcomes.seeds import seed_from

BASES = ("R1", "n20", "all")
K_GRID = (1, 2, 4, 8)


def base_households(panel: pd.DataFrame, base: str, r1: set[str] | None, min_n: int = winners.MIN_EVENTS_FULL_SAMPLE) -> pd.Series:
    """Event counts (index = household id, sorted) of the households in the base set of ONE draw."""
    counts = panel.groupby(HH, sort=True).size()
    if base == "R1":
        if r1 is None:
            raise ValueError("base 'R1' needs the R1 household set")
        return counts[counts.index.isin(list(r1))]
    if base == "n20":
        return counts[counts >= min_n]
    if base == "all":
        return counts
    raise ValueError(f"unknown base {base!r}; expected one of {BASES}")


def inject_k_households(
    panel: pd.DataFrame, k: int, mu: float, name: str, base: str, r1: set[str] | None = None,
    selection: str = "uniform", y_col: str = "y_h20", min_n: int = winners.MIN_EVENTS_FULL_SAMPLE,
    panel_id: str | None = None, k_frac: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """k informed households (theta_h = mu) drawn from the base set of each draw; returns (panel, info).

    `k_frac` (0 < k_frac <= 1) replaces k by round(k_frac * H) per draw (k_frac = 1: every household of the
    base set informed, a pure mean shift on that set). info: n_base (H of the draw), k_effective, selection, base."""
    if selection not in ("uniform", "activity"):
        raise ValueError(f"unknown selection {selection!r}")
    if k_frac is not None and not 0.0 < k_frac <= 1.0:
        raise ValueError("k_frac must lie in (0, 1]")
    if k_frac is None and k < 0:
        raise ValueError("k must be non-negative")
    pid = panel_id_of(panel) if panel_id is None else panel_id
    out = panel.copy()
    y = out[y_col].to_numpy(dtype=float).copy()
    theta_h = np.zeros(len(out))
    info = {"base": base, "selection": selection, "k": int(k) if k_frac is None else -1, "mu": float(mu),
            "k_frac": float(k_frac) if k_frac is not None else np.nan}
    for draw, ix in out.groupby("draw", sort=True).indices.items():
        sub = out.iloc[ix]
        cand = base_households(sub, base, r1, min_n)
        H = int(len(cand))
        m = min(int(k), H) if k_frac is None else int(round(k_frac * H))
        rng = np.random.default_rng(seed_from(name, int(draw), "sparse_k", pid))
        p = (cand.to_numpy(dtype=float) / float(cand.sum())) if selection == "activity" else None
        chosen = rng.choice(cand.index.to_numpy(), size=m, replace=False, p=p) if m > 0 else np.array([], dtype=object)
        hit = sub[HH].isin(set(chosen)).to_numpy()
        y[ix[hit]] += float(mu)
        theta_h[ix[hit]] = float(mu)
        info.update({"n_base": H, "k_effective": m})
    out[y_col] = y
    out["theta_h"] = theta_h
    return out, info
