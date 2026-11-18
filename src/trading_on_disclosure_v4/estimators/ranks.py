"""A4: Mogstad-Romano-Shaikh-Wilhelm simultaneous confidence sets for ranks (plan §7.2).

Rank 1 = largest mean. From a joint bootstrap distribution of household means the pairwise
differences D_jk are studentized; c = (1-alpha) quantile over draws of max_{j<k} |D*_jk - D_jk|/se_jk;
household j's set is [1 + #{k: D_jk + c se_jk < 0}, H - #{k: D_jk - c se_jk > 0}].

Bootstrap draws with any non-finite entry (a household absent from a block draw) are dropped WHOLE
and counted (``n_draws_dropped``), so the studentized max is always taken over complete rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankSets:
    household_id: np.ndarray
    estimate: np.ndarray
    point_rank: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    critical_value: float
    level: float
    n_pairs_resolved: int
    n_pairs: int
    n_draws_used: int = 0
    n_draws_dropped: int = 0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "household_id": self.household_id, "estimate": self.estimate, "point_rank": self.point_rank,
                "rank_lower": self.lower, "rank_upper": self.upper, "width": self.upper - self.lower + 1,
            }
        )


def _pair_index(H: int) -> tuple[np.ndarray, np.ndarray]:
    j, k = np.triu_indices(H, k=1)
    return j, k


def rank_confidence_sets(
    estimate: np.ndarray, boot: np.ndarray, level: float = 0.95, household_id: np.ndarray | None = None,
    min_se: float = 1e-12, min_valid_share: float = 0.5,
) -> RankSets:
    """Simultaneous (1-level) rank sets from point estimates (H,) and joint bootstrap draws (B, H).

    Draws with any non-finite entry are dropped whole; fewer than ``min_valid_share`` complete draws is an error."""
    est = np.asarray(estimate, dtype=float)
    boot = np.asarray(boot, dtype=float)
    H = len(est)
    if boot.ndim != 2 or boot.shape[1] != H:
        raise ValueError("boot must be (B, H) aligned with estimate")
    if H < 2:
        raise ValueError("need at least two households")
    if not np.all(np.isfinite(est)):
        raise ValueError("estimates must be finite")
    complete = np.all(np.isfinite(boot), axis=1)
    n_used, n_dropped = int(complete.sum()), int((~complete).sum())
    if n_used < 2 or n_used < min_valid_share * len(boot):
        raise ValueError(f"only {n_used} of {len(boot)} bootstrap draws are complete")
    boot = boot[complete]
    j, k = _pair_index(H)
    D = est[j] - est[k]
    Dstar = boot[:, j] - boot[:, k]
    se = np.std(Dstar, axis=0, ddof=1)
    se = np.where(se > min_se, se, min_se)
    dev = np.abs(Dstar - D[None, :]) / se[None, :]
    max_dev = np.max(dev, axis=1)
    c = float(np.quantile(max_dev, level))
    lo_d, hi_d = D - c * se, D + c * se
    above = np.zeros(H, dtype=int)  # k confidently above j (D_jk < 0 jointly)
    below = np.zeros(H, dtype=int)  # k confidently below j
    np.add.at(above, j, hi_d < 0)
    np.add.at(below, k, hi_d < 0)
    np.add.at(below, j, lo_d > 0)
    np.add.at(above, k, lo_d > 0)
    ids = np.asarray(household_id) if household_id is not None else np.arange(H)
    order = np.argsort(-est, kind="stable")
    point_rank = np.empty(H, dtype=int)
    point_rank[order] = np.arange(1, H + 1)
    resolved = int(np.sum((hi_d < 0) | (lo_d > 0)))
    return RankSets(ids, est, point_rank, 1 + above, H - below, c, level, resolved, len(D), n_used, n_dropped)


def rank_set_summary(sets: RankSets) -> dict:
    """Median width / H, share of pairwise orderings resolved, households distinguishable from the median rank."""
    H = len(sets.estimate)
    width = sets.upper - sets.lower + 1
    med_ranks = {(H + 1) // 2, H // 2 + 1} if H % 2 == 0 else {(H + 1) // 2}
    distinguishable = np.array([all(r < lo or r > hi for r in med_ranks) for lo, hi in zip(sets.lower, sets.upper, strict=True)])
    return {
        "n_households": H,
        "critical_value": sets.critical_value,
        "median_width_over_H": float(np.median(width) / H),
        "mean_width_over_H": float(width.mean() / H),
        "share_pairs_resolved": float(sets.n_pairs_resolved / sets.n_pairs) if sets.n_pairs else np.nan,
        "n_distinguishable_from_median": int(distinguishable.sum()),
        "n_singleton_sets": int((width == 1).sum()),
        "n_full_sets": int((width == H).sum()),
    }


def true_ranks(theta: np.ndarray) -> np.ndarray:
    """Ranks of the true parameters (1 = largest; ties share the smallest rank)."""
    theta = np.asarray(theta, dtype=float)
    order = np.argsort(-theta, kind="stable")
    r = np.empty(len(theta), dtype=int)
    r[order] = np.arange(1, len(theta) + 1)
    for v in np.unique(theta):
        m = theta == v
        r[m] = r[m].min()
    return r


def joint_coverage(sets: RankSets, theta: np.ndarray) -> bool:
    """True when every household's true rank lies in its set (validated on N2 panels)."""
    r = true_ranks(theta)
    return bool(np.all((sets.lower <= r) & (r <= sets.upper)))
