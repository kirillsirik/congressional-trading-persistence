"""Multiplicity corrections from one canonical result frame (MUL-001..007).

Romano-Wolf stepdown consumes shared bootstrap draws; Holm is the analytic
cross-check; Benjamini-Hochberg is exploratory-only; the global Bonferroni
audit runs across all inference-eligible rows. The frame's content hash is
reported so post-result deletion is detectable (MUL-006).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "test_id",
    "family",
    "evidence_tier",
    "inference_eligible",
    "estimate",
    "raw_p",
)


class ResultFrameError(ValueError):
    pass


def result_frame_hash(frame: pd.DataFrame) -> str:
    canon = frame.loc[:, list(REQUIRED_COLUMNS)].sort_values("test_id")
    return hashlib.sha256(canon.to_csv(index=False).encode()).hexdigest()


def validate_result_frame(
    frame: pd.DataFrame, registry: pd.DataFrame | None = None
) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ResultFrameError(f"result frame missing columns: {missing}")
    if frame["test_id"].duplicated().any():
        dups = frame.loc[frame["test_id"].duplicated(), "test_id"].tolist()
        raise ResultFrameError(f"duplicate test ids: {dups}")
    if frame["test_id"].isna().any():
        raise ResultFrameError("missing test ids")
    if registry is not None:
        want = set(registry["test_id"])
        got = set(frame["test_id"])
        if want != got:
            raise ResultFrameError(
                f"executed tests differ from frozen registry: "
                f"missing={sorted(want - got)} extra={sorted(got - want)}"
            )


def holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * m / (rank + 1))
        adj[idx] = running
    return adj


def romano_wolf_stepdown(
    t_obs: np.ndarray, t_boot: np.ndarray
) -> np.ndarray:
    """Stepdown adjusted p-values from centered bootstrap t-statistics.

    ``t_boot`` is (n_draws, m) of null-centered statistics sharing draws
    across the family (INF-008); comparisons use absolute values (two-sided).
    """
    t_obs = np.abs(np.asarray(t_obs, dtype=float))
    t_boot = np.abs(np.asarray(t_boot, dtype=float))
    n_draws, m = t_boot.shape
    if len(t_obs) != m:
        raise ResultFrameError("t_obs and t_boot dimensions disagree")
    order = np.argsort(-t_obs)
    adj = np.empty(m)
    prev = 0.0
    for step, idx in enumerate(order):
        remaining = order[step:]
        max_null = t_boot[:, remaining].max(axis=1)
        p = (1.0 + (max_null >= t_obs[idx]).sum()) / (n_draws + 1.0)
        prev = max(prev, p)
        adj[idx] = min(1.0, prev)
    return adj


def adjust_family(
    frame: pd.DataFrame,
    registry: pd.DataFrame | None = None,
    t_boot: pd.DataFrame | None = None,
    t_col: str = "t_stat",
) -> pd.DataFrame:
    """Family and global corrections for one canonical result frame.

    ``t_boot`` columns must be test ids; Romano-Wolf is emitted per family
    only where draws cover the whole family.
    """
    validate_result_frame(frame, registry)
    out = frame.copy()
    out["p_holm"] = np.nan
    out["p_bh"] = np.nan
    out["p_romano_wolf"] = np.nan
    # robust to "yes"/"no" string flags: bare astype(bool) treats every
    # non-empty string as True and would drag descriptive/placebo rows into
    # the families and the global Bonferroni count
    eligible = (
        out["inference_eligible"]
        .map(
            lambda v: str(v).strip().lower() in ("yes", "true", "1")
            if not isinstance(v, bool)
            else v
        )
        .astype(bool)
    )
    family_groups = out.loc[eligible].groupby("family", observed=True).groups
    for _, idx in family_groups.items():
        p = out.loc[idx, "raw_p"].to_numpy(dtype=float)
        out.loc[idx, "p_holm"] = holm(p)
        out.loc[idx, "p_bh"] = benjamini_hochberg(p)
        ids = out.loc[idx, "test_id"].tolist()
        if t_boot is not None and set(ids) <= set(t_boot.columns) and t_col in out:
            out.loc[idx, "p_romano_wolf"] = romano_wolf_stepdown(
                out.loc[idx, t_col].to_numpy(dtype=float),
                t_boot.loc[:, ids].to_numpy(dtype=float),
            )
    m_global = int(eligible.sum())
    out["p_bonferroni_global"] = np.where(
        eligible, np.minimum(1.0, out["raw_p"] * m_global), np.nan
    )
    out.attrs["result_frame_sha256"] = result_frame_hash(frame)
    out.attrs["n_inference_eligible"] = m_global
    return out
