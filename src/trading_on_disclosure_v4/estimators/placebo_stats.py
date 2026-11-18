"""Per-draw statistics on the placebo datasets (plan §5): the same panel transformation and the same
statistic as on actual outcomes, once per stored draw, cached under data/v4/restricted/e/placebo_stats/.

One table per (clock, flavour, placebo set): columns `draw, key, stat, value`, where `key` is the
spec's cache key (`spec_key`) — a spec whose definition changes gets a new key and is recomputed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import subperiods as sub_mod
from . import tau as tau_mod
from . import tests_t1_t4 as tt
from . import winners
from .specs import FULL_MIN_N, Spec, to_panel

PLACEBO_KINDS = ("tau", "mean", "best", "tail", "subperiod")
STAT_NAMES = {"tau": ("tau2",), "mean": ("mean",), "best": ("max_mean", "max_t", "top_decile_mean", "n_p05"),
              "tail": (*(f"share_gt_{z:g}" for z in tt.Z_THRESHOLDS), *(tt.lower_tail_stat(z) for z in tt.Z_THRESHOLDS)),
              "subperiod": sub_mod.SUBPERIOD_STATS}
#: kinds whose statistic needs the trading calendar (the 20-session completion rule)
CALENDAR_KINDS = ("subperiod",)
CutoffFn = Callable[[Spec], dict | None]
FrameSource = Callable[[str, str, str], Iterable[tuple[int, pd.DataFrame]]]  # (clock, flavour, dup) -> (batch, frame)


def spec_key(spec: Spec) -> str:
    return "|".join([spec.name, spec.panel_key, spec.estimator_key])


def statistic(panel: pd.DataFrame, spec: Spec, calendar: pd.DatetimeIndex | None = None) -> dict[str, float]:
    """The statistics whose placebo distribution a spec needs, on one panel (NaN when not computable, so the
    draw is recorded and the cache stays complete)."""
    nan = {k: np.nan for k in STAT_NAMES.get(spec.kind, ())}
    try:
        if spec.kind == "tau":
            table = tau_mod.household_table(panel, spec.weights, residualize_activity=spec.residualize_activity)
            return {"tau2": tau_mod.tau2_household_covariance(table)}
        if spec.kind == "subperiod":
            if calendar is None:
                raise ValueError("the subperiod statistic needs the trading calendar")
            s = sub_mod.subperiod_statistics(panel, calendar)
            return {k: s[k] for k in STAT_NAMES["subperiod"]}
        if spec.kind == "mean":
            y = panel["y_h20"].to_numpy(dtype=float)
            return {"mean": float(np.mean(y)) if len(y) else np.nan}
        if spec.kind == "best":
            w = winners.panel_winner_stats(panel, min_n=FULL_MIN_N)
            return {"max_mean": w["max_mean"], "max_t": w["max_t"], "top_decile_mean": w["top_decile_mean"],
                    "n_p05": float(w["n_nominal_significant"])}
        if spec.kind == "tail":
            ok = np.isfinite(panel["sigma_event"].to_numpy(dtype=float))
            s = tt.t4_tail_shares(panel.loc[ok])
            return {k: s[k] for k in STAT_NAMES["tail"]}
    except (ValueError, KeyError):
        return nan
    return nan


def draw_statistics(frame: pd.DataFrame, specs: list[Spec], cutoffs: CutoffFn, draw: int,
                    calendar: pd.DatetimeIndex | None = None) -> list[dict]:
    """Rows {draw, key, stat, value} for every spec on one draw; one panel per distinct panel_key."""
    rows = []
    by_panel: dict[str, list[Spec]] = {}
    for s in specs:
        by_panel.setdefault(s.panel_key, []).append(s)
    for group in by_panel.values():
        panel = to_panel(frame, group[0], cutoffs(group[0]), draw=draw)
        for s in group:
            for stat, v in statistic(panel, s, calendar).items():
                rows.append({"draw": int(draw), "key": spec_key(s), "stat": stat, "value": float(v)})
    return rows


def cache_path(cache_dir: Path, clock: str, flavour: str, dup: str) -> Path:
    return Path(cache_dir) / f"placebo_{clock}_{flavour}_{'keepdups' if dup == 'keepdups' else 'default'}.parquet"


def placebo_table(clock: str, flavour: str, specs: list[Spec], cutoffs: CutoffFn, source: FrameSource,
                  cache_dir: Path | None, n_draws: int = 2000, dup: str = "default", log=print,
                  calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Per-draw statistics for `specs`, read from the cache where complete (every statistic of the spec's
    kind on every draw), computed over the stored batches otherwise (only the missing specs, every
    statistic), and appended to the cache."""
    specs = [s for s in specs if s.has_placebo and s.kind in PLACEBO_KINDS]
    keys = {spec_key(s) for s in specs}
    path = cache_path(cache_dir, clock, flavour, dup) if cache_dir is not None else None
    cached = pd.read_parquet(path) if path is not None and path.exists() else pd.DataFrame(columns=["draw", "key", "stat", "value"])
    have = cached.groupby(["key", "stat"])["draw"].nunique() if len(cached) else pd.Series(dtype=int)
    missing = [s for s in specs if any(have.get((spec_key(s), st), 0) < n_draws for st in STAT_NAMES[s.kind])]
    if calendar is None and any(s.kind in CALENDAR_KINDS for s in missing):
        raise ValueError("the placebo pass needs the trading calendar for " + ", ".join(sorted(CALENDAR_KINDS)))
    if missing:
        t0 = time.time()
        rows: list[dict] = []
        n_seen = 0
        for _, batch in source(clock, flavour, dup):
            for draw, g in batch.groupby("draw", sort=True):
                if draw >= n_draws:
                    continue
                rows += draw_statistics(g.reset_index(drop=True), missing, cutoffs, int(draw), calendar)
                n_seen += 1
            log(f"placebo {clock}/{flavour}/{dup}: {n_seen} draws, {len(missing)} specs, {time.time() - t0:.0f} s")
        new = pd.DataFrame(rows, columns=["draw", "key", "stat", "value"])
        kept = cached.loc[~cached["key"].isin({spec_key(s) for s in missing})]
        cached = pd.concat([kept, new], ignore_index=True) if len(kept) else new
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            cached.to_parquet(path, index=False)
        log(f"placebo {clock}/{flavour}/{dup}: {len(missing)} specs x {n_seen} draws in {time.time() - t0:.0f} s")
    return cached.loc[cached["key"].isin(keys) & (cached["draw"] < n_draws)].reset_index(drop=True)


def values(table: pd.DataFrame, spec: Spec, stat: str) -> np.ndarray:
    """The per-draw vector of one statistic for one spec (finite values only, draw order)."""
    t = table.loc[(table["key"] == spec_key(spec)) & (table["stat"] == stat)].sort_values("draw")
    v = t["value"].to_numpy(dtype=float)
    return v[np.isfinite(v)]
