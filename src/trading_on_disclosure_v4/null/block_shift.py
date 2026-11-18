"""Block-shift placebo null (analysis plan §4, A0).

Each period's trading calendar is cut into consecutive L-session blocks by
anchor date; per draw and block one shift s ~ Uniform{+-40..+-250} sessions
moves every event in the block; shifted events without an 'ok' cache cell are
dropped; a block with > 10% infeasible events is redrawn (cap 50). N1 = raw
shifted outcomes; N2 = N1 minus each household-period's average per-draw
placebo mean over the same draw set. Seeds: seed_from(test_name, draw).

Duplicate filter (plan §11): `null.dup_rules` — ALL-LINES by default on
the public clock (`dup_primary_all_lines`), nothing on the T clock; `any_line`
and `keepdups` are labelled options. Carry `resid_sd` adds `sigma_event` =
resid_sd x sqrt(20) to every N1/N2 row (never centered).

Overlap property (referee M4): a shifted 20-session window may overlap the
actual window of another event of the same (household, PERMNO), or the shifted
window of that household-security's event in the other period. Every N1 row
carries `overlaps_actual` and `overlaps_cross_period`; the sensitivity
`exclude_overlap_with_actual=True` drops rows with `overlaps_actual` AFTER the
feasibility/redraw step (so the shift mechanism is unchanged) and logs them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.outcomes import factor_models as fm
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes.seeds import seed_from

DEFAULT_PERIODS = {"A": ("2015-01-01", "2020-12-31"), "B": ("2021-01-01", "2025-12-31")}
Y_COLUMNS = {"y_h20": "bhar_ff5mom_h20", "y_h10": "bhar_ff5mom_h10",
             "y_h60": "bhar_ff5mom_h60", "y_noalpha_h20": "bhar_ff5mom_noalpha_h20",
             "y_mkt_h20": "bhar_mkt_h20"}
CARRY_COLUMNS = {"resid_sd": "resid_sd"}  # looked up at the shifted cell, never centered
SIGMA_HORIZON = 20  # sigma_event = resid_sd x sqrt(SIGMA_HORIZON): the h20 event-level scale (T4 denominator)
PANEL_COLUMNS = ["draw", "event_id", "household_id", "clock", "period"]
FLAG_COLUMNS = ["overlaps_actual", "overlaps_cross_period"]
_KEY_BASE = 1 << 20  # > any calendar position; packs (group, anchor_idx) into one int64


@dataclass(frozen=True)
class ShiftConfig:
    test_name: str
    block_length: int = 20
    shift_min: int = 40
    shift_max: int = 250
    direction: str = "both"  # 'both' | 'backward' | 'forward'
    max_infeasible_share: float = 0.10
    max_redraws: int = 50
    outcomes: tuple[str, ...] = ("y_h20",)
    carry: tuple[str, ...] = ()  # e.g. ('resid_sd',) for the T4 z denominator
    exclude_overlap_with_actual: bool = False
    overlap_horizon: int = 20
    periods: dict = field(default_factory=lambda: dict(DEFAULT_PERIODS))

    def __post_init__(self) -> None:
        if self.direction not in ("both", "backward", "forward"):
            raise ValueError(f"unknown direction {self.direction!r}")
        if self.block_length <= 0 or self.shift_min <= 0 or self.shift_max < self.shift_min:
            raise ValueError("block_length, shift_min > 0 and shift_max >= shift_min required")
        unknown = (set(self.outcomes) - set(Y_COLUMNS)) | (set(self.carry) - set(CARRY_COLUMNS))
        if unknown:
            raise ValueError(f"unknown outcome/carry columns {sorted(unknown)}")
        if self.overlap_horizon <= 0:
            raise ValueError("overlap_horizon must be positive")


def draw_shift(rng: np.random.Generator, cfg: ShiftConfig) -> int:
    """One shift: uniform magnitude on [shift_min, shift_max], sign per ``direction``."""
    magnitude = int(rng.integers(cfg.shift_min, cfg.shift_max + 1))
    if cfg.direction == "backward":
        return -magnitude
    if cfg.direction == "forward":
        return magnitude
    return magnitude if rng.integers(0, 2) == 1 else -magnitude


class CacheGrid:
    """Dense (PERMNO x anchor position) view of the forward cache for O(1) lookups."""

    def __init__(self, cache: pd.DataFrame, outcomes: tuple[str, ...] = ("y_h20",),
                 carry: tuple[str, ...] = ()):
        cache = cache.sort_values(["permno", "anchor_idx"], ignore_index=True)
        self.permnos = np.sort(cache["permno"].unique())
        self.anchors = np.sort(cache["anchor_idx"].unique())
        self.lo, self.hi = int(self.anchors[0]), int(self.anchors[-1])
        n_p, n_a = len(self.permnos), len(self.anchors)
        dense = (len(cache) == n_p * n_a
                 and np.array_equal(self.anchors, np.arange(self.lo, self.hi + 1))
                 and cache.groupby("permno").size().eq(n_a).all()
                 and np.array_equal(cache["anchor_idx"].to_numpy().reshape(n_p, n_a)[0], self.anchors)
                 and not cache.duplicated(["permno", "anchor_idx"]).any())
        if not dense:
            raise ValueError("cache must hold every anchor exactly once for every PERMNO")
        self.ok = (cache["reason_h20"].to_numpy() == fm.REASON_OK).reshape(n_p, n_a)
        self.y = {name: cache[Y_COLUMNS[name]].to_numpy(float).reshape(n_p, n_a) for name in outcomes}
        self.y.update({name: cache[CARRY_COLUMNS[name]].to_numpy(float).reshape(n_p, n_a) for name in carry})
        self.dates = cache["anchor_date"].iloc[:n_a].to_numpy()

    def rows(self, permno: np.ndarray) -> np.ndarray:
        """Row index per PERMNO (-1 when the PERMNO is not in the cache)."""
        pos = np.searchsorted(self.permnos, permno)
        pos = np.clip(pos, 0, len(self.permnos) - 1)
        return np.where(self.permnos[pos] == permno, pos, -1)

    def feasible(self, row: np.ndarray, idx: np.ndarray) -> np.ndarray:
        inside = (row >= 0) & (idx >= self.lo) & (idx <= self.hi)
        col = np.where(inside, idx - self.lo, 0)
        return inside & self.ok[np.where(inside, row, 0), col]

    def lookup(self, name: str, row: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return self.y[name][row, idx - self.lo]


def assign_blocks(
    events: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: ShiftConfig
) -> pd.DataFrame:
    """Add ``anchor_idx`` and ``block`` (consecutive L-session blocks per period)."""
    out = events.copy()
    idx = calendar.get_indexer(pd.DatetimeIndex(pd.to_datetime(out["anchor_date"])))
    if (idx < 0).any():
        raise ValueError(f"{int((idx < 0).sum())} anchor dates are not calendar sessions")
    out["anchor_idx"] = idx.astype(np.int64)
    starts = {p: int(calendar.searchsorted(pd.Timestamp(lo), side="left")) for p, (lo, _) in cfg.periods.items()}
    unknown = set(out["period"]) - set(starts)
    if unknown:
        raise ValueError(f"events carry periods without bounds: {sorted(unknown)}")
    start = out["period"].map(starts).to_numpy(np.int64)
    out["block"] = (out["anchor_idx"].to_numpy() - start) // cfg.block_length
    return out


def _block_groups(events: pd.DataFrame) -> list[tuple[str, int, np.ndarray]]:
    keys = events.groupby(["period", "block"], sort=True).indices
    return [(p, int(b), np.asarray(ix)) for (p, b), ix in sorted(keys.items())]


def _overlaps(sorted_keys: np.ndarray, group: np.ndarray, idx: np.ndarray, h: int) -> np.ndarray:
    """True where some packed key lies within +-(h-1) sessions of (group, idx): windows overlap."""
    lo = np.searchsorted(sorted_keys, group * _KEY_BASE + idx - (h - 1), side="left")
    hi = np.searchsorted(sorted_keys, group * _KEY_BASE + idx + (h - 1), side="right")
    return hi > lo


class Geometry:
    """Per-event block groups, cache rows and the (household, PERMNO) actual-anchor index."""

    def __init__(self, events: pd.DataFrame, grid: CacheGrid):
        self.groups = _block_groups(events)
        self.rows = grid.rows(events["permno"].to_numpy(np.int64))
        self.idx = events["anchor_idx"].to_numpy(np.int64)
        key = events["household_id"].astype(str) + "|" + events["permno"].astype(str)
        self.hh_sec = pd.factorize(key)[0].astype(np.int64)
        self.actual_keys = np.sort(self.hh_sec * _KEY_BASE + self.idx)
        self.period = events["period"].to_numpy()

    def cross_period(self, keep: np.ndarray, shifted: np.ndarray, h: int) -> np.ndarray:
        """Kept rows whose shifted window overlaps a kept shifted window of the same
        household-security in ANOTHER period of the same draw."""
        out = np.zeros(len(keep), dtype=bool)
        for p in np.unique(self.period[keep]):
            mine = keep & (self.period == p)
            other = keep & (self.period != p)
            if other.any():
                keys = np.sort(self.hh_sec[other] * _KEY_BASE + shifted[other])
                out[mine] = _overlaps(keys, self.hh_sec[mine], shifted[mine], h)
        return out


def shift_one_draw(
    events: pd.DataFrame, grid: CacheGrid, cfg: ShiftConfig, draw: int,
    geom: Geometry | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One placebo panel (N1 rows) and its per-block drop log for ``draw``."""
    rng = np.random.default_rng(seed_from(cfg.test_name, draw))
    g = Geometry(events, grid) if geom is None else geom
    keep = np.zeros(len(events), dtype=bool)
    shifted = np.zeros(len(events), dtype=np.int64)
    log = []
    for period, block, ix in g.groups:
        n, n_redraws = len(ix), 0
        while True:
            s = draw_shift(rng, cfg)
            new_idx = g.idx[ix] + s
            feas = grid.feasible(g.rows[ix], new_idx)
            n_inf = int(n - feas.sum())
            if n_inf <= cfg.max_infeasible_share * n or n_redraws >= cfg.max_redraws:
                break
            n_redraws += 1
        keep[ix] = feas
        shifted[ix] = new_idx
        log.append({"draw": draw, "period": period, "block": block, "shift": s,
                    "n_events": n, "n_infeasible": n_inf, "n_redraws": n_redraws,
                    "cap_hit": n_inf > cfg.max_infeasible_share * n, "n_overlap_dropped": 0})
    return _finish_draw(events, grid, cfg, draw, g, keep, shifted, pd.DataFrame(log))


def _finish_draw(
    events: pd.DataFrame, grid: CacheGrid, cfg: ShiftConfig, draw: int, g: Geometry,
    keep: np.ndarray, shifted: np.ndarray, log: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overlap flags, the optional overlap exclusion (logged per block) and the N1 frame."""
    h = cfg.overlap_horizon
    ov_actual = keep & _overlaps(g.actual_keys, g.hh_sec, shifted, h)
    if cfg.exclude_overlap_with_actual and ov_actual.any():
        dropped = pd.DataFrame({"period": g.period[ov_actual],
                                "block": events["block"].to_numpy()[ov_actual]})
        counts = dropped.groupby(["period", "block"]).size()
        log["n_overlap_dropped"] = [int(counts.get((p, b), 0))
                                    for p, b in zip(log["period"], log["block"], strict=True)]
        keep = keep & ~ov_actual
    ov_cross = g.cross_period(keep, shifted, h)
    kept = events.loc[keep, PANEL_COLUMNS[1:]].reset_index(drop=True)
    kept.insert(0, "draw", np.int32(draw))
    r, i = g.rows[keep], shifted[keep]
    for name in cfg.outcomes + cfg.carry:
        kept[name] = grid.lookup(name, r, i)
    if "resid_sd" in cfg.carry:
        kept["sigma_event"] = kept["resid_sd"] * np.sqrt(SIGMA_HORIZON)
    kept["shifted_anchor_date"] = grid.dates[i - grid.lo]
    kept["overlaps_actual"] = ov_actual[keep]
    kept["overlaps_cross_period"] = ov_cross[keep]
    return kept, log


def generate_draws(
    events: pd.DataFrame, grid: CacheGrid, cfg: ShiftConfig, draws: range | list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """N1 panel and drop log for several draws (events already block-assigned)."""
    geom = Geometry(events, grid)
    panels, logs = [], []
    for d in draws:
        p, lg = shift_one_draw(events, grid, cfg, int(d), geom)
        panels.append(p)
        logs.append(lg)
    return pd.concat(panels, ignore_index=True), pd.concat(logs, ignore_index=True)


HH_KEY = ["household_id", "clock", "period"]


def _per_draw_sums(panel: pd.DataFrame, outcomes: tuple[str, ...]) -> pd.DataFrame:
    """Per (household, clock, period, draw): sum and count of each outcome (the accumulator)."""
    g = panel.groupby(HH_KEY + ["draw"], sort=True)
    sums = g[list(outcomes)].sum(min_count=1)
    sums.columns = [f"sum_{c}" for c in outcomes]
    for c in outcomes:  # non-NaN counts, so sum/count equals the per-draw nan-skipping mean
        sums[f"cnt_{c}"] = g[c].count()
    sums["n_events"] = g.size()
    return sums.reset_index()


def centering_from_sums(sums: pd.DataFrame, outcomes: tuple[str, ...]) -> pd.DataFrame:
    """Household-period centering means from accumulated per-draw sums (batch-safe)."""
    per_draw = sums.copy()
    for c in outcomes:
        per_draw[c] = per_draw[f"sum_{c}"] / per_draw[f"cnt_{c}"].where(per_draw[f"cnt_{c}"] > 0)
    means = per_draw.groupby(HH_KEY, sort=True)[list(outcomes)].mean()
    means.columns = [f"center_{c}" for c in outcomes]
    means["n_draws_present"] = per_draw.groupby(HH_KEY).size()
    means["mean_events_per_draw"] = per_draw.groupby(HH_KEY)["n_events"].mean()
    return means.reset_index()


def centering_means(panel: pd.DataFrame, outcomes: tuple[str, ...]) -> pd.DataFrame:
    """Per household-period: average over draws of the per-draw placebo mean (single frame)."""
    return centering_from_sums(_per_draw_sums(panel, outcomes), outcomes)


def center_panel(panel: pd.DataFrame, means: pd.DataFrame, outcomes: tuple[str, ...]) -> pd.DataFrame:
    """N2: subtract the household-period centering means from every outcome column."""
    out = panel.merge(means[HH_KEY + [f"center_{c}" for c in outcomes]], on=HH_KEY, how="left")
    for c in outcomes:
        out[c] = out[c] - out.pop(f"center_{c}")
    return out


def run_batches(
    events: pd.DataFrame, grid: CacheGrid, cfg: ShiftConfig, n_draws: int,
    out_dir, batch_size: int = 200, write_n2: bool = True, log_path: Path | None = None,
    extra_summary: dict | None = None,
) -> dict:
    """Write N1 (and N2) panels in batches of draws plus the drop log; count-only summary.

    Memory: the centering means come from per-batch (household, period, draw)
    sum/count accumulators, so no batch is ever concatenated with another.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    geom = Geometry(events, grid)
    logs, n1_paths, sums, flag_counts, n_rows = [], [], [], np.zeros(len(FLAG_COLUMNS)), 0
    for start in range(0, n_draws, batch_size):
        panels = []
        for d in range(start, min(start + batch_size, n_draws)):
            p, lg = shift_one_draw(events, grid, cfg, d, geom)
            panels.append(p)
            logs.append(lg)
        batch = pd.concat(panels, ignore_index=True)
        path = out_dir / f"n1_batch_{start // batch_size:04d}.parquet"
        batch.to_parquet(path, index=False)
        n1_paths.append(path)
        sums.append(_per_draw_sums(batch, cfg.outcomes))
        flag_counts += batch[FLAG_COLUMNS].sum().to_numpy(float)
        n_rows += len(batch)
        if log_path is not None:
            fc._log(log_path, f"{cfg.test_name}: batch {start // batch_size} draws {start}-{d} "
                              f"({len(batch)} rows) elapsed {time.time() - t0:.1f}s")
    droplog = pd.concat(logs, ignore_index=True)
    droplog.to_parquet(out_dir / "droplog.parquet", index=False)
    means = centering_from_sums(pd.concat(sums, ignore_index=True), cfg.outcomes)
    means.to_parquet(out_dir / "centering_means.parquet", index=False)
    if write_n2:
        for p in n1_paths:
            center_panel(pd.read_parquet(p), means, cfg.outcomes).to_parquet(
                out_dir / p.name.replace("n1_", "n2_"), index=False)
    flags = {c: float(v / n_rows) if n_rows else 0.0 for c, v in zip(FLAG_COLUMNS, flag_counts, strict=True)}
    summary = summarize_run(events, n_rows, droplog, cfg, time.time() - t0, flag_shares=flags)
    summary.update(extra_summary or {})
    return summary


def _shift_stats(log: pd.DataFrame) -> dict:
    a = log["shift"].abs()
    return {"shift_abs_mean": float(a.mean()),
            "shift_abs_p10_p50_p90": [float(x) for x in np.percentile(a, [10, 50, 90])],
            "shift_positive_share": float((log["shift"] > 0).mean()),
            "blocks_with_redraw_share": float((log["n_redraws"] > 0).mean()),
            "n_draw_blocks": int(len(log))}


def summarize_run(
    events: pd.DataFrame, n1: pd.DataFrame | int, droplog: pd.DataFrame, cfg: ShiftConfig,
    seconds: float, flag_shares: dict | None = None,
) -> dict:
    """Count-only summary of a draw set (safe to log); ``n1`` may be the panel or its row count."""
    if isinstance(n1, pd.DataFrame):
        flag_shares = {c: float(n1[c].mean()) for c in FLAG_COLUMNS if c in n1.columns}
        n1 = len(n1)
    n_draws = int(droplog["draw"].nunique())
    by_period = droplog.groupby("period").agg(n_events=("n_events", "sum"), n_infeasible=("n_infeasible", "sum"),
                                              n_overlap_dropped=("n_overlap_dropped", "sum"))
    return {
        "test_name": cfg.test_name, "block_length": cfg.block_length, "direction": cfg.direction,
        "exclude_overlap_with_actual": cfg.exclude_overlap_with_actual,
        "overlap_horizon": cfg.overlap_horizon, "outcomes": list(cfg.outcomes), "carry": list(cfg.carry),
        "n_draws": n_draws, "n_events_in": int(len(events)),
        "n_households_in": int(events["household_id"].nunique()),
        "n_blocks": int(droplog.groupby(["period", "block"]).ngroups),
        "rows_per_draw_mean": float(n1 / n_draws) if n_draws else 0.0,
        "drop_rate_overall": float(droplog["n_infeasible"].sum() / droplog["n_events"].sum()),
        "drop_rate_by_period": {p: float(r.n_infeasible / r.n_events) for p, r in by_period.iterrows()},
        "overlap_drop_rate_overall": float(droplog["n_overlap_dropped"].sum() / droplog["n_events"].sum()),
        "overlap_drop_rate_by_period": {p: float(r.n_overlap_dropped / r.n_events) for p, r in by_period.iterrows()},
        "redraws_total": int(droplog["n_redraws"].sum()),
        "cap_hit_blocks": int(droplog["cap_hit"].sum()),
        **_shift_stats(droplog),
        "shift_by_period": {p: _shift_stats(g) for p, g in droplog.groupby("period", sort=True)},
        "n1_flag_shares": flag_shares or {},
        "seconds": round(seconds, 2),
    }


def test_name_for(clock: str, directions: str, periods: str, block_length: int, direction: str,
                  drop_duplicates: bool, drop_duplicates_on_T: bool, exclude_overlap: bool,
                  dup_rule: str = "all_lines", tag: str = "") -> str:
    """`A0_<clock>_<dirs>_<periods>_L<L>_<shift dir>_<alllines|nodup|keepdups>[_xovl][_<tag>]`
    (seeds derive from this name)."""
    from trading_on_disclosure_v4.null import dup_rules as dr

    rule = dr.rule_from_switches(drop_duplicates, dup_rule)
    name = (f"A0_{clock}_{directions.replace(',', '+')}_{periods.replace(',', '')}"
            f"_L{block_length}_{direction}{dr.dup_suffix(clock, rule, drop_duplicates_on_T)}")
    return name + ("_xovl" if exclude_overlap else "") + (f"_{tag}" if tag else "")


def load_events_for_null(clock: str, directions: str = "purchase", periods: str = "A,B",
                         drop_duplicates: bool = True, drop_duplicates_on_T: bool = False,
                         cfg: ShiftConfig | None = None, dup_rule: str = "all_lines") -> tuple[pd.DataFrame, dict]:
    """D's events filtered on direction/period, duplicate-filtered (plan §11 default: all-lines rule
    on P, nothing on T), block-assigned; count-only report with both rules' flag counts."""
    from trading_on_disclosure_v4.null import dup_rules as dr
    from trading_on_disclosure_v4.outcomes import event_filters as ef
    from trading_on_disclosure_v4.outcomes import panel

    ev = pd.read_parquet(ef.EVENTS_DIR / f"events_{clock}.parquet")
    ev = ev[ev["direction"].isin(directions.split(",")) & ev["period"].isin(periods.split(","))]
    rule = dr.rule_from_switches(drop_duplicates, dup_rule)
    ev, report = dr.filter_events(ev.reset_index(drop=True), clock, rule, drop_duplicates_on_T)
    if cfg is not None:
        ev = assign_blocks(ev, panel.v2_trading_calendar(), cfg)
    return ev, report


def build_parser():
    import argparse

    ap = argparse.ArgumentParser(description="block-shift null panels (A0)")
    ap.add_argument("--clock", choices=("T", "P"), required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--block-length", type=int, default=20)
    ap.add_argument("--direction", choices=("both", "backward", "forward"), default="both")
    ap.add_argument("--outcomes", default="y_h20")
    ap.add_argument("--carry", default="", help="comma list of carry columns, e.g. resid_sd")
    ap.add_argument("--directions", default="purchase")
    ap.add_argument("--periods", default="A,B")
    ap.add_argument("--keep-duplicates", action="store_true", help="do not drop dup_primary events")
    ap.add_argument("--dup-rule", choices=("all_lines", "any_line"), default="all_lines",
                    help="which flag marks a duplicate event (plan §11 default: all_lines)")
    ap.add_argument("--drop-duplicates-on-T", action="store_true", help="also drop flagged T episodes")
    ap.add_argument("--exclude-overlap-with-actual", action="store_true")
    ap.add_argument("--log", type=Path, default=None)
    return ap


def main(argv: list[str] | None = None) -> dict:
    """CLI: production draw set for one clock on D's event tables; prints the count-only summary."""
    import json

    from trading_on_disclosure_v4.outcomes import panel

    args = build_parser().parse_args(argv)
    drop = not args.keep_duplicates
    name = test_name_for(args.clock, args.directions, args.periods, args.block_length, args.direction,
                         drop, args.drop_duplicates_on_T, args.exclude_overlap_with_actual, args.dup_rule)
    cfg = ShiftConfig(test_name=name, block_length=args.block_length, direction=args.direction,
                      outcomes=tuple(args.outcomes.split(",")),
                      carry=tuple(c for c in args.carry.split(",") if c),
                      exclude_overlap_with_actual=args.exclude_overlap_with_actual)
    cols = ["permno", "anchor_idx", "anchor_date", "reason_h20"] + [Y_COLUMNS[o] for o in cfg.outcomes]
    cols += [CARRY_COLUMNS[c] for c in cfg.carry]
    grid = CacheGrid(fc.load_cache(columns=cols), outcomes=cfg.outcomes, carry=cfg.carry)
    evb, report = load_events_for_null(args.clock, args.directions, args.periods, drop,
                                       args.drop_duplicates_on_T, cfg, args.dup_rule)
    out_dir = panel.RESTRICTED_N / "null" / name
    summary = run_batches(evb, grid, cfg, n_draws=args.draws, out_dir=out_dir, batch_size=args.batch_size,
                          log_path=args.log, extra_summary={"duplicate_filter": report})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
