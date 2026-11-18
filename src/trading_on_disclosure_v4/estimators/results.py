"""Household estimates on one event frame per clock -> the tidy rows of data/v4/results.csv.
Called by scripts/v4/run_results.py, or by tests on synthetic frames.

A `Context` holds the event frames, the trading calendar, the winsorization cutoffs, and a source of
placebo frames (N's block-shift draws joined to the event metadata). Every placebo p-value is the rank of
the actual statistic among the 2,000 placebo values of the same statistic; intervals and the upper bound
on tau come from the household bootstrap; every mean or slope carries a household x 20-trading-day-block
clustered SE. Seeds: `seed_from(spec.name, clock, purpose)` (`outcomes/seeds.py`).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..inference import block_bootstrap as bb
from ..inference import cgm
from ..inference.holm import holm_adjust
from ..inference.placebo import placebo_p_value
from ..outcomes.seeds import seed_from
from . import continuation as cont_mod
from . import eb as eb_mod
from . import no_control as nc
from . import placebo_stats as ps
from . import ranks as ranks_mod
from . import subperiods as sub_mod
from . import tau as tau_mod
from . import tests_t1_t4 as tt
from . import ttd as ttd_mod
from . import winners
from .specs import (
    ANCHOR_WINDOWS,
    CONTINUATION,
    EXCL_COVID,
    FULL_MIN_N,
    HOUSEHOLD_SCATTER,
    OUTCOMES,
    ROBUSTNESS,
    SAMPLE_VARIANTS,
    SUBPERIOD,
    TAIL_VARIANTS,
    TAU_NO_CONTROL,
    WEIGHT_NAMES,
    Spec,
    analysis_specs,
    continuation_rate,
    headline_spec,
    to_panel,
)

BLOCK_L = 20
N_BOOT = 4999
N_PLACEBO = 2000
SPAN_YEARS = 11  # anchors 2015-2025
COLUMNS = ("clock", "analysis", "variant", "statistic", "value", "unit", "placebo_p", "boot_lo", "boot_hi", "se_block",
           "n_events", "n_households", "notes")
RANK_COLUMNS = ("clock", "household_index", "point_rank", "estimate", "rank_lower", "rank_upper", "width", "n_events")
SCATTER_COLUMNS = ("clock", "household_index", "x", "y", "w")
TTD_COLUMNS = ("clock", "theta_annual_pct", "theta_20_bp", "n_per_year", "sigma_bp", "design_effect", "years", "years_best_of_h", "n_households")
FLAVOUR_VARIANT = {"N2": "vs_centered_placebo", "N1": "vs_timing_placebo"}
COVID_NOTE = "anchors dated February-May 2020 excluded (the shifted anchor on a placebo draw)"
SAMPLE_NOTES = {"official_only": "official filings only",
                "excl_two_vendor_households": "without the two households of the no-parsed-official-filing class with the most vendor lines",
                "period_A": "period A (2015-2020) only", "period_B": "period B (2021-2025) only",
                EXCL_COVID: COVID_NOTE, f"{EXCL_COVID}_period_A": f"period A (2015-2020) only; {COVID_NOTE}",
                f"{EXCL_COVID}_period_B": f"period B (2021-2025) only; {COVID_NOTE}"}
ADDED = "added after results, descriptive"
# the calibration's choice table: the bootstrap level the calibrated upper bound uses, per clock
CALIBRATION_CHOICE = Path(__file__).resolve().parents[3] / "data" / "v4" / "calibration" / "upper_bound_calibration_choice.csv"
CALIBRATION_TARGET = ("realized_dispersion", "before_winsor")
# the inverted placebo-injection upper bound on tau (calibration report section 13), one row per
#: (clock, weights, outcome label); `outcome` labels -> the Spec outcome they were computed on
INVERTED_BOUND = CALIBRATION_CHOICE.with_name("inverted_bound.csv")
INJECTION_OUTCOMES = {"headline": "ff5mom_h20", "market_adjusted": "mkt_h20", "no_alpha": "ff5mom_noalpha_h20"}
INJECTION_TOL = 1e-6  # relative tolerance between the file's observed tau2 and this run's tau2_hat
FrameSource = Callable[[str, str, str], Iterable[tuple[int, pd.DataFrame]]]


def row(clock: str, analysis: str, variant: str, statistic: str, value, unit: str, placebo_p=np.nan, boot_lo=np.nan,
        boot_hi=np.nan, se_block=np.nan, n_events=np.nan, n_households=np.nan, notes: str = "") -> dict:
    return dict(clock=clock, analysis=analysis, variant=variant, statistic=statistic, value=value, unit=unit, placebo_p=placebo_p,
                boot_lo=boot_lo, boot_hi=boot_hi, se_block=se_block, n_events=n_events, n_households=n_households, notes=notes)


def read_calibrated_levels(path: Path = CALIBRATION_CHOICE, target: str = CALIBRATION_TARGET[0],
                           injection: str = CALIBRATION_TARGET[1]) -> dict[str, float]:
    """{clock: calibrated bootstrap level} from the calibration's choice table:
    the smallest grid level with >= 95% coverage at every injected tau for the named target and
    injection order. A clock whose rule selected no level is absent."""
    t = pd.read_csv(path)
    t = t.loc[(t["target"] == target) & (t["injection"] == injection)]
    out = {}
    for r in t.itertuples(index=False):
        if pd.notna(r.calibrated_level):
            out[str(r.clock)] = float(r.calibrated_level)
    return out


def read_inverted_bounds(path: Path = INVERTED_BOUND) -> pd.DataFrame:
    """The inverted placebo-injection bounds (calibration report section 13), indexed by (clock, weights, outcome
    label). Every key must be unique and every outcome label known."""
    t = pd.read_csv(path)
    keys = ["clock", "weights", "outcome"]
    if t.duplicated(keys).any():
        raise ValueError(f"{path}: duplicate (clock, weights, outcome) rows")
    unknown = sorted(set(t["outcome"]) - set(INJECTION_OUTCOMES))
    if unknown:
        raise ValueError(f"{path}: unknown outcome labels {unknown}")
    return t.set_index(keys).sort_index()


def results_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    for c in ("value", "placebo_p", "boot_lo", "boot_hi", "se_block"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
    for c in ("n_events", "n_households"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype("Int64")
    return df


@dataclass
class Context:
    frames: dict[str, pd.DataFrame]
    calendar: pd.DatetimeIndex
    cutoffs: Callable[[str, str], dict[str, tuple[float, float]]]  # (clock, outcome) -> committed purchase cutoffs
    placebo_source: FrameSource  # (clock, flavour, dup) -> (batch, frame)
    placebo_set_name: Callable[[str, str], str] = lambda clock, dup: dup  # (clock, dup) -> stored set (T: keepdups == default)
    n_boot: int = N_BOOT
    n_placebo: int = N_PLACEBO
    cache_dir: Path | None = None
    log: Callable[[str], None] = print
    calibrated_levels: dict[str, float] = field(default_factory=dict)  # clock -> bootstrap level of the calibrated upper bound
    inverted_bounds: pd.DataFrame | None = None  # `read_inverted_bounds()`; None = no injection-bound rows
    _placebo: dict = field(default_factory=dict)
    _cutoffs: dict = field(default_factory=dict)
    rank_rows: list[dict] = field(default_factory=list)
    ttd_rows: list[dict] = field(default_factory=list)
    scatter_rows: list[dict] = field(default_factory=list)

    def cutoffs_for(self, spec: Spec) -> dict | None:
        if not spec.winsorized:
            return None
        if spec.direction == "purchase":
            k = (spec.clock, spec.outcome)
            if k not in self._cutoffs:
                self._cutoffs[k] = self.cutoffs(spec.clock, spec.outcome)
            return self._cutoffs[k]
        from .event_frames import sample_cutoffs
        from .specs import apply_direction

        f = apply_direction(self.frames[spec.clock], spec.direction, OUTCOMES[spec.outcome])
        return sample_cutoffs(f[OUTCOMES[spec.outcome]].to_numpy(dtype=float), f["period"].to_numpy())

    def panel(self, spec: Spec) -> pd.DataFrame:
        return to_panel(self.frames[spec.clock], spec, self.cutoffs_for(spec))

    def placebo(self, spec: Spec, flavour: str, stat: str) -> np.ndarray:
        """Per-draw placebo values of `stat` for `spec` (all specs of the clock are computed in one pass)."""
        set_name = self.placebo_set_name(spec.clock, spec.dup)
        dup = "default" if set_name == self.placebo_set_name(spec.clock, "default") else spec.dup
        key = (spec.clock, flavour, set_name)
        if key not in self._placebo:
            specs = [s for s in analysis_specs(spec.clock).values() if self.placebo_set_name(spec.clock, s.dup) == set_name]
            self._placebo[key] = ps.placebo_table(spec.clock, flavour, specs, self.cutoffs_for, self.placebo_source,
                                                  self.cache_dir, self.n_placebo, dup, self.log, self.calendar)
        table = self._placebo[key]
        if spec.has_placebo and spec.kind in ps.PLACEBO_KINDS and ps.spec_key(spec) not in set(table["key"]):
            extra = ps.placebo_table(spec.clock, flavour, [spec], self.cutoffs_for, self.placebo_source, self.cache_dir,
                                     self.n_placebo, dup, self.log, self.calendar)
            table = self._placebo[key] = pd.concat([table, extra], ignore_index=True)
        return ps.values(table, spec, stat)


# ---------------------------------------------------------------- helpers

def block_cluster(panel: pd.DataFrame, calendar: pd.DatetimeIndex, L: int = BLOCK_L) -> pd.DataFrame:
    """Add `anchor_blk` = calendar position // L (the 20-trading-day block of the anchor)."""
    pos = calendar.get_indexer(pd.DatetimeIndex(panel["anchor_date"]))
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} anchors are not calendar sessions")
    return panel.assign(anchor_blk=pos // L)


def _counts(panel: pd.DataFrame) -> tuple[int, int]:
    return int(len(panel)), int(panel["household_id"].nunique())


def _placebo_rows(ctx: Context, spec: Spec, analysis: str, variant: str, stat: str, actual: float, unit: str,
                  flavour: str, alternative: str, n_events: int, n_households: int, extra_notes: str = "") -> list[dict]:
    """The statistic with its placebo p plus the placebo mean / 95th percentile rows."""
    vals = ctx.placebo(spec, flavour, stat)
    if len(vals) == 0 or not np.isfinite(actual):
        return [row(spec.clock, analysis, variant, stat, actual, unit, n_events=n_events, n_households=n_households,
                    notes=f"no placebo values ({extra_notes})".strip())]
    pl = placebo_p_value(actual, vals, alternative)
    notes = f"{alternative}; {pl['n_null']} placebo draws" + (f"; {extra_notes}" if extra_notes else "")
    return [row(spec.clock, analysis, variant, stat, actual, unit, placebo_p=pl["p"], n_events=n_events, n_households=n_households, notes=notes),
            row(spec.clock, analysis, variant, f"{stat}_placebo_mean", pl["null_mean"], unit, n_events=n_events, n_households=n_households),
            row(spec.clock, analysis, variant, f"{stat}_placebo_q95", pl["null_q95"], unit, n_events=n_events, n_households=n_households)]


# ---------------------------------------------------------------- analyses

def run_tau(ctx: Context, spec: Spec, analysis: str, variant: str) -> list[dict]:
    """tau2 / tau with block-clustered SE, household-bootstrap interval and upper bound, centered placebo p."""
    c = spec.clock
    panel = ctx.panel(spec)
    if panel.empty:
        return [row(c, analysis, variant, "tau2", np.nan, "bp2", n_events=0, n_households=0, notes="no events for this variant")]
    pb = block_cluster(panel, ctx.calendar)
    seed = seed_from(spec.name, c, "household bootstrap")
    if spec.residualize_activity:
        r = cgm.tau2_i1(pb, spec.weights, anchor="anchor_blk", activity_control=spec.activity_control)
        rd = cgm.tau2_i1(pb, spec.weights, anchor="anchor_date", activity_control=spec.activity_control)
        b = bb.tau2_household_bootstrap(panel, spec.weights, ctx.n_boot, seed)
        control_note = f"activity control {spec.activity_control}"
    else:
        r = nc.tau2_no_control_i1(pb, spec.weights, anchor="anchor_blk")
        rd = nc.tau2_no_control_i1(pb, spec.weights, anchor="anchor_date")
        b = nc.tau2_no_control_household_bootstrap(panel, spec.weights, ctx.n_boot, seed)
        control_note = "no activity control at either stage"
    n_ev, n_hh = int(r["n_events_a"] + r["n_events_b"]), int(r["n_households"])
    se_tau2 = float(r["var_wx"] * r["se_b"])
    p_block = float(stats.t.sf(r["t_stat"], df=r["dof"])) if np.isfinite(r["t_stat"]) else np.nan
    p_day = float(stats.t.sf(rd["t_stat"], df=rd["dof"])) if np.isfinite(rd["t_stat"]) else np.nan
    notes = (f"{WEIGHT_NAMES.get(spec.weights, spec.weights)} weights; events A {r['n_events_a']}, B {r['n_events_b']}; "
             f"{r['n_anchor_dates']} blocks, dof {r['dof']}; winsorized {spec.winsorized}; {control_note}")
    pl_p, pl_notes = np.nan, "no placebo set (purchase-only placebo data)"
    vals = ctx.placebo(spec, "N2", "tau2") if spec.has_placebo else np.empty(0)
    pl = None
    if len(vals):
        pl = placebo_p_value(r["tau2_hat"], vals, "greater")
        pl_p, pl_notes = pl["p"], f"centered placebo, {pl['n_null']} draws"
    rows = [
        row(c, analysis, variant, "tau2", r["tau2_hat"], "bp2", placebo_p=pl_p, boot_lo=b["tau2_low"], boot_hi=b["tau2_high"], se_block=se_tau2,
            n_events=n_ev, n_households=n_hh, notes=f"{notes}; {pl_notes}"),
        row(c, analysis, variant, "tau", r["tau_hat"], "bp", boot_lo=b["tau_low"], boot_hi=b["tau_high"], n_events=n_ev, n_households=n_hh,
            notes="signed square root; household bootstrap 95% interval"),
        row(c, analysis, variant, "tau_annualized", tau_mod.annualized_scale(r["tau_hat"]), "bp", n_events=n_ev, n_households=n_hh, notes="x 12.6, scale convention"),
        row(c, analysis, variant, "tau_upper", b["tau_upper_one_sided"], "bp", n_events=n_ev, n_households=n_hh,
            notes="one-sided 95% upper bound: 95th percentile of the household-bootstrap tau2, signed root"),
        row(c, analysis, variant, "tau_upper_annualized", tau_mod.annualized_scale(b["tau_upper_one_sided"]), "bp", n_events=n_ev, n_households=n_hh),
        row(c, analysis, variant, "t_block", r["t_stat"], "t", n_events=n_ev, n_households=n_hh, notes=f"household x block clustered; dof {r['dof']}"),
        row(c, analysis, variant, "p_block", p_block, "p", n_events=n_ev, n_households=n_hh, notes="one-sided t p, descriptive"),
        row(c, analysis, variant, "tau2_se_day", float(rd["var_wx"] * rd["se_b"]), "bp2", n_events=n_ev, n_households=n_hh,
            notes=f"household x anchor day clustered; {rd['n_anchor_dates']} days, dof {rd['dof']}; appendix only"),
        row(c, analysis, variant, "p_day", p_day, "p", n_events=n_ev, n_households=n_hh, notes="one-sided t p with day clusters; appendix only"),
        row(c, analysis, variant, "boot_valid_draws", b["n_valid"], "count", n_events=n_ev, n_households=n_hh, notes=f"of {b['n_draws']} household-bootstrap draws"),
    ]
    if pl is not None:
        rows += [row(c, analysis, variant, "tau2_placebo_mean", pl["null_mean"], "bp2", n_events=n_ev, n_households=n_hh),
                 row(c, analysis, variant, "tau2_placebo_q95", pl["null_q95"], "bp2", n_events=n_ev, n_households=n_hh)]
    vt = ctx.placebo(spec, "N1", "tau2") if spec.has_placebo else np.empty(0)
    if len(vt):
        pt = placebo_p_value(r["tau2_hat"], vt, "greater")
        rows += [row(c, analysis, variant, "tau2_placebo_p_timing", pt["p"], "p", placebo_p=pt["p"], n_events=n_ev, n_households=n_hh,
                     notes=f"placebo p of tau2 against the timing-only placebo, {pt['n_null']} draws; added after results, descriptive"),
                 row(c, analysis, variant, "tau2_placebo_mean_timing", pt["null_mean"], "bp2", n_events=n_ev, n_households=n_hh, notes="timing-only placebo"),
                 row(c, analysis, variant, "tau2_placebo_q95_timing", pt["null_q95"], "bp2", n_events=n_ev, n_households=n_hh, notes="timing-only placebo")]
    rows += calibrated_bound_rows(ctx, c, analysis, variant, b["statistics"], n_ev, n_hh)
    rows += injection_bound_rows(ctx, spec, analysis, variant, r["tau2_hat"], n_ev, n_hh)
    return rows


def calibrated_bound_rows(ctx: Context, clock: str, analysis: str, variant: str, tau2_draws: np.ndarray, n_ev: int, n_hh: int) -> list[dict]:
    """The one-sided upper bound on tau at the clock's calibrated bootstrap level (`ctx.calibrated_levels`): the signed
    root of that quantile of the same household-bootstrap tau2 draws the nominal 95% bound uses. No rows when the
    clock has no calibrated level."""
    level = ctx.calibrated_levels.get(clock, np.nan)
    if not np.isfinite(level):
        return []
    f = np.asarray(tau2_draws, dtype=float)
    f = f[np.isfinite(f)]
    q = float(np.quantile(f, level)) if len(f) else np.nan
    up = tau_mod.signed_root(q) if np.isfinite(q) else np.nan
    pct = f"{100 * level:g}"
    return [row(clock, analysis, variant, "calibrated_level", level, "level", n_events=n_ev, n_households=n_hh,
                notes=f"smallest household-bootstrap level with >= 95% coverage of the realized dispersion at every injected tau in "
                      f"{{0, 50, 75, 100}} bp on 1,000 centered placebo datasets (calibration report section 12; the population-tau "
                      f"rule selects no level up to 0.999); {ADDED}"),
            row(clock, analysis, variant, "tau_upper_calibrated", up, "bp", n_events=n_ev, n_households=n_hh,
                notes=f"one-sided upper bound at the calibrated level: signed root of the {pct}th percentile of the household-bootstrap "
                      f"tau2 (the draws of tau_upper, whose root is clipped at zero); {ADDED}"),
            row(clock, analysis, variant, "tau_upper_calibrated_annualized", tau_mod.annualized_scale(up), "bp", n_events=n_ev, n_households=n_hh,
                notes=f"x 12.6, scale convention; {ADDED}")]


def injection_bound_key(spec: Spec) -> tuple[str, str, str] | None:
    """(clock, weights, outcome label) of the inverted-bound row a spec can use: the headline spec of its clock with
    any weights (label `headline`) or the headline event-share spec with the `market_adjusted` / `no_alpha` outcome.
    None for every other spec (the injection curves were computed on those panels only)."""
    if spec.kind != "tau" or spec.weights not in WEIGHT_NAMES:
        return None
    for label, outcome in INJECTION_OUTCOMES.items():
        base = replace(headline_spec(spec.clock, spec.weights), name=spec.name, outcome=outcome)
        if spec == base and (label == "headline" or spec.weights == "ES"):
            return spec.clock, WEIGHT_NAMES[spec.weights], label
    return None


def injection_bound_rows(ctx: Context, spec: Spec, analysis: str, variant: str, tau2_hat: float, n_ev: int, n_hh: int) -> list[dict]:
    """The one-sided 95% upper bound on tau from inverting the placebo-injection test (`ctx.inverted_bounds`, calibration
    report section 13): read from the file for the spec's (clock, weights, outcome) when the file's observed tau2 is this
    run's tau2_hat (relative tolerance `INJECTION_TOL`); no rows otherwise (a mismatch is logged, not raised)."""
    key = injection_bound_key(spec)
    if ctx.inverted_bounds is None or key is None or key not in ctx.inverted_bounds.index:
        return []
    b = ctx.inverted_bounds.loc[key]
    observed = float(b["observed_tau2"])
    if not np.isfinite(tau2_hat) or abs(observed - tau2_hat) > INJECTION_TOL * max(1.0, abs(tau2_hat)):
        ctx.log(f"inverted bound {key}: observed tau2 on file {observed} differs from this run's {tau2_hat}; no injection-bound rows")
        return []
    up = float(b["bound_bp"])
    lo, hi, w = float(b["crossing_lower_grid"]), float(b["crossing_upper_grid"]), float(b["interp_weight"])
    n_pan, alpha = int(b["n_panels_per_tau"]), float(b["alpha"])
    method = (f"one-sided {100 * (1 - alpha):g}% upper bound by inverting the placebo-injection test: the largest injected tau whose "
              f"{100 * alpha:g}th percentile of tau2_hat over {n_pan:,d} centered placebo datasets with dispersion tau injected is <= the "
              f"observed tau2 ({observed:.1f}), linear crossing between grid taus {lo:g} and {hi:g} bp (weight {w:.3f}); "
              f"coverage of the injected tau {100 * (1 - alpha):g}% at grid taus by construction; calibration report section 13, "
              f"inverted_bound.csv")
    return [row(spec.clock, analysis, variant, "tau_upper_injection", up, "bp", n_events=n_ev, n_households=n_hh, notes=f"{method}; {ADDED}"),
            row(spec.clock, analysis, variant, "tau_upper_injection_annualized", tau_mod.annualized_scale(up), "bp", n_events=n_ev, n_households=n_hh,
                notes=f"x 12.6, scale convention; {ADDED}"),
            row(spec.clock, analysis, variant, "tau_upper_injection_mc_se", float(b["bound_mc_se_bp"]), "bp",
                boot_lo=float(b["bound_mc_p025"]), boot_hi=float(b["bound_mc_p975"]), n_events=n_ev, n_households=n_hh,
                notes=f"Monte Carlo SE of the injection bound: {int(b['bound_mc_n_boot']):,d} bootstrap resamples of the placebo-draw index "
                      f"(jointly across taus), 2.5th / 97.5th percentiles in boot_lo / boot_hi; {ADDED}")]


NO_CONTROL_SUPERSEDED = "identical to the headline by construction; superseded by tau_no_control"
NO_CONTROL_METHOD = ("the w-weighted covariance of the two UNRESIDUALIZED period means (no activity control in either stage); "
                     "the headline residualizes the first-period mean on [1, log(1 + first-period purchases)] and x is "
                     "w-orthogonal to that design, so the no_activity_control robustness variant reproduces the headline exactly")


def mark_notes(rows: list[dict], note: str) -> list[dict]:
    """Append a note to every row (in place); rows with no note get the note alone."""
    for r in rows:
        r["notes"] = f"{r['notes']}; {note}" if r["notes"] else note
    return rows


def run_tau_no_control(ctx: Context, spec: Spec, variant: str) -> list[dict]:
    """The genuine no-control estimand with the headline's inference, its difference from the headline and
    the weighted correlation that explains the difference."""
    c = spec.clock
    rows = run_tau(ctx, spec, TAU_NO_CONTROL, variant)
    panel = ctx.panel(spec)
    if not panel.empty:
        head = float(cgm.tau2_i1(block_cluster(panel, ctx.calendar), spec.weights, anchor="anchor_blk")["tau2_hat"])
        tau2 = float(rows[0]["value"])
        n_ev, n_hh = rows[0]["n_events"], rows[0]["n_households"]
        table = nc.household_table(panel, spec.weights)
        rows += [
            row(c, TAU_NO_CONTROL, variant, "tau2_headline", head, "bp2", n_events=n_ev, n_households=n_hh,
                notes="the headline tau2 of the same clock and weights (the activity control in both stages)"),
            row(c, TAU_NO_CONTROL, variant, "tau2_diff_headline", tau2 - head, "bp2", n_events=n_ev, n_households=n_hh,
                notes="no-control minus headline"),
            row(c, TAU_NO_CONTROL, variant, "tau_diff_headline", tau_mod.signed_root(tau2) - tau_mod.signed_root(head), "bp",
                n_events=n_ev, n_households=n_hh, notes="signed roots, no-control minus headline"),
            row(c, TAU_NO_CONTROL, variant, "activity_correlation", nc.activity_correlation(table), "correlation",
                n_events=n_ev, n_households=n_hh,
                notes="w-weighted correlation between the first-period mean and log(1 + first-period purchases): "
                      "the reason the activity control does or does not move the estimand"),
        ]
    return mark_notes(rows, f"{NO_CONTROL_METHOD}; {ADDED}")


def run_subperiod(ctx: Context, clock: str) -> list[dict]:
    """The 3x3 w-weighted covariance matrix of subperiod mean purchase outcomes, with the centered-placebo
    distribution of every element: is the two-period covariance special?"""
    spec = analysis_specs(clock)[SUBPERIOD]
    panel = ctx.panel(spec)
    s = sub_mod.subperiod_statistics(panel, ctx.calendar)
    n_hh = int(s["n_households_all_three"])
    n_ev = int(sum(s[f"n_events_{k}"] for k in sub_mod.SUBPERIODS))
    b = sub_mod.SUBPERIOD_BOUNDS
    where = ", ".join(f"{name} {b[i]} to {b[i + 1]}" for i, name in enumerate(sub_mod.SUBPERIODS))
    extra = (f"three consecutive subperiods of equal calendar length ({where}, half-open, 44 months each); "
             f"a purchase counts only when its {sub_mod.HORIZON}-session window also ends inside its subperiod; "
             f"households with purchases in all three; pairwise weights sqrt(n_i n_j)")
    rows = [row(clock, SUBPERIOD, "default", "n_households", n_hh, "count", n_events=n_ev, n_households=n_hh, notes=extra),
            row(clock, SUBPERIOD, "default", "n_households_panel", s["n_households_panel"], "count", n_events=n_ev, n_households=n_hh,
                notes="households with purchases anywhere in 2015-2025 (the pool)"),
            row(clock, SUBPERIOD, "default", "n_events_dropped", s["n_events_dropped"], "count", n_events=n_ev, n_households=n_hh,
                notes="purchases whose window does not complete inside the subperiod holding the anchor")]
    for name in sub_mod.SUBPERIODS:
        rows.append(row(clock, SUBPERIOD, "default", f"n_events_{name}", s[f"n_events_{name}"], "count", n_events=n_ev,
                        n_households=n_hh, notes="purchases of those households in this subperiod"))
    for stat in sub_mod.SUBPERIOD_STATS:
        rows += _placebo_rows(ctx, spec, SUBPERIOD, "default", stat, s[stat], "bp2", "N2", "greater", n_ev, n_hh, extra)
    for a, bb_ in sub_mod.SUBPERIOD_PAIRS:
        stat = f"corr_{a}_{bb_}"
        rows.append(row(clock, SUBPERIOD, "default", stat, s[stat], "correlation", n_events=n_ev, n_households=n_hh,
                        notes="the same matrix as correlations"))
    return mark_notes(rows, ADDED)


def run_continuation(ctx: Context, clock: str) -> list[dict]:
    """Continuation into the second period on the first-period mean given log activity (selection on performance)."""
    spec = analysis_specs(clock)[CONTINUATION]
    t = cont_mod.continuation_table(ctx.panel(spec))
    n_hh, n_ev = int(len(t)), int(t["n_a"].sum())
    extra = ("households meeting the first-period half of the persistence rule (the population before the second-period "
             f"condition); one row per household, HC1 standard errors; the first-period mean enters per {cont_mod.MEAN_SCALE:g} bp")
    rows: list[dict] = []
    for outcome in cont_mod.OUTCOMES:
        meaning = ("at least one purchase in 2021-2025" if outcome == "any_purchase"
                   else "meets the second-period half of the persistence rule (the estimate's sample)")
        for method, fn in (("ols", cont_mod.ols_hc1), ("logit", cont_mod.logit)):
            try:
                f = fn(t, outcome)
            except (ValueError, KeyError, np.linalg.LinAlgError) as e:  # noqa: BLE001
                rows.append(row(clock, CONTINUATION, outcome, f"{method}_failed", np.nan, "", n_events=n_ev, n_households=n_hh,
                                notes=f"{type(e).__name__}: {str(e)[:120]}"))
                continue
            unit = "probability" if method == "ols" else "log_odds"
            for name in cont_mod.REGRESSORS + ("intercept",):
                rows.append(row(clock, CONTINUATION, outcome, f"{method}_{name}", f[f"coef_{name}"], unit,
                                se_block=f[f"se_{name}"], n_events=n_ev, n_households=n_hh,
                                notes=f"{'linear probability model' if method == 'ols' else 'logit'}, {meaning}; "
                                      f"standard error in se_block; {extra}"))
            rows.append(row(clock, CONTINUATION, outcome, f"{method}_r2", f["r2"], "share", n_events=n_ev, n_households=n_hh,
                            notes="R2" if method == "ols" else "McFadden pseudo-R2"))
            if method == "logit":
                rows.append(row(clock, CONTINUATION, outcome, "logit_converged", float(f["converged"]), "0/1",
                                n_events=n_ev, n_households=n_hh))
        rows.append(row(clock, CONTINUATION, outcome, "share_continuing", float(t[outcome].mean()), "share",
                        n_events=n_ev, n_households=n_hh, notes=f"share of the {n_hh} households: {meaning}"))
        rows.append(row(clock, CONTINUATION, outcome, "n_households_regression", n_hh, "count", n_events=n_ev, n_households=n_hh,
                        notes=extra))
    return mark_notes(rows, ADDED)


def wls_slope_hc1(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """y = a + slope*x by weighted least squares with HC1 (heteroskedasticity-robust) standard errors."""
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    xtwx = X.T @ (X * w[:, None])
    bread = np.linalg.inv(xtwx)
    beta = bread @ (X.T @ (w * y))
    u = y - X @ beta
    meat = (X * (w * u)[:, None]).T @ (X * (w * u)[:, None])
    V = bread @ meat @ bread * n / (n - 2)
    se = np.sqrt(np.diag(V))
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "se_intercept": float(se[0]), "se_slope": float(se[1]), "n": int(n)}


def run_household_scatter(ctx: Context, clock: str) -> list[dict]:
    """The anonymized household scatter behind the new figure: the first-period residual against the
    second-period mean, bubble size = w_h, plus the fitted weighted slope. No household ids leave this function."""
    spec = headline_spec(clock)
    panel = ctx.panel(spec)
    table = tau_mod.household_table(panel, spec.weights)
    fit = wls_slope_hc1(table.x, table.ybar_b, table.w)
    boot = bb.tau2_household_bootstrap(panel, spec.weights, ctx.n_boot, seed_from(spec.name, clock, "household bootstrap"))
    order = np.argsort(-table.x, kind="stable")  # 1 = largest first-period residual; the index replaces the id
    for i, k in enumerate(order, start=1):
        ctx.scatter_rows.append(dict(clock=clock, household_index=i, x=float(table.x[k]), y=float(table.ybar_b[k]),
                                     w=float(table.w[k])))
    n_ev, n_hh = int(table.n_a.sum() + table.n_b.sum()), int(len(table.household_id))
    notes = ("weighted least squares of the second-period mean on the first-period residual, weights w_h = "
             "sqrt(first-period x second-period purchases); the household rows are in results_household_scatter.csv")
    rows = [row(clock, HOUSEHOLD_SCATTER, "event_share", "slope", fit["slope"], "slope", boot_lo=boot["b_low"], boot_hi=boot["b_high"],
                se_block=fit["se_slope"], n_events=n_ev, n_households=n_hh,
                notes=f"{notes}; HC1 standard error in se_block, household-bootstrap 2.5th / 97.5th percentiles in boot_lo / boot_hi"),
            row(clock, HOUSEHOLD_SCATTER, "event_share", "intercept", fit["intercept"], "bp", se_block=fit["se_intercept"],
                n_events=n_ev, n_households=n_hh, notes="w-weighted, so the intercept is the w-mean second-period outcome"),
            row(clock, HOUSEHOLD_SCATTER, "event_share", "var_x", tau_mod.var_w(table.x, table.w), "bp2", n_events=n_ev,
                n_households=n_hh, notes="w-weighted variance of the first-period residual; slope x var_x = tau2"),
            row(clock, HOUSEHOLD_SCATTER, "event_share", "n_households", n_hh, "count", n_events=n_ev, n_households=n_hh)]
    return mark_notes(rows, ADDED)


def run_influence(ctx: Context, clock: str) -> list[dict]:
    """Leave-one-household-out and leave-one-year-out ranges of the event-share tau."""
    spec = headline_spec(clock)
    panel = ctx.panel(spec)
    rows = []
    for variant, loo in (("leave_one_household_out", tau_mod.leave_one_household_out(panel, "ES")),
                         ("leave_one_year_out", tau_mod.leave_one_year_out(panel, "ES"))):
        n_ev, n_hh = _counts(panel)
        notes = f"{len(loo)} refits of the event-share tau"
        rows += [row(clock, "influence", variant, "tau_min", float(loo["tau_hat"].min()), "bp", n_events=n_ev, n_households=n_hh, notes=notes),
                 row(clock, "influence", variant, "tau_max", float(loo["tau_hat"].max()), "bp", n_events=n_ev, n_households=n_hh, notes=notes),
                 row(clock, "influence", variant, "tau2_min", float(loo["tau2_hat"].min()), "bp2", n_events=n_ev, n_households=n_hh, notes=notes),
                 row(clock, "influence", variant, "tau2_max", float(loo["tau2_hat"].max()), "bp2", n_events=n_ev, n_households=n_hh, notes=notes),
                 row(clock, "influence", variant, "refits", len(loo), "count", n_events=n_ev, n_households=n_hh)]
    return rows


def _sample_note(variant_prefix: str) -> str:
    return SAMPLE_NOTES.get(variant_prefix.rstrip("_"), "")


def run_best_household(ctx: Context, spec: Spec, variant_prefix: str = "", flavours: tuple[str, ...] = ("N2", "N1")) -> list[dict]:
    """Largest mean, largest t, top-decile mean and nominal p < 0.05 count, each in the requested placebo distributions."""
    panel = ctx.panel(spec)
    w = winners.panel_winner_stats(panel, min_n=FULL_MIN_N)
    n_ev, n_hh = int(len(panel)), int(w["n_households"])
    rows = []
    extra = f"households with >= {FULL_MIN_N} purchases; winsorized {spec.winsorized}"
    if _sample_note(variant_prefix):
        extra += f"; {_sample_note(variant_prefix)}"
    for flavour in flavours:
        variant = variant_prefix + FLAVOUR_VARIANT[flavour]
        for stat, key, unit in (("max_mean", "max_mean", "bp"), ("max_t", "max_t", "t"), ("top_decile_mean", "top_decile_mean", "bp"),
                                ("n_p05", "n_nominal_significant", "count")):
            rows += _placebo_rows(ctx, spec, "best_household", variant, stat, float(w[key]), unit, flavour, "greater", n_ev, n_hh, extra)
    return rows


def run_tail(ctx: Context, spec: Spec, variant_prefix: str = "") -> list[dict]:
    """Share of purchases with raw outcome / sigma_event above 2 and 3, and beside it below -2 and -3, in both
    placebo distributions (the lower-tail rows and the anchor-window samples were added after results)."""
    panel = ctx.panel(spec)
    ok = np.isfinite(panel["sigma_event"].to_numpy(dtype=float))
    n_excluded_anchors = int(panel.attrs.get("n_dropped_anchor_window", 0))
    panel = panel.loc[ok]
    s = tt.t4_tail_shares(panel)
    n_ev, n_hh = _counts(panel)
    rows = []
    extra = f"z = raw outcome / sigma_event; {int((~ok).sum())} events without sigma_event dropped"
    if _sample_note(variant_prefix):
        extra += f"; {_sample_note(variant_prefix)}"
    if spec.exclude_anchor_window:
        lo, hi = ANCHOR_WINDOWS[spec.exclude_anchor_window]
        extra += f"; {n_excluded_anchors} anchors in [{lo}, {hi}) excluded; {ADDED}"
    for flavour in ("N2", "N1"):
        for z in tt.Z_THRESHOLDS:
            stat = f"share_gt_{z:g}"
            rows += _placebo_rows(ctx, spec, "tail", variant_prefix + FLAVOUR_VARIANT[flavour], stat, s[stat], "share", flavour, "greater", n_ev, n_hh, extra)
    lower = f"{extra}; lower tail: share of z below minus the threshold" + ("" if spec.exclude_anchor_window else f"; {ADDED}")
    for flavour in ("N2", "N1"):
        for z in tt.Z_THRESHOLDS:
            stat = tt.lower_tail_stat(z)
            rows += _placebo_rows(ctx, spec, "tail", variant_prefix + FLAVOUR_VARIANT[flavour], stat, s[stat], "share", flavour, "greater", n_ev, n_hh, lower)
    return rows


def run_purchase_mean(ctx: Context, spec: Spec, variant_prefix: str = "") -> list[dict]:
    """Raw mean purchase outcome against zero (block-clustered) and against the timing-only placebo mean."""
    c = spec.clock
    pb = block_cluster(ctx.panel(spec), ctx.calendar)
    r = tt.t1_clustered_mean(pb, anchor="anchor_blk")
    rd = tt.t1_clustered_mean(pb, anchor="anchor_date")
    n_ev, n_hh = int(r["n_events"]), int(r["n_households"])
    vz, vp = variant_prefix + "vs_zero", variant_prefix + "vs_timing_placebo"
    sample = "raw outcome, all households, periods A + B" if not _sample_note(variant_prefix) else f"raw outcome, {_sample_note(variant_prefix)}, periods A + B"
    rows = [row(c, "purchase_mean", vz, "mean", r["estimate"], "bp", se_block=r["se"], n_events=n_ev, n_households=n_hh,
                notes=f"{sample}; household x block clustered; {r['n_anchor_dates']} blocks"),
            row(c, "purchase_mean", vz, "t_block", r["t_stat"], "t", n_events=n_ev, n_households=n_hh),
            row(c, "purchase_mean", vz, "p_block", r["p_two_sided"], "p", n_events=n_ev, n_households=n_hh, notes="two-sided"),
            row(c, "purchase_mean", vz, "dof_block", r["dof"], "count", n_events=n_ev, n_households=n_hh),
            row(c, "purchase_mean", vz, "n_blocks", r["n_anchor_dates"], "count", n_events=n_ev, n_households=n_hh),
            row(c, "purchase_mean", vz, "se_day", rd["se"], "bp", n_events=n_ev, n_households=n_hh,
                notes=f"household x anchor day clustered; {rd['n_anchor_dates']} days; appendix only"),
            row(c, "purchase_mean", vz, "p_day", rd["p_two_sided"], "p", n_events=n_ev, n_households=n_hh, notes="two-sided; appendix only")]
    vals = ctx.placebo(spec, "N1", "mean")
    if len(vals):
        pl = placebo_p_value(r["estimate"], vals, "two-sided")
        rows += [row(c, "purchase_mean", vp, "mean", r["estimate"], "bp", placebo_p=pl["p"], n_events=n_ev, n_households=n_hh,
                     notes=f"two-sided placement about the placebo median; {pl['n_null']} timing-only placebo draws"),
                 row(c, "purchase_mean", vp, "placebo_mean", pl["null_mean"], "bp", n_events=n_ev, n_households=n_hh),
                 row(c, "purchase_mean", vp, "placebo_sd", pl["null_sd"], "bp", n_events=n_ev, n_households=n_hh)]
    return rows


def run_rank_sets(ctx: Context, spec: Spec) -> list[dict]:
    """Simultaneous 95% rank sets from the within-household 20-day-block bootstrap of household means."""
    c = spec.clock
    panel = ctx.panel(spec)
    hb = bb.household_block_bootstrap(panel, ctx.calendar, BLOCK_L, ctx.n_boot, seed_from(spec.name, c, "rank sets"))
    sets = ranks_mod.rank_confidence_sets(hb.means, hb.boot_means, 0.95, hb.household_id)
    s = ranks_mod.rank_set_summary(sets)
    n_ev, n_hh = int(len(panel)), int(s["n_households"])
    notes = f"within-household block bootstrap of household means, L = {BLOCK_L}, {sets.n_draws_used} draws ({sets.n_draws_dropped} dropped)"
    rows = [row(c, "rank_sets", "default", k, s[k], u, n_events=n_ev, n_households=n_hh, notes=notes) for k, u in
            (("median_width_over_H", "share"), ("mean_width_over_H", "share"), ("share_pairs_resolved", "share"),
             ("n_distinguishable_from_median", "count"), ("n_singleton_sets", "count"), ("n_full_sets", "count"), ("critical_value", "t"))]
    rows.append(row(c, "rank_sets", "default", "boot_draws_used", sets.n_draws_used, "count", n_events=n_ev, n_households=n_hh))
    f = sets.frame().sort_values("point_rank")
    n_by_h = pd.Series(hb.n_events, index=hb.household_id)
    for r in f.itertuples(index=False):
        ctx.rank_rows.append(dict(clock=c, household_index=int(r.point_rank), point_rank=int(r.point_rank), estimate=float(r.estimate),
                                  rank_lower=int(r.rank_lower), rank_upper=int(r.rank_upper), width=int(r.width), n_events=int(n_by_h[r.household_id])))
    return rows


def run_shrinkage(ctx: Context, spec: Spec) -> list[dict]:
    """Empirical Bayes on first-period means; boundary frequency over bootstrap refits; forecast / reliability slopes."""
    c = spec.clock
    panel = ctx.panel(spec)
    table = tau_mod.household_table(panel, "ES")
    a = panel.loc[panel["period"] == "A"].groupby("household_id")["y_h20"].agg(["mean", "std", "count"]).reindex(table.household_id)
    se_a = (a["std"] / np.sqrt(a["count"])).to_numpy(dtype=float)
    fit = eb_mod.fit_eb(table.ybar_a, se_a, table.n_a)
    rl = eb_mod.reliability_regression(table.ybar_a, table.ybar_b)
    hb = bb.household_block_bootstrap(panel, ctx.calendar, BLOCK_L, ctx.n_boot, seed_from(spec.name, c, "shrinkage"), periods=("A",))
    order = pd.Index(hb.household_id).get_indexer(table.household_id)
    at_zero, n_ok = 0, 0
    for d in range(hb.boot_means.shape[0]):
        yb = hb.boot_means[d, order]
        if not np.all(np.isfinite(yb)):
            continue
        try:
            at_zero += int(eb_mod.fit_eb(yb, se_a, table.n_a).at_zero_boundary)
            n_ok += 1
        except ValueError:
            continue
    n_ev, n_hh = int(table.n_a.sum() + table.n_b.sum()), int(fit.n)
    rows = [row(c, "shrinkage", "default", "tau_eb", fit.tau, "bp", n_events=n_ev, n_households=n_hh, notes="marginal ML; prior mean linear in log(1+n)"),
            row(c, "shrinkage", "default", "at_zero_boundary", int(fit.at_zero_boundary), "0/1", n_events=n_ev, n_households=n_hh),
            row(c, "shrinkage", "default", "boundary_share", at_zero / n_ok if n_ok else np.nan, "share", n_events=n_ev, n_households=n_hh,
                notes=f"{n_ok} within-household block-bootstrap refits of the first-period means"),
            row(c, "shrinkage", "default", "reliability_slope", rl["slope"], "slope", n_events=n_ev, n_households=n_hh, notes="second-period on raw first-period means, HC1"),
            row(c, "shrinkage", "default", "reliability_slope_se", rl["se_slope"], "slope", n_events=n_ev, n_households=n_hh)]
    if fit.tau2 > 0:
        fc = eb_mod.forecast_regression(fit.posterior_mean, table.ybar_b)
        rows += [row(c, "shrinkage", "default", "forecast_slope", fc["slope"], "slope", n_events=n_ev, n_households=n_hh,
                     notes="second-period means on unstandardized first-period posterior means, HC1; 1 = calibrated"),
                 row(c, "shrinkage", "default", "forecast_slope_se", fc["se_slope"], "slope", n_events=n_ev, n_households=n_hh)]
    return rows


def design_effect(panel: pd.DataFrame, calendar: pd.DatetimeIndex, min_n: int = FULL_MIN_N) -> tuple[float, int]:
    """(d, households used): median over households with n_h >= min_n of the 20-day-block cluster-robust
    variance of the household mean relative to its iid variance."""
    p = block_cluster(panel, calendar)
    ratios = []
    for _, g in p.groupby("household_id"):
        y = g["y_h20"].to_numpy(dtype=float)
        n = len(y)
        if n < min_n:
            continue
        e = y - y.mean()
        iid = float(np.sum(e * e)) / (n * (n - 1))
        blocks = pd.Series(e).groupby(g["anchor_blk"].to_numpy()).sum().to_numpy()
        G = len(blocks)
        if G < 2 or iid <= 0:
            continue
        clu = float(np.sum(blocks * blocks)) / (n * n) * G / (G - 1)
        ratios.append(clu / iid)
    return (float(np.median(ratios)) if ratios else np.nan), len(ratios)


THETA_WORDS = {2.0: "Two", 5.0: "Five", 10.0: "Ten", 20.0: "Twenty"}
N_WORDS = {10: "Ten", 25: "TwentyFive", 50: "Fifty", 100: "Hundred", 250: "TwoFifty", 500: "FiveHundred"}


def run_time_to_detect(ctx: Context, spec: Spec) -> list[dict]:
    """Years to detect with the raw pooled SD and the measured design effect; grid to results_time_to_detect.csv."""
    c = spec.clock
    panel = ctx.panel(spec)
    sigma = float(np.std(panel["y_h20"].to_numpy(dtype=float), ddof=1))
    d, n_d = design_effect(panel, ctx.calendar)
    H = int(ctx.panel(headline_spec(c)).attrs["n_households_eligible"])
    d_use = d if np.isfinite(d) and d > 0 else 1.0
    yrs = pd.DatetimeIndex(panel["anchor_date"]).year
    per_year = panel.assign(_y=yrs).groupby(["household_id", "_y"]).size()
    n_ev, n_hh = _counts(panel)
    grid = ttd_mod.ttd_grid(sigma, d_use, H)
    for g in grid.itertuples(index=False):
        ctx.ttd_rows.append(dict(clock=c, theta_annual_pct=g.theta_annual_pct, theta_20_bp=g.theta_20_bp, n_per_year=g.n_per_year, sigma_bp=sigma,
                                 design_effect=d_use, years=g.years, years_best_of_h=g.years_best_of_h, n_households=H))
    rows = [row(c, "time_to_detect", "default", "sigma", sigma, "bp", n_events=n_ev, n_households=n_hh, notes="raw pooled SD, periods A + B"),
            row(c, "time_to_detect", "default", "design_effect", d, "ratio", n_events=n_ev, n_households=n_d,
                notes=f"median over {n_d} households with >= {FULL_MIN_N} purchases of block-clustered / iid variance of the household mean"),
            row(c, "time_to_detect", "default", "H", H, "count", n_events=n_ev, n_households=H, notes="persistence-rule households, for best-of-H"),
            row(c, "time_to_detect", "default", "median_purchases_per_household_year", float(per_year.median()), "count", n_events=n_ev, n_households=n_hh),
            row(c, "time_to_detect", "default", "span_years", SPAN_YEARS, "years", n_events=n_ev, n_households=n_hh)]
    for g in grid.itertuples(index=False):
        base = f"years_theta{THETA_WORDS[float(g.theta_annual_pct)]}_n{N_WORDS[int(g.n_per_year)]}"
        rows += [row(c, "time_to_detect", "default", base, g.years, "years", n_events=n_ev, n_households=H, notes=f"one-sided 5%, 80% power, d = {d_use:.3f}"),
                 row(c, "time_to_detect", "default", base + "_best_of_h", g.years_best_of_h, "years", n_events=n_ev, n_households=H, notes=f"Bonferroni over H = {H}")]
    return rows


def run_sample(ctx: Context, clock: str) -> list[dict]:
    """Counts of the analysis sample and the official-only sample (no household identity)."""
    from .specs import apply_dup, eligible_households

    frame = ctx.frames[clock]
    rows = []
    for variant, dup in (("default", "default"), ("official_only", "official")):
        f = frame.loc[(frame["direction"] == "purchase") & frame["period"].isin(["A", "B"])] if "direction" in frame.columns else frame
        n_before = len(f)
        f = apply_dup(f, dup)
        y = pd.to_numeric(f["y_h20"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)
        fin = f.loc[ok]
        hh_all = fin["household_id"].nunique()
        b = f.loc[f["period"] == "B"].groupby("household_id").size().sort_values(ascending=False)
        share = (b / b.sum()).to_numpy(dtype=float) if len(b) else np.empty(0)
        cont = continuation_rate(fin)
        persistent = eligible_households(fin, "R1")
        pers = fin.loc[fin["household_id"].isin(persistent)]
        stats = [("n_events", int(len(fin)), "count"), ("n_events_A", int((fin["period"] == "A").sum()), "count"),
                 ("n_events_B", int((fin["period"] == "B").sum()), "count"), ("n_households", int(hh_all), "count"),
                 ("n_households_persistence", int(len(persistent)), "count"),
                 ("n_events_persistence_A", int((pers["period"] == "A").sum()), "count"),
                 ("n_events_persistence_B", int((pers["period"] == "B").sum()), "count"),
                 ("n_households_full20", int(len(eligible_households(fin, "full_n20"))), "count"),
                 ("n_dropped_duplicates", int(n_before - len(f)), "count"),
                 ("n_dropped_nonfinite", int((~ok).sum()), "count"), ("continuation_rate", cont["continuation_rate"], "share"),
                 ("top2_share_B", float(share[:2].sum()) if len(share) else np.nan, "share"),
                 ("effective_households_B", float(1.0 / np.sum(share ** 2)) if len(share) else np.nan, "count")]
        for stat, v, unit in stats:
            notes = ""
            if stat == "n_dropped_duplicates":
                notes = "all-lines duplicate rule" if dup == "default" else "non-official events removed"
            if stat == "continuation_rate":
                notes = f"{cont['n_continuing']} of {cont['n_eligible_A']} households meeting the period-A half of the rule"
            if stat.startswith("n_events_persistence"):
                notes = "purchases of the households meeting the persistence rule (the rows of the tau estimate)"
            rows.append(row(clock, "sample", variant, stat, v, unit, n_events=int(len(fin)), n_households=int(hh_all), notes=notes))
    return rows


def holm_rows(rows: list[dict], analysis: str, members: list[tuple[str, str]], family: str) -> list[dict]:
    """Holm-adjusted p across a family of (clock, variant) tau rows: the centered placebo p of `tau2` as the nominal p
    (`placebo_p_holm`) and, separately, the block-clustered one-sided p (`p_block_holm`). Members without a nominal p
    get no adjusted row and do not count toward the family size."""
    df = pd.DataFrame(rows)
    if df.empty:
        return []

    def pick(clock: str, variant: str, statistic: str, col: str) -> tuple[float, dict | None]:
        m = (df["clock"] == clock) & (df["analysis"] == analysis) & (df["variant"] == variant) & (df["statistic"] == statistic)
        if int(m.sum()) != 1:
            return np.nan, None
        r = df.loc[m].iloc[0]
        return float(r[col]) if pd.notna(r[col]) else np.nan, r.to_dict()

    out = []
    for statistic, col, name, label in (("tau2", "placebo_p", "placebo_p_holm", "centered placebo p of tau2"),
                                        ("p_block", "value", "p_block_holm", "block-clustered one-sided p")):
        picked = [pick(c, v, statistic, col) for c, v in members]
        nominal = np.array([p for p, _ in picked], dtype=float)
        adjusted = holm_adjust(nominal)
        m = int(np.isfinite(nominal).sum())
        for (c, v), p, adj, (_, src) in zip(members, nominal, adjusted, picked, strict=True):
            if src is None or not np.isfinite(p):
                continue
            out.append(row(c, analysis, v, name, float(adj), "p", placebo_p=float(p), n_events=src["n_events"], n_households=src["n_households"],
                           notes=f"Holm step-down over {m} {family} with a {label} (nominal p in placebo_p); added after results, descriptive"))
    return out


# ---------------------------------------------------------------- driver

def run_clock(ctx: Context, clock: str) -> list[dict]:
    """Every analysis of plan §4-§7 on one clock; an analysis that fails yields a row with the error in `notes`."""
    specs = analysis_specs(clock)
    jobs: list[tuple[str, Callable[[], list[dict]]]] = [("sample", lambda: run_sample(ctx, clock))]
    for w in ("ES", "EH", "PW"):
        s = specs[f"tau/{WEIGHT_NAMES[w]}"]
        jobs.append((s.name, lambda s=s: run_tau(ctx, s, "tau", WEIGHT_NAMES[s.weights])))
    for w in ("ES", "EH", "PW"):
        s = specs[f"{TAU_NO_CONTROL}/{WEIGHT_NAMES[w]}"]
        jobs.append((s.name, lambda s=s: run_tau_no_control(ctx, s, WEIGHT_NAMES[s.weights])))
    jobs.append(("influence", lambda: run_influence(ctx, clock)))
    jobs.append(("best_household", lambda: run_best_household(ctx, specs["best_household"])))
    jobs.append(("best_household/raw_outcome", lambda: run_best_household(ctx, specs["best_household/raw_outcome"], "raw_outcome_", ("N2",))))
    for v in SAMPLE_VARIANTS:
        jobs.append((f"best_household/{v}", lambda v=v: run_best_household(ctx, specs[f"best_household/{v}"], f"{v}_")))
    jobs.append(("tail", lambda: run_tail(ctx, specs["tail"])))
    for v in TAIL_VARIANTS:
        jobs.append((f"tail/{v}", lambda v=v: run_tail(ctx, specs[f"tail/{v}"], f"{v}_")))
    jobs.append(("purchase_mean", lambda: run_purchase_mean(ctx, specs["purchase_mean"])))
    for v in SAMPLE_VARIANTS:
        jobs.append((f"purchase_mean/{v}", lambda v=v: run_purchase_mean(ctx, specs[f"purchase_mean/{v}"], f"{v}_")))
    jobs.append(("rank_sets", lambda: run_rank_sets(ctx, specs["rank_sets"])))
    jobs.append(("shrinkage", lambda: run_shrinkage(ctx, specs["shrinkage"])))
    jobs.append(("time_to_detect", lambda: run_time_to_detect(ctx, specs["time_to_detect"])))
    jobs.append((SUBPERIOD, lambda: run_subperiod(ctx, clock)))
    jobs.append((CONTINUATION, lambda: run_continuation(ctx, clock)))
    jobs.append((HOUSEHOLD_SCATTER, lambda: run_household_scatter(ctx, clock)))
    for name in ROBUSTNESS:
        s = specs[f"robustness/{name}"]
        jobs.append((s.name, lambda s=s, name=name: run_tau(ctx, s, "robustness", name)
                     if name != "no_activity_control" else mark_notes(run_tau(ctx, s, "robustness", name), NO_CONTROL_SUPERSEDED)))
    rows: list[dict] = []
    for name, fn in jobs:
        t0 = time.time()
        try:
            new = fn()
        except Exception as e:  # noqa: BLE001 — every analysis gets a row, failed or not
            ctx.log(f"ERROR {clock} {name}: {type(e).__name__}: {e}")
            a, _, v = name.partition("/")
            new = [row(clock, a, v or "default", "failed", np.nan, "", notes=f"{type(e).__name__}: {str(e)[:200]}")]
        rows += new
        ctx.log(f"{clock} {name}: {len(new)} rows in {time.time() - t0:.1f} s")
    rows.append(row(clock, "robustness", "industry_adjusted", "tau2", np.nan, "bp2", notes="not run: no industry portfolio data on disk (plan §7)"))
    rows += holm_rows(rows, "robustness", [(clock, name) for name in ROBUSTNESS], f"of the {len(ROBUSTNESS)} robustness variants of this clock")
    return rows


def scatter_frame(ctx: Context) -> pd.DataFrame:
    """The anonymized household scatter rows collected by `run_household_scatter` (transaction clock first)."""
    df = pd.DataFrame(ctx.scatter_rows, columns=list(SCATTER_COLUMNS))
    if df.empty:
        return df
    order = {c: i for i, c in enumerate(("T", "P"))}
    return df.assign(_o=df["clock"].map(lambda c: order.get(c, 99))).sort_values(["_o", "household_index"], kind="stable").drop(columns="_o").reset_index(drop=True)


def run_all(ctx: Context, clocks: Iterable[str] = ("T", "P")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(results, rank sets, time-to-detect grid) for the requested clocks."""
    rows: list[dict] = []
    clocks = tuple(clocks)
    for c in clocks:
        rows += run_clock(ctx, c)
    members = [(c, WEIGHT_NAMES[w]) for c in clocks for w in ("ES", "EH", "PW")]
    rows += holm_rows(rows, "tau", members, f"tau rows (three weightings x {len(clocks)} clock{'s' if len(clocks) > 1 else ''})")
    return (results_frame(rows), pd.DataFrame(ctx.rank_rows, columns=list(RANK_COLUMNS)),
            pd.DataFrame(ctx.ttd_rows, columns=list(TTD_COLUMNS)))
