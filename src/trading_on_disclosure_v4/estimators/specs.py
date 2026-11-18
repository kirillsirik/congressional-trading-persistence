"""Analysis specifications and the spec-driven transformation of an event frame into the harness panel.
Pure functions on frames: nothing here reads data.

An EVENT FRAME carries, per event: `event_id, household_id, clock, period, anchor_date, anchor_year,
window_end_date, record_source, permno, direction, dup_any_line, dup_all_lines`, the outcome columns of
`OUTCOMES` (bp) and `sigma_event` (bp). The same transformation runs on the actual frame and on every
placebo draw joined to the event metadata by `event_id`, so a statistic's placebo distribution is built
exactly as the statistic itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..data.eligibility import RULES

#: harness outcome column -> event-frame column (bp)
OUTCOMES = {"ff5mom_h20": "y_h20", "ff5mom_h10": "y_h10", "ff5mom_h60": "y_h60",
            "ff5mom_noalpha_h20": "y_noalpha_h20", "mkt_h20": "y_mkt_h20"}
#: weight code -> plain name (plan §4)
WEIGHT_NAMES = {"ES": "event_share", "EH": "equal_household", "PW": "pair"}
FULL_MIN_N = 20
CLOCK_NAMES = {"T": "transaction", "P": "public"}
#: exclusion label -> household ids, registered at run time (`register_household_exclusion`); never written anywhere
HOUSEHOLD_EXCLUSIONS: dict[str, frozenset[str]] = {}
EXCL_TWO_VENDOR = "two_vendor_households"
SAMPLE_VARIANTS = ("official_only", "excl_two_vendor_households")
#: anchor-date windows an analysis may exclude (half-open [start, end)); on a placebo draw the SHIFTED anchor is tested
ANCHOR_WINDOWS = {"covid_2020": ("2020-02-01", "2020-06-01")}
EXCL_COVID = "excl_covid"
#: plain analysis names added after the results run (descriptive)
TAU_NO_CONTROL = "tau_no_control"
SUBPERIOD = "subperiod"
CONTINUATION = "continuation"
HOUSEHOLD_SCATTER = "household_scatter"
#: tail samples beyond the default one (period subsets, the two sample variants, and the same without Feb-May 2020 anchors)
TAIL_VARIANTS = ("period_A", "period_B", *SAMPLE_VARIANTS, EXCL_COVID, f"{EXCL_COVID}_period_A", f"{EXCL_COVID}_period_B")


def register_household_exclusion(label: str, household_ids) -> None:
    """Hold the household ids behind an exclusion label in memory (only `to_panel` reads them)."""
    HOUSEHOLD_EXCLUSIONS[label] = frozenset(str(h) for h in household_ids)


@dataclass(frozen=True)
class Spec:
    name: str
    clock: str
    kind: str  # tau | mean | best | tail | ranks | shrinkage | detect
    weights: str = "ES"
    outcome: str = "ff5mom_h20"
    winsorized: bool = True
    rule: str = "R1"  # R0 | R1 | R2 | full_n20 | all
    split: str = "AB"  # AB | ODDEVEN | WITHINB
    dup: str = "default"  # default (all-lines) | keepdups | official
    collapse_hsm: bool = False
    activity_control: bool = True
    direction: str = "purchase"  # purchase | sale | signed
    exclude: str = ""  # label in HOUSEHOLD_EXCLUSIONS; "" = nobody excluded
    period_subset: str = "AB"  # AB | A | B (rows kept after the split)
    exclude_anchor_window: str = ""  # label in ANCHOR_WINDOWS; "" = no anchors excluded
    residualize_activity: bool = True  # False = x_h is the w-demeaned first-period mean (the no-control estimand)

    @property
    def panel_key(self) -> str:
        """Everything that changes the panel a statistic is computed on (the placebo-cache key)."""
        key = [self.clock, self.outcome, "wins" if self.winsorized else "raw", self.rule, self.split, self.dup,
               "hsm" if self.collapse_hsm else "nohsm", self.direction]
        if self.exclude:
            key.append(f"excl:{self.exclude}")
        if self.period_subset != "AB":
            key.append(f"period:{self.period_subset}")
        if self.exclude_anchor_window:
            key.append(f"anchors_excl:{self.exclude_anchor_window}")
        return "|".join(key)

    @property
    def estimator_key(self) -> str:
        """Everything beyond the panel that changes the statistic (the placebo-cache key's tail)."""
        key = [self.weights, "ctrl" if self.activity_control else "noctrl"]
        if not self.residualize_activity:
            key.append("noresid")
        return "|".join(key)

    @property
    def has_placebo(self) -> bool:
        """The stored placebo sets are purchase-only."""
        return self.direction == "purchase"


#: robustness variant -> change to the headline spec (plan §7)
ROBUSTNESS: dict[str, dict] = {
    "no_activity_control": {"activity_control": False},
    "loose_rule": {"rule": "R0"},
    "strict_rule": {"rule": "R2"},
    "odd_even": {"split": "ODDEVEN"},
    "official_only": {"dup": "official"},
    "keep_duplicates": {"dup": "keepdups"},
    "market_adjusted": {"outcome": "mkt_h20"},
    "no_alpha": {"outcome": "ff5mom_noalpha_h20"},
    "horizon_10": {"outcome": "ff5mom_h10"},
    "horizon_60": {"outcome": "ff5mom_h60"},
    "raw_outcome": {"winsorized": False},
    "one_per_security_month": {"collapse_hsm": True},
    "within_second_period": {"split": "WITHINB"},
    "sales": {"direction": "sale"},
    "signed_all": {"direction": "signed"},
}


def headline_spec(clock: str, weights: str = "ES") -> Spec:
    return Spec(f"tau/{WEIGHT_NAMES[weights]}", clock, "tau", weights=weights)


def no_control_spec(clock: str, weights: str = "ES") -> Spec:
    """The genuine no-control estimand: the activity control is dropped from BOTH stages, so tau2 is the
    w-weighted covariance of the two unresidualized period means (`estimators.no_control`)."""
    return Spec(f"{TAU_NO_CONTROL}/{WEIGHT_NAMES[weights]}", clock, "tau", weights=weights,
                activity_control=False, residualize_activity=False)


def analysis_specs(clock: str) -> dict[str, Spec]:
    """Every analysis of plan §4-§7 for one clock, keyed by plain name."""
    specs: dict[str, Spec] = {}
    for w in ("ES", "EH", "PW"):
        s = headline_spec(clock, w)
        specs[s.name] = s
    for w in ("ES", "EH", "PW"):
        s = no_control_spec(clock, w)
        specs[s.name] = s
    base = headline_spec(clock)
    for name, change in ROBUSTNESS.items():
        specs[f"robustness/{name}"] = replace(base, name=f"robustness/{name}", **change)
    specs["purchase_mean"] = Spec("purchase_mean", clock, "mean", weights="none", winsorized=False, rule="all")
    specs["best_household"] = Spec("best_household", clock, "best", weights="none", rule="full_n20")
    specs["best_household/raw_outcome"] = Spec("best_household/raw_outcome", clock, "best", weights="none", winsorized=False, rule="full_n20")
    specs["tail"] = Spec("tail", clock, "tail", weights="none", winsorized=False, rule="all")
    for per in ("A", "B"):
        specs[f"tail/period_{per}"] = replace(specs["tail"], name=f"tail/period_{per}", period_subset=per)
    specs[f"tail/{EXCL_COVID}"] = replace(specs["tail"], name=f"tail/{EXCL_COVID}", exclude_anchor_window="covid_2020")
    for per in ("A", "B"):
        specs[f"tail/{EXCL_COVID}_period_{per}"] = replace(specs[f"tail/{EXCL_COVID}"], name=f"tail/{EXCL_COVID}_period_{per}", period_subset=per)
    for base_name in ("purchase_mean", "best_household", "tail"):
        for variant, change in (("official_only", {"dup": "official"}), ("excl_two_vendor_households", {"exclude": EXCL_TWO_VENDOR})):
            specs[f"{base_name}/{variant}"] = replace(specs[base_name], name=f"{base_name}/{variant}", **change)
    specs[SUBPERIOD] = Spec(SUBPERIOD, clock, "subperiod", weights="none", rule="all")
    specs[CONTINUATION] = Spec(CONTINUATION, clock, "continuation", weights="none", rule="all")
    specs["rank_sets"] = Spec("rank_sets", clock, "ranks", weights="none", rule="full_n20")
    specs["shrinkage"] = Spec("shrinkage", clock, "shrinkage")
    specs["time_to_detect"] = Spec("time_to_detect", clock, "detect", weights="none", winsorized=False, rule="all")
    return specs


# ---------------------------------------------------------------- frame -> panel

def winsorize(panel: pd.DataFrame, cutoffs: dict[str, tuple[float, float]], y: str = "y_h20") -> pd.DataFrame:
    """Clip y at the period's pooled (p1, p99) in bp (plan §3); periods without cutoffs untouched."""
    out = panel.copy()
    yv = out[y].to_numpy(dtype=float).copy()
    per = out["period"].to_numpy()
    for p, (lo, hi) in cutoffs.items():
        m = per == p
        yv[m] = np.clip(yv[m], lo, hi)
    out[y] = yv
    return out


def assign_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Period labels for the split: AB (as stored), ODDEVEN (odd years = A, even = B; events whose
    20-session window crosses a year boundary purged), WITHINB (2021-2022 = A, 2023-2025 = B, period-B rows only)."""
    if split == "AB":
        return frame
    out = frame.copy()
    yr = pd.Series(_dates(out["anchor_date"]).astype("datetime64[Y]").astype(int) + 1970, index=out.index) if "anchor_year" not in out.columns else out["anchor_year"].astype(int)
    if split == "ODDEVEN":
        end_year = pd.Series(_dates(out["window_end_date"]).astype("datetime64[Y]").astype(int) + 1970, index=out.index)
        out = out.loc[(end_year == yr) & out["period"].isin(["A", "B"])].copy()
        out["period"] = np.where((yr.loc[out.index] % 2) == 1, "A", "B")
        return out
    if split == "WITHINB":
        out = out.loc[out["period"] == "B"].copy()
        out["period"] = np.where(yr.loc[out.index] <= 2022, "A", "B")
        return out
    raise ValueError(f"unknown split {split!r}")


def apply_dup(frame: pd.DataFrame, dup: str) -> pd.DataFrame:
    """default: drop events whose EVERY line is a cross-source duplicate (plan §2; P only by construction,
    0 T episodes); keepdups: nothing; official: official-source events only."""
    if dup == "keepdups":
        return frame
    if dup == "default":
        return frame.loc[~frame["dup_all_lines"].astype(bool)]
    if dup == "official":
        return frame.loc[frame["record_source"] == "official"]
    raise ValueError(f"unknown duplicate treatment {dup!r}")


def apply_direction(frame: pd.DataFrame, direction: str, ycol: str) -> pd.DataFrame:
    """purchase / sale: that direction only; signed: purchases as they are, sales with the outcome negated."""
    if "direction" not in frame.columns:
        if direction == "purchase":
            return frame
        raise ValueError("frame has no direction column")
    if direction in ("purchase", "sale"):
        return frame.loc[frame["direction"] == direction]
    if direction == "signed":
        out = frame.loc[frame["direction"].isin(["purchase", "sale"])].copy()
        sign = np.where(out["direction"].to_numpy() == "sale", -1.0, 1.0)
        out[ycol] = pd.to_numeric(out[ycol], errors="coerce").to_numpy(dtype=float) * sign
        return out
    raise ValueError(f"unknown direction {direction!r}")


def _dates(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).to_numpy(dtype="datetime64[ns]") if s.dtype == object else s.to_numpy(dtype="datetime64[ns]")


def _months(s: pd.Series) -> np.ndarray:
    """Months since 1970 of a date column (integer, fast)."""
    return _dates(s).astype("datetime64[M]").astype(np.int64)


def collapse_hsm(frame: pd.DataFrame) -> pd.DataFrame:
    """One episode per household x PERMNO x anchor month, keeping the first anchor's row (plan §7)."""
    month = _months(frame["anchor_date"])
    order = frame.assign(_m=month).sort_values(["household_id", "permno", "_m", "anchor_date", "event_id"], kind="stable")
    keep = ~order.duplicated(["household_id", "permno", "_m"], keep="first")
    return order.loc[keep].drop(columns="_m")


def eligible_households(frame: pd.DataFrame, rule: str) -> pd.Index:
    """Households passing the rule on the frame's own period labels (R0/R1/R2 on A vs B; full_n20 on A u B)."""
    ab = frame.loc[frame["period"].isin(["A", "B"])]
    if rule == "all":
        return pd.Index(sorted(ab["household_id"].unique()))
    if rule.startswith("full_n"):
        n = ab.groupby("household_id").size()
        return pd.Index(sorted(n.index[n >= int(rule[6:])]))
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    na, qa, sa, nb, qb = RULES[rule]
    g = period_activity(ab)
    a, b = _period_rows(g, "A"), _period_rows(g, "B")
    ok_a = a.index[(a["n"] >= na) & (a["q"] >= qa) & (a["s"] >= sa)]
    ok_b = b.index[(b["n"] >= nb) & (b["q"] >= qb)]
    return pd.Index(sorted(ok_a.intersection(ok_b)))


def _period_rows(g: pd.DataFrame, period: str) -> pd.DataFrame:
    if period not in g.index.get_level_values("period"):
        return g.iloc[:0].droplevel("period")
    return g.xs(period, level="period")


def period_activity(ab: pd.DataFrame) -> pd.DataFrame:
    """Per household x period: purchases (n), distinct quarters (q), distinct securities (s); index (household_id, period)."""
    hh_codes, hh = pd.factorize(ab["household_id"], sort=True)
    per_codes, per = pd.factorize(ab["period"], sort=True)
    key = hh_codes * len(per) + per_codes
    size = len(hh) * len(per)
    n = np.bincount(key, minlength=size)
    q = _months(ab["anchor_date"]) // 3
    q_codes = pd.factorize(q)[0]
    s_codes = pd.factorize(ab["permno"])[0] if "permno" in ab.columns else np.zeros(len(ab), dtype=int)
    nq = np.bincount(np.unique(key * (q_codes.max() + 1) + q_codes) // (q_codes.max() + 1), minlength=size) if len(ab) else n
    ns = np.bincount(np.unique(key * (s_codes.max() + 1) + s_codes) // (s_codes.max() + 1), minlength=size) if len(ab) else n
    idx = pd.MultiIndex.from_product([hh, per], names=["household_id", "period"])
    out = pd.DataFrame({"n": n, "q": nq, "s": ns}, index=idx)
    return out.loc[out["n"] > 0]


def continuation_rate(frame: pd.DataFrame, rule: str = "R1") -> dict:
    """Share of households meeting the period-A half of the rule that also meet the period-B half."""
    na, qa, sa, nb, qb = RULES[rule]
    g = period_activity(frame.loc[frame["period"].isin(["A", "B"])])
    a = _period_rows(g, "A")
    ok_a = a.index[(a["n"] >= na) & (a["q"] >= qa) & (a["s"] >= sa)]
    b = _period_rows(g, "B").reindex(ok_a).fillna(0)
    ok_b = b.index[(b["n"] >= nb) & (b["q"] >= qb)]
    return {"n_eligible_A": int(len(ok_a)), "n_continuing": int(len(ok_b)),
            "continuation_rate": float(len(ok_b) / len(ok_a)) if len(ok_a) else np.nan}


def to_panel(frame: pd.DataFrame, spec: Spec, cutoffs: dict[str, tuple[float, float]] | None, draw: int = 0) -> pd.DataFrame:
    """Harness panel for `spec`: direction, non-finite outcomes dropped (so eligibility counts finite
    outcomes only), then dup treatment, split, anchor-window exclusion (on the frame's own `anchor_date`, i.e. the
    shifted anchor of a placebo draw), household x security x month collapse, eligibility, winsorization.

    The dropped count is in `panel.attrs['n_dropped_nonfinite']`. `cutoffs` are the pooled (p1, p99) per
    period for the spec's outcome and clock; required when `spec.winsorized`. The selection is one boolean
    mask over numpy columns (the same rules as `apply_direction`, `apply_dup`, `assign_split`)."""
    ycol = OUTCOMES[spec.outcome]
    y = pd.to_numeric(frame[ycol], errors="coerce").to_numpy(dtype=float).copy()
    n = len(y)
    mask = np.ones(n, dtype=bool)
    if "direction" in frame.columns:
        dirn = frame["direction"].to_numpy()
        if spec.direction in ("purchase", "sale"):
            mask &= dirn == spec.direction
        elif spec.direction == "signed":
            mask &= (dirn == "purchase") | (dirn == "sale")
            y = y * np.where(dirn == "sale", -1.0, 1.0)
        else:
            raise ValueError(f"unknown direction {spec.direction!r}")
    elif spec.direction != "purchase":
        raise ValueError("frame has no direction column")
    ok = np.isfinite(y)
    n_dropped = int((mask & ~ok).sum())
    mask &= ok
    if spec.dup == "default":
        mask &= ~frame["dup_all_lines"].to_numpy(dtype=bool)
    elif spec.dup == "official":
        mask &= frame["record_source"].to_numpy() == "official"
    elif spec.dup != "keepdups":
        raise ValueError(f"unknown duplicate treatment {spec.dup!r}")
    if spec.exclude:
        if spec.exclude not in HOUSEHOLD_EXCLUSIONS:
            raise ValueError(f"household exclusion {spec.exclude!r} is not registered")
        mask &= ~np.isin(frame["household_id"].to_numpy().astype(str), list(HOUSEHOLD_EXCLUSIONS[spec.exclude]))
    period = frame["period"].to_numpy().astype(object)
    if spec.split != "AB":
        yr = (frame["anchor_year"].to_numpy(dtype=int) if "anchor_year" in frame.columns
              else _dates(frame["anchor_date"]).astype("datetime64[Y]").astype(int) + 1970)
        if spec.split == "ODDEVEN":
            end_year = _dates(frame["window_end_date"]).astype("datetime64[Y]").astype(int) + 1970
            mask &= (end_year == yr) & ((period == "A") | (period == "B"))
            period = np.where(yr % 2 == 1, "A", "B").astype(object)
        elif spec.split == "WITHINB":
            mask &= period == "B"
            period = np.where(yr <= 2022, "A", "B").astype(object)
        else:
            raise ValueError(f"unknown split {spec.split!r}")
    mask &= (period == "A") | (period == "B")
    if spec.period_subset in ("A", "B"):
        mask &= period == spec.period_subset
    elif spec.period_subset != "AB":
        raise ValueError(f"unknown period subset {spec.period_subset!r}")
    n_dropped_anchors = 0
    if spec.exclude_anchor_window:
        if spec.exclude_anchor_window not in ANCHOR_WINDOWS:
            raise ValueError(f"unknown anchor window {spec.exclude_anchor_window!r}")
        lo, hi = (np.datetime64(d, "ns") for d in ANCHOR_WINDOWS[spec.exclude_anchor_window])
        ad = _dates(frame["anchor_date"])
        in_window = (ad >= lo) & (ad < hi)
        n_dropped_anchors = int((mask & in_window).sum())
        mask &= ~in_window
    f = pd.DataFrame({"event_id": frame["event_id"].to_numpy()[mask], "household_id": frame["household_id"].to_numpy().astype(str)[mask],
                      "period": period[mask], "y_h20": y[mask], "anchor_date": _dates(frame["anchor_date"])[mask]})
    if "sigma_event" in frame.columns:
        f["sigma_event"] = pd.to_numeric(frame["sigma_event"], errors="coerce").to_numpy(dtype=float)[mask]
    if "permno" in frame.columns:
        f["permno"] = frame["permno"].to_numpy()[mask]
    if spec.collapse_hsm:
        f = collapse_hsm(f)
    hh = eligible_households(f, spec.rule)
    f = f.loc[f["household_id"].isin(hh)].reset_index(drop=True)
    panel = pd.DataFrame({"draw": int(draw), "event_id": f["event_id"], "household_id": f["household_id"], "clock": spec.clock,
                          "period": f["period"], "y_h20": f["y_h20"], "anchor_date": f["anchor_date"]})
    for c in ("sigma_event", "permno"):
        if c in f.columns:
            panel[c] = f[c]
    if spec.winsorized:
        if cutoffs is None:
            raise ValueError(f"{spec.name}: winsorized spec needs pooled cutoffs")
        panel = winsorize(panel, cutoffs)
    panel.attrs["n_dropped_nonfinite"] = n_dropped
    panel.attrs["n_dropped_anchor_window"] = n_dropped_anchors
    panel.attrs["n_households_eligible"] = int(len(hh))
    return panel
