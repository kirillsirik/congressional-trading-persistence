"""data/v4/results.csv -> paper/generated/results_macros.tex with plain macro names.

Every macro is written as `\\providecommand{\\Name}{}\\renewcommand{\\Name}{value}` so the file works with
or without a placeholder file loaded before it. Values carry no unit and no percent sign; a value the
results file does not hold renders as `[PENDING]`.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
MACROS_TEX = REPO_ROOT / "paper" / "generated" / "results_macros.tex"
CLOCK = {"T": "Transaction", "P": "Public"}
WEIGHT = {"event_share": "", "equal_household": "EqualHousehold", "pair": "Pair"}
ROBUSTNESS = {"no_activity_control": "NoActivityControl", "loose_rule": "LooseRule", "strict_rule": "StrictRule", "odd_even": "OddEven",
              "official_only": "OfficialOnly", "keep_duplicates": "KeepDuplicates", "market_adjusted": "MarketAdjusted", "no_alpha": "NoAlpha",
              "horizon_10": "HorizonTen", "horizon_60": "HorizonSixty", "raw_outcome": "RawOutcome", "one_per_security_month": "OnePerSecurityMonth",
              "within_second_period": "WithinSecondPeriod", "sales": "Sales", "signed_all": "SignedAll"}
THETA = {2.0: "Two", 5.0: "Five", 10.0: "Ten", 20.0: "Twenty"}
#: sample variants of the purchase mean, tail and best-household rows (results.csv variant prefix -> macro suffix)
SAMPLE = {"": "", "official_only_": "OfficialOnly", "excl_two_vendor_households_": "ExclTwo"}
TAIL_SAMPLE = {**SAMPLE, "period_A_": "PeriodA", "period_B_": "PeriodB", "excl_covid_": "ExclCovid",
               "excl_covid_period_A_": "ExclCovidPeriodA", "excl_covid_period_B_": "ExclCovidPeriodB"}
NPY = {10: "Ten", 25: "TwentyFive", 50: "Fifty", 100: "Hundred", 250: "TwoFifty", 500: "FiveHundred"}
PENDING = "[PENDING]"


# ---------------------------------------------------------------- formatting

def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x)


MINUS = r"\ensuremath{-}"  # a proper minus in text and inside $...$ alike (a bare "-" is a hyphen in text; "$-$" breaks math mode)


def fmt_num(x, digits: int = 1) -> str:
    if _nan(x):
        return PENDING
    s = f"{float(x):,.{digits}f}"
    return MINUS + s[1:] if s.startswith("-") else s



def fmt_sci(x, digits: int = 0) -> str:
    """Scientific notation safe in text and math, with the requested mantissa precision."""
    if _nan(x):
        return PENDING
    x = float(x)
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    power = f"10^{{{int(exponent)}}}"
    if float(mantissa) == 1:
        expression = power
    elif float(mantissa) == -1:
        expression = "-" + power
    else:
        expression = mantissa + r"\times " + power
    return r"\ensuremath{" + expression + "}"


def fmt_pct_of_bp(x, digits: int = 1) -> str:
    """A bp value on the percent scale (bp / 100), one decimal; no percent sign."""
    return PENDING if _nan(x) else fmt_num(float(x) / 100.0, digits)


def fmt_int(x) -> str:
    return PENDING if _nan(x) else f"{int(round(float(x))):,d}"


def fmt_p(x) -> str:
    if _nan(x):
        return PENDING
    x = float(x)
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def fmt_share(x, digits: int = 3) -> str:
    return PENDING if _nan(x) else f"{float(x):.{digits}f}"


def fmt_years(x) -> str:
    if _nan(x):
        return PENDING
    x = float(x)
    return f"{x:,.0f}" if x >= 100 else f"{x:.1f}"


def fmt_yes_no(x) -> str:
    return PENDING if _nan(x) else ("yes" if float(x) >= 0.5 else "no")


def fmt_level(x) -> str:
    """A bootstrap level such as 0.99 or 0.999 (up to three decimals, trailing zeros dropped)."""
    return PENDING if _nan(x) else f"{float(x):.3f}".rstrip("0").rstrip(".")


def fmt_level_pct(x) -> str:
    """The same level on the percent scale (99, 99.9); no percent sign."""
    return PENDING if _nan(x) else f"{100 * float(x):.1f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- lookup

class Results:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def row(self, clock: str, analysis: str, variant: str, statistic: str) -> pd.Series | None:
        m = (self.df["clock"].fillna("") == clock) & (self.df["analysis"] == analysis) & (self.df["variant"] == variant) & (self.df["statistic"] == statistic)
        r = self.df.loc[m]
        if len(r) > 1:
            raise ValueError(f"results hold {len(r)} rows for ({clock}, {analysis}, {variant}, {statistic})")
        return r.iloc[0] if len(r) == 1 else None

    def value(self, clock: str, analysis: str, variant: str, statistic: str, col: str = "value"):
        r = self.row(clock, analysis, variant, statistic)
        return float("nan") if r is None else r[col]


# ---------------------------------------------------------------- emitters

def _tau(res: Results, clock: str, analysis: str, variant: str, prefix: str) -> dict[str, str]:
    v = lambda stat, col="value": res.value(clock, analysis, variant, stat, col)  # noqa: E731
    return {
        f"{prefix}Sq": fmt_num(v("tau2"), 0), f"{prefix}SqSE": fmt_num(v("tau2", "se_block"), 0),
        f"{prefix}SqSEDay": fmt_num(v("tau2_se_day"), 0), f"{prefix}SqPDay": fmt_p(v("p_day")),
        f"{prefix}SqPlaceboMean": fmt_num(v("tau2_placebo_mean"), 0), f"{prefix}SqPlaceboNinetyFive": fmt_num(v("tau2_placebo_q95"), 0),
        f"{prefix}": fmt_num(v("tau")), f"{prefix}Annualized": fmt_num(v("tau_annualized"), 0),
        f"{prefix}AnnualizedPct": fmt_pct_of_bp(v("tau_annualized")),
        f"{prefix}BootLo": fmt_num(v("tau", "boot_lo")), f"{prefix}BootHi": fmt_num(v("tau", "boot_hi")),
        f"{prefix}Upper": fmt_num(v("tau_upper")), f"{prefix}UpperAnnualized": fmt_num(v("tau_upper_annualized"), 0),
        f"{prefix}UpperAnnualizedPct": fmt_pct_of_bp(v("tau_upper_annualized")),
        f"{prefix}PlaceboP": fmt_p(v("tau2", "placebo_p")), f"{prefix}TBlock": fmt_num(v("t_block"), 2), f"{prefix}PBlock": fmt_p(v("p_block")),
        f"{prefix}Households": fmt_int(v("tau2", "n_households")), f"{prefix}Events": fmt_int(v("tau2", "n_events")),
        f"{prefix}PlaceboPTiming": fmt_p(v("tau2_placebo_p_timing")),
        f"{prefix}SqPlaceboMeanTiming": fmt_num(v("tau2_placebo_mean_timing"), 0),
        f"{prefix}SqPlaceboNinetyFiveTiming": fmt_num(v("tau2_placebo_q95_timing"), 0),
        f"{prefix}PlaceboPHolm": fmt_p(v("placebo_p_holm")), f"{prefix}PBlockHolm": fmt_p(v("p_block_holm")),
        f"{prefix}UpperCalibrated": fmt_num(v("tau_upper_calibrated")),
        f"{prefix}UpperCalibratedAnnualized": fmt_num(v("tau_upper_calibrated_annualized"), 0),
        f"{prefix}UpperCalibratedAnnualizedPct": fmt_pct_of_bp(v("tau_upper_calibrated_annualized")),
        f"{prefix}UpperInjection": fmt_num(v("tau_upper_injection")),
        f"{prefix}UpperInjectionAnnualized": fmt_num(v("tau_upper_injection_annualized"), 0),
        f"{prefix}UpperInjectionAnnualizedPct": fmt_pct_of_bp(v("tau_upper_injection_annualized")),
        f"{prefix}UpperInjectionMCSE": fmt_num(v("tau_upper_injection_mc_se")),
    }


INJECTION_KEYS = ("UpperInjection", "UpperInjectionAnnualized", "UpperInjectionAnnualizedPct", "UpperInjectionMCSE")


def _headline(res: Results, clock: str) -> dict[str, str]:
    C = CLOCK[clock]
    out = {}
    for variant, w in WEIGHT.items():
        cells = _tau(res, clock, "tau", variant, "Tau")
        for k, val in cells.items():
            name = ("ResTauSq" + w + C + k[5:]) if k.startswith("TauSq") else ("ResTau" + w + C + k[3:])
            out[name] = val
    # purchases of the persistence sample by period (sample rows; the same rows the tau estimate uses)
    out[f"ResTau{C}EventsA"] = fmt_int(res.value(clock, "sample", "default", "n_events_persistence_A"))
    out[f"ResTau{C}EventsB"] = fmt_int(res.value(clock, "sample", "default", "n_events_persistence_B"))
    level = res.value(clock, "tau", "event_share", "calibrated_level")
    out[f"ResCalibratedLevel{C}"], out[f"ResCalibratedLevelPct{C}"] = fmt_level(level), fmt_level_pct(level)
    if clock == "T":
        out.update({"ResTauPlaceboP": out["ResTauTransactionPlaceboP"], "ResTauUpper": out["ResTauTransactionUpper"],
                    "ResTauUpperAnnualized": out["ResTauTransactionUpperAnnualized"],
                    "ResTauUpperAnnualizedPct": out["ResTauTransactionUpperAnnualizedPct"],
                    "ResTauUpperCalibrated": out["ResTauTransactionUpperCalibrated"],
                    "ResTauUpperCalibratedAnnualized": out["ResTauTransactionUpperCalibratedAnnualized"],
                    "ResTauUpperCalibratedAnnualizedPct": out["ResTauTransactionUpperCalibratedAnnualizedPct"]})
        out.update({f"ResTau{k}": out[f"ResTauTransaction{k}"] for k in INJECTION_KEYS})
    return out


def _robustness(res: Results, clock: str) -> dict[str, str]:
    C = CLOCK[clock]
    out = {}
    for variant, name in ROBUSTNESS.items():
        cells = _tau(res, clock, "robustness", variant, "Tau")
        base = f"ResRob{name}{C}"
        out.update({f"{base}TauSq": cells["TauSq"], f"{base}Tau": cells["Tau"], f"{base}Upper": cells["TauUpper"],
                    f"{base}UpperAnnualized": cells["TauUpperAnnualized"], f"{base}UpperAnnualizedPct": cells["TauUpperAnnualizedPct"],
                    f"{base}PlaceboP": cells["TauPlaceboP"], f"{base}BootLo": cells["TauBootLo"], f"{base}BootHi": cells["TauBootHi"],
                    f"{base}Households": cells["TauHouseholds"], f"{base}PBlock": cells["TauPBlock"],
                    f"{base}PlaceboPTiming": cells["TauPlaceboPTiming"], f"{base}PlaceboPHolm": cells["TauPlaceboPHolm"],
                    f"{base}PBlockHolm": cells["TauPBlockHolm"], f"{base}UpperCalibrated": cells["TauUpperCalibrated"],
                    f"{base}UpperCalibratedAnnualized": cells["TauUpperCalibratedAnnualized"],
                    f"{base}UpperCalibratedAnnualizedPct": cells["TauUpperCalibratedAnnualizedPct"]})
        if cells["TauUpperInjection"] != PENDING:  # only the variants the injection curves were computed on
            out.update({f"{base}{k}": cells[f"Tau{k}"] for k in INJECTION_KEYS})
    return out


#: the fields of `_tau` the no-control rows publish (the injection bound was computed for the headline only)
NO_CONTROL_KEYS = ("TauSq", "TauSqSE", "TauSqPlaceboMean", "TauSqPlaceboNinetyFive", "Tau", "TauAnnualized", "TauAnnualizedPct",
                   "TauBootLo", "TauBootHi", "TauUpper", "TauUpperAnnualized", "TauUpperAnnualizedPct", "TauPlaceboP",
                   "TauPlaceboPTiming", "TauTBlock", "TauPBlock", "TauHouseholds", "TauEvents", "TauUpperCalibrated")
#: subperiod index words (the matrix is 3x3)
SUBPERIOD_WORDS = ("One", "Two", "Three")
SUBPERIOD_NAMES = ("first", "second", "third")
SUBPERIOD_PAIRS = ((0, 1), (0, 2), (1, 2))
#: continuation outcomes: results.csv variant -> macro infix (the plan's rule indicator carries `Rule`)
CONTINUATION = {"any_purchase": "", "eligibility_rule": "Rule"}


def _tau_no_control(res: Results, clock: str) -> dict[str, str]:
    """The genuine no-control estimand: the w-weighted covariance of the unresidualized period means."""
    C = CLOCK[clock]
    out = {}
    for variant, w in WEIGHT.items():
        cells = _tau(res, clock, "tau_no_control", variant, "Tau")
        for k in NO_CONTROL_KEYS:
            out[("ResTauNoControlSq" + w + C + k[5:]) if k.startswith("TauSq") else ("ResTauNoControl" + w + C + k[3:])] = cells[k]
        v = lambda stat, var=variant: res.value(clock, "tau_no_control", var, stat)  # noqa: E731
        out[f"ResTauNoControl{w}{C}DiffHeadline"] = fmt_num(v("tau_diff_headline"))
        out[f"ResTauNoControlSq{w}{C}DiffHeadline"] = fmt_num(v("tau2_diff_headline"), 0)
        out[f"ResTauNoControl{w}{C}ActivityCorrelation"] = fmt_num(v("activity_correlation"), 3)
    return out


def _subperiod(res: Results, clock: str) -> dict[str, str]:
    """The three-subperiod covariance matrix (appendix)."""
    C = CLOCK[clock]
    v = lambda stat, col="value": res.value(clock, "subperiod", "default", stat, col)  # noqa: E731
    out = {f"ResSubperiodHouseholds{C}": fmt_int(v("n_households")),
           f"ResSubperiodHouseholdsPool{C}": fmt_int(v("n_households_panel")),
           f"ResSubperiodEventsDropped{C}": fmt_int(v("n_events_dropped"))}
    for i, word in enumerate(SUBPERIOD_WORDS):
        name = SUBPERIOD_NAMES[i]
        out[f"ResSubperiodEvents{word}{C}"] = fmt_int(v(f"n_events_{name}"))
        out[f"ResSubperiodVar{word}{C}"] = fmt_num(v(f"var_{name}"), 0)
        out[f"ResSubperiodVar{word}{C}PlaceboP"] = fmt_p(v(f"var_{name}", "placebo_p"))
    for i, j in SUBPERIOD_PAIRS:
        pair, stat = SUBPERIOD_WORDS[i] + SUBPERIOD_WORDS[j], f"{SUBPERIOD_NAMES[i]}_{SUBPERIOD_NAMES[j]}"
        out[f"ResSubperiodCov{pair}{C}"] = fmt_num(v(f"cov_{stat}"), 0)
        out[f"ResSubperiodCov{pair}{C}PlaceboP"] = fmt_p(v(f"cov_{stat}", "placebo_p"))
        out[f"ResSubperiodCov{pair}{C}PlaceboMean"] = fmt_num(v(f"cov_{stat}_placebo_mean"), 0)
        out[f"ResSubperiodCov{pair}{C}PlaceboNinetyFive"] = fmt_num(v(f"cov_{stat}_placebo_q95"), 0)
        out[f"ResSubperiodCorr{pair}{C}"] = fmt_num(v(f"corr_{stat}"), 3)
    return out


def _continuation(res: Results, clock: str) -> dict[str, str]:
    """Continuation into the second period on the first-period mean given log activity (appendix)."""
    C = CLOCK[clock]
    out = {}
    for variant, infix in CONTINUATION.items():
        v = lambda stat, col="value", var=variant: res.value(clock, "continuation", var, stat, col)  # noqa: E731
        b = f"ResContinuation{infix}"
        out.update({f"{b}Slope{C}": fmt_num(v("ols_mean_a"), 3), f"{b}SlopeSE{C}": fmt_num(v("ols_mean_a", "se_block"), 3),
                    f"{b}ActivitySlope{C}": fmt_num(v("ols_log_activity_a"), 3),
                    f"{b}ActivitySlopeSE{C}": fmt_num(v("ols_log_activity_a", "se_block"), 3),
                    f"{b}LogitSlope{C}": fmt_num(v("logit_mean_a"), 3), f"{b}LogitSlopeSE{C}": fmt_num(v("logit_mean_a", "se_block"), 3),
                    f"{b}LogitActivitySlope{C}": fmt_num(v("logit_log_activity_a"), 3),
                    f"{b}LogitActivitySlopeSE{C}": fmt_num(v("logit_log_activity_a", "se_block"), 3),
                    f"{b}Share{C}": fmt_share(v("share_continuing"), 2), f"{b}Households{C}": fmt_int(v("n_households_regression"))})
    return out


def _scatter(res: Results, clock: str) -> dict[str, str]:
    """The fitted slope of the household scatter figure."""
    C = CLOCK[clock]
    v = lambda stat, col="value": res.value(clock, "household_scatter", "event_share", stat, col)  # noqa: E731
    return {f"ResScatterSlope{C}": fmt_num(v("slope"), 2), f"ResScatterSlopeSE{C}": fmt_num(v("slope", "se_block"), 2),
            f"ResScatterSlopeBootLo{C}": fmt_num(v("slope", "boot_lo"), 2), f"ResScatterSlopeBootHi{C}": fmt_num(v("slope", "boot_hi"), 2),
            f"ResScatterVarX{C}": fmt_num(v("var_x"), 0), f"ResScatterHouseholds{C}": fmt_int(v("n_households"))}


def _influence(res: Results, clock: str) -> dict[str, str]:
    C = CLOCK[clock]
    out = {}
    for variant, name in (("leave_one_household_out", "LeaveHouseholdOut"), ("leave_one_year_out", "LeaveYearOut")):
        out.update({f"ResTau{name}{C}Min": fmt_num(res.value(clock, "influence", variant, "tau_min")),
                    f"ResTau{name}{C}Max": fmt_num(res.value(clock, "influence", variant, "tau_max")),
                    f"ResTau{name}{C}Refits": fmt_int(res.value(clock, "influence", variant, "refits"))})
    return out


def _best(res: Results, clock: str) -> dict[str, str]:
    C = CLOCK[clock]
    out = {}
    fmt = {"max_mean": (fmt_num, "BestMean"), "max_t": (lambda x: fmt_num(x, 2), "BestT"), "top_decile_mean": (fmt_num, "TopDecileMean"),
           "n_p05": (fmt_int, "CountSignificant")}
    for pre, suffix in SAMPLE.items():
        cen, tim = pre + "vs_centered_placebo", pre + "vs_timing_placebo"
        for stat, (f, name) in fmt.items():
            b = f"Res{name}{C}{suffix}"
            out[b] = f(res.value(clock, "best_household", cen, stat))
            out[f"{b}PlaceboP"] = fmt_p(res.value(clock, "best_household", cen, stat, "placebo_p"))
            out[f"{b}TimingPlaceboP"] = fmt_p(res.value(clock, "best_household", tim, stat, "placebo_p"))
            out[f"{b}PlaceboNinetyFive"] = f(res.value(clock, "best_household", cen, f"{stat}_placebo_q95"))
            out[f"{b}PlaceboMean"] = (fmt_num if stat == "n_p05" else f)(res.value(clock, "best_household", cen, f"{stat}_placebo_mean"))
            out[f"{b}TimingPlaceboNinetyFive"] = f(res.value(clock, "best_household", tim, f"{stat}_placebo_q95"))
            out[f"{b}TimingPlaceboMean"] = (fmt_num if stat == "n_p05" else f)(res.value(clock, "best_household", tim, f"{stat}_placebo_mean"))
        out[f"ResBestHouseholds{C}{suffix}"] = fmt_int(res.value(clock, "best_household", cen, "max_mean", "n_households"))
    out[f"ResBestMean{C}Raw"] = fmt_num(res.value(clock, "best_household", "raw_outcome_vs_centered_placebo", "max_mean"))
    out[f"ResBestMean{C}RawPlaceboP"] = fmt_p(res.value(clock, "best_household", "raw_outcome_vs_centered_placebo", "max_mean", "placebo_p"))
    return out


def _tail(res: Results, clock: str) -> dict[str, str]:
    C = CLOCK[clock]
    out = {}
    for pre, suffix in TAIL_SAMPLE.items():
        cen, tim = pre + "vs_centered_placebo", pre + "vs_timing_placebo"
        for z, name in ((2, "Two"), (3, "Three")):
            # `\ResTailShare<Z>` is the share above z (its `Above<Z>` twin is an alias); `Below<Z>` the share below -z
            for stat, names in ((f"share_gt_{z}", (name, f"Above{name}")), (f"share_lt_minus_{z}", (f"Below{name}",))):
                cells = {"": fmt_share(res.value(clock, "tail", cen, stat), 4),
                         "PlaceboMean": fmt_share(res.value(clock, "tail", cen, f"{stat}_placebo_mean"), 4),
                         "PlaceboP": fmt_p(res.value(clock, "tail", cen, stat, "placebo_p")),
                         "TimingPlaceboP": fmt_p(res.value(clock, "tail", tim, stat, "placebo_p")),
                         "TimingPlaceboMean": fmt_share(res.value(clock, "tail", tim, f"{stat}_placebo_mean"), 4)}
                for n in names:
                    out.update({f"ResTailShare{n}{C}{suffix}{k}": val for k, val in cells.items()})
        out[f"ResTailEvents{C}{suffix}"] = fmt_int(res.value(clock, "tail", cen, "share_gt_2", "n_events"))
    return out


def _mean(res: Results, clock: str) -> dict[str, str]:
    C = CLOCK[clock]
    out = {}
    for pre, suffix in SAMPLE.items():
        vz, vp, b = pre + "vs_zero", pre + "vs_timing_placebo", f"ResMean{C}{suffix}"
        v = lambda s, col="value", var=vz: res.value(clock, "purchase_mean", var, s, col)  # noqa: E731
        out.update({b: fmt_num(v("mean")), f"{b}SE": fmt_num(v("mean", "se_block")), f"{b}TBlock": fmt_num(v("t_block"), 2),
                    f"{b}PBlock": fmt_p(v("p_block")), f"{b}Blocks": fmt_int(v("n_blocks")), f"{b}SEDay": fmt_num(v("se_day")),
                    f"{b}PDay": fmt_p(v("p_day")), f"{b}PlaceboMean": fmt_num(v("placebo_mean", var=vp)),
                    f"{b}PlaceboP": fmt_p(v("mean", "placebo_p", vp)), f"{b}Events": fmt_int(v("mean", "n_events")),
                    f"{b}Households": fmt_int(v("mean", "n_households"))})
    return out


def _ranks(res: Results, clock: str) -> dict[str, str]:
    C = CLOCK[clock]
    v = lambda s: res.value(clock, "rank_sets", "default", s)  # noqa: E731
    return {f"ResRankWidth{C}": fmt_share(v("median_width_over_H"), 2), f"ResRankPairsResolved{C}": fmt_share(v("share_pairs_resolved"), 2),
            f"ResRankDistinctFromMedian{C}": fmt_int(v("n_distinguishable_from_median")), f"ResRankSingletons{C}": fmt_int(v("n_singleton_sets")),
            f"ResRankHouseholds{C}": fmt_int(res.value(clock, "rank_sets", "default", "median_width_over_H", "n_households"))}


def _shrinkage(res: Results, clock: str) -> dict[str, str]:
    C = CLOCK[clock]
    v = lambda s: res.value(clock, "shrinkage", "default", s)  # noqa: E731
    return {f"ResShrinkTau{C}": fmt_num(v("tau_eb")), f"ResShrinkAtZero{C}": fmt_yes_no(v("at_zero_boundary")),
            f"ResShrinkBoundaryShare{C}": fmt_share(v("boundary_share")), f"ResShrinkForecastSlope{C}": fmt_num(v("forecast_slope"), 2),
            f"ResShrinkForecastSlopeSE{C}": fmt_num(v("forecast_slope_se"), 2), f"ResShrinkReliabilitySlope{C}": fmt_num(v("reliability_slope"), 2),
            f"ResShrinkReliabilitySlopeSE{C}": fmt_num(v("reliability_slope_se"), 2),
            f"ResShrinkHouseholds{C}": fmt_int(res.value(clock, "shrinkage", "default", "tau_eb", "n_households"))}


def _detect(res: Results, clock: str) -> dict[str, str]:
    C = CLOCK[clock]
    v = lambda s: res.value(clock, "time_to_detect", "default", s)  # noqa: E731
    out = {f"ResDetectSigma{C}": fmt_num(v("sigma"), 0), f"ResDetectDesignEffect{C}": fmt_num(v("design_effect"), 2),
           f"ResDetectHouseholds{C}": fmt_int(v("H")), f"ResDetectMedianPurchasesPerYear{C}": fmt_num(v("median_purchases_per_household_year"), 0),
           f"ResDetectSpanYears{C}": fmt_int(v("span_years"))}
    for tw in THETA.values():
        for nw in NPY.values():
            base = f"years_theta{tw}_n{nw}"
            out[f"ResDetectYears{tw}At{nw}{C}"] = fmt_years(v(base))
            out[f"ResDetectYears{tw}At{nw}BestOfH{C}"] = fmt_years(v(base + "_best_of_h"))
    return out


def _sample(res: Results, clock: str) -> dict[str, str]:
    C = CLOCK[clock]
    v = lambda s, var="default": res.value(clock, "sample", var, s)  # noqa: E731
    return {f"ResEvents{C}": fmt_int(v("n_events")), f"ResEventsA{C}": fmt_int(v("n_events_A")), f"ResEventsB{C}": fmt_int(v("n_events_B")),
            f"ResHouseholds{C}": fmt_int(v("n_households")), f"ResHouseholdsPersistence{C}": fmt_int(v("n_households_persistence")),
            f"ResHouseholdsFullTwenty{C}": fmt_int(v("n_households_full20")), f"ResDroppedDuplicates{C}": fmt_int(v("n_dropped_duplicates")),
            f"ResDroppedNonfinite{C}": fmt_int(v("n_dropped_nonfinite")), f"ResContinuationRate{C}": fmt_share(v("continuation_rate"), 2),
            f"ResTopTwoShareB{C}": fmt_share(v("top2_share_B"), 2), f"ResEffectiveHouseholdsB{C}": fmt_num(v("effective_households_B")),
            f"ResOfficialEvents{C}": fmt_int(v("n_events", "official_only")), f"ResOfficialHouseholds{C}": fmt_int(v("n_households", "official_only")),
            f"ResOfficialHouseholdsPersistence{C}": fmt_int(v("n_households_persistence", "official_only")),
            f"ResOfficialTopTwoShareB{C}": fmt_share(v("top2_share_B", "official_only"), 2),
            f"ResOfficialEventsPersistenceA{C}": fmt_int(v("n_events_persistence_A", "official_only")),
            f"ResOfficialEventsPersistenceB{C}": fmt_int(v("n_events_persistence_B", "official_only"))}


def _run(res: Results) -> dict[str, str]:
    h, d = res.row("", "run", "default", "head_sha"), res.row("", "run", "default", "date_utc")
    return {"ResRunHead": PENDING if h is None else str(h["notes"]), "ResRunDate": PENDING if d is None else str(d["notes"])}


EMITTERS = (_headline, _robustness, _influence, _best, _tail, _mean, _ranks, _shrinkage, _detect, _sample,
            _tau_no_control, _subperiod, _continuation, _scatter)


def emit(df: pd.DataFrame, clocks=("T", "P")) -> dict[str, str]:
    """Macro name -> formatted value for every documented macro (missing values render as [PENDING])."""
    res = Results(df)
    out: dict[str, str] = {}
    for clock in clocks:
        for fn in EMITTERS:
            out.update(fn(res, clock))
    out.update(_run(res))
    return out


def render(values: dict[str, str], header: str = "") -> str:
    lines = ["% Generated by trading_on_disclosure_v4.ledger.macros from data/v4/results.csv; do not edit.",
             "% Names: docs/v4/results_schema.md section 5. Works with or without a placeholder file loaded first."]
    if header:
        lines += [f"% {h}" for h in header.splitlines()]
    lines += [f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{values[k]}}}" for k in sorted(values)]
    return "\n".join(lines) + "\n"


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=True)
    for c in ("clock", "analysis", "variant", "statistic", "unit", "notes"):
        df[c] = df[c].fillna("").astype(str)
    return df


def write_macros(df: pd.DataFrame, out: Path = MACROS_TEX, header: str = "") -> tuple[Path, int]:
    """Write the macro file from a results frame; returns (path, number of macros)."""
    values = emit(df)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(values, header))
    return out, len(values)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="results.csv -> results_macros.tex")
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "data" / "v4" / "results.csv")
    ap.add_argument("--out", type=Path, default=MACROS_TEX)
    a = ap.parse_args(argv)
    path, n = write_macros(read_results(a.results), a.out)
    print(f"{path}: {n} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
