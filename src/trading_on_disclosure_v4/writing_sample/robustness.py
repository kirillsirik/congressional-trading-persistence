"""The 15 rows of Table F5 under both six-factor benchmarks on identical events.

python -m trading_on_disclosure_v4.writing_sample.robustness
For each return date and row: the original event-study specification and its factor-only counterpart (same
events, weights, controls and eligibility; benchmark-specific cutoffs; 10/60-session factor-only outcomes from the
SAME fitted loadings). Point covariance and conditional household x 20-session-block CGM SE with the original
estimators; a paired household bootstrap that reuses the row's original selections (seed_from(row spec name,
clock, 'household bootstrap'), 4,999 draws) for both benchmarks and refits both activity projections; the gap
with its paired interval and its conditional two-way SE from the difference of event-level scores. Sales and
signed trades also get the bilinear decomposition K11 = K00 + KDD + K0D + KD0 (raw and winsorized) with paired
percentile and delete-one-household intervals, and the purchase/sale cohort overlap. C14 compares the baseline and
vendor-duplicates designs byte for byte and, where identical, reuses the baseline selections.
Component `robustness_both_benchmarks`; tables under data/v4/writing_sample/computation_2026-11-11/robustness/.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import no_control as nc
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.inference import block_bootstrap as bb
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes.seeds import seed_from
from trading_on_disclosure_v4.writing_sample import common as cm
from trading_on_disclosure_v4.writing_sample import panels as pn

sys.path.insert(0, str(cm.REPO_ROOT / "scripts" / "coherence"))
from aggregate_checks import jackknife, moments  # noqa: E402

OUT = cm.OUT / "robustness"
N_BOOT = 4999
ROWS = [("baseline", "tau/event_share", "Principal specification"), ("equal_weights", "tau/equal_household", "Equal household weights"),
        ("product_weights", "tau/pair", "Product weights"), ("no_activity", "tau_no_control/event_share", "Without activity control"),
        ("five_each_period", "robustness/loose_rule", "Five purchases in each period"), ("stricter", "robustness/strict_rule", "Stricter eligibility"),
        ("odd_even", "robustness/odd_even", "Odd versus even years"), ("official_only", "robustness/official_only", "Official sources only"),
        ("vendor_duplicates", "robustness/keep_duplicates", "Vendor duplicates retained"), ("h10", "robustness/horizon_10", "10-day returns"),
        ("h60", "robustness/horizon_60", "60-day returns"), ("one_per_security_month", "robustness/one_per_security_month", "One purchase per household-security-month"),
        ("within_late", "robustness/within_second_period", "2021-2022 versus 2023-2025"), ("sales", "robustness/sales", "Sales only"),
        ("signed", "robustness/signed_all", "Signed purchases and sales")]
TARGETS_T = {"baseline": (7964, 38), "equal_weights": (7964, 38), "product_weights": (7964, 38), "no_activity": (7964, 38), "five_each_period": (8702, 49),
             "stricter": (7553, 31), "odd_even": (18995, 80), "official_only": (7898, 37), "vendor_duplicates": (7964, 38), "h10": (7964, 38), "h60": (7953, 38),
             "one_per_security_month": (6994, 37), "within_late": (3269, 21), "sales": (7233, 39), "signed": (15857, 48)}
WEIGHT_RULE = {"ES": "sqrt(n_A n_B)", "EH": "1", "PW": "n_A n_B"}


def frozen_row(clock: str, name: str) -> pd.Series:
    t = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "results.csv")
    analysis, variant = name.split("/")
    m = t[(t.clock == clock) & (t.analysis == analysis) & (t.variant == variant) & (t.statistic == "tau2")]
    if len(m) != 1:
        raise KeyError(f"{clock} {name}")
    return m.iloc[0]


def point_and_se(panel: pd.DataFrame, spec: sp.Spec, cal) -> dict:
    pb = rs.block_cluster(panel, cal)
    if spec.residualize_activity:
        r = cgm.tau2_i1(pb, spec.weights, anchor="anchor_blk", activity_control=spec.activity_control)
    else:
        r = nc.tau2_no_control_i1(pb, spec.weights, anchor="anchor_blk")
    table = tau_mod.household_table(panel, spec.weights, residualize_activity=spec.residualize_activity)
    ev = tau_mod.event_design(pb, table, anchor="anchor_blk")
    X = tau_mod.design_matrix(ev, spec.activity_control) if spec.residualize_activity else np.column_stack([np.ones(len(ev)), ev["x"].to_numpy(float)])
    w = ev["event_weight"].to_numpy(float)
    y = ev["y"].to_numpy(float)
    bread = np.linalg.inv(X.T @ (w[:, None] * X))
    beta = bread @ (X.T @ (w * y))
    e = y - X @ beta
    scores = (X @ bread[:, 1]) * w * e * r["var_wx"]
    return {"tau2": r["tau2_hat"], "se_block": r["var_wx"] * r["se_b"], "dof": r["dof"], "psd_corrected": bool(r["psd_corrected"]),
            "var_wx": r["var_wx"], "slope": r["b_hat"], "se_slope": r["se_b"], "n_events": int(r["n_events_a"] + r["n_events_b"]),
            "n_households": int(r["n_households"]), "n_blocks": int(r["n_anchor_dates"]), "table": table, "scores": scores,
            "household": ev["household_id"].to_numpy(), "block": ev["anchor_blk"].to_numpy(), "k": X.shape[1], "events_B": len(ev)}


def two_way_scalar(scores, household, block, k) -> dict:
    n = len(scores)
    cl = pd.DataFrame({"h": household, "b": block, "s": scores})
    parts, counts = [], []
    for cols in (["h"], ["b"], ["h", "b"]):
        sums = cl.groupby(cols).s.sum()
        G = len(sums)
        parts.append(G / (G - 1) * (n - 1) / (n - k) * float(np.sum(sums.to_numpy() ** 2)))
        counts.append(G)
    v = parts[0] + parts[1] - parts[2]
    return {"variance_raw": v, "se": float(np.sqrt(v)) if v > 0 else np.nan, "dof": min(counts[:2]) - 1, "household_clusters": counts[0], "block_clusters": counts[1]}


def draws_for(table: tau_mod.HouseholdTable, mult: np.ndarray, residualize: bool) -> np.ndarray:
    s_a, s_b = table.ybar_a * table.n_a, table.ybar_b * table.n_b
    f = bb._tau2_from_sums if residualize else nc._tau2_from_sums_no_control
    return np.array([f(table.n_a, s_a, table.n_b, s_b, table.w * m)[0] for m in mult])


def multiplicities(seed: int, n_h: int, n_boot: int = N_BOOT) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([bb._household_draw(rng, n_h) for _ in range(n_boot)])


def pct(v: np.ndarray) -> tuple[float, float, int]:
    f = v[np.isfinite(v)]
    lo, hi = np.quantile(f, [0.025, 0.975]) if len(f) else (np.nan, np.nan)
    return float(lo), float(hi), int(len(f))


def design_hash(panel: pd.DataFrame, table: tau_mod.HouseholdTable, cutoffs) -> dict:
    return {"events": cm.sha256_array(panel["event_id"].astype(str).to_numpy()), "outcomes": cm.sha256_array(panel["y_h20"].to_numpy(float)),
            "households": cm.sha256_array(np.asarray(table.household_id).astype(str)), "weights": cm.sha256_array(table.w),
            "activity": cm.sha256_array(table.log_n_a), "periods": cm.sha256_array(panel["period"].astype(str).to_numpy()),
            "cutoffs": cm.sha256_array(np.array(sorted((k, *v) for k, v in (cutoffs or {}).items()), dtype=object))}


def run_row(ctx, clock: str, label: str, name: str, rows: cm.Rows, log) -> dict:
    s1 = sp.analysis_specs(clock)[name]
    s0 = pn.factor_only(s1)
    p1, p0 = ctx.panel(s1), ctx.panel(s0)
    if not (p1[["event_id", "household_id", "period"]].equals(p0[["event_id", "household_id", "period"]])):
        raise AssertionError(f"{clock}/{label}: benchmarks select different events")
    r1, r0 = point_and_se(p1, s1, ctx.calendar), point_and_se(p0, s0, ctx.calendar)
    if not (np.array_equal(r1["table"].household_id, r0["table"].household_id) and np.array_equal(r1["household"], r0["household"])
            and np.array_equal(r1["block"], r0["block"])):
        raise AssertionError(f"{clock}/{label}: household tables or event rows differ")
    fz = frozen_row(clock, name)
    seed = seed_from(s1.name, clock, "household bootstrap")
    mult = multiplicities(seed, len(r1["table"].household_id))
    d1, d0 = draws_for(r1["table"], mult, s1.residualize_activity), draws_for(r0["table"], mult, s0.residualize_activity)
    lo1, hi1, v1 = pct(d1)
    lo0, hi0, v0 = pct(d0)
    gap = d1 - d0
    log_ = {"clock": clock, "row": label, "spec": name, "tau2_event_study": r1["tau2"], "frozen_tau2": float(fz["value"]), "se_event_study": r1["se_block"],
            "frozen_se": float(fz["se_block"]), "boot_lo_event_study": lo1, "frozen_boot_lo": float(fz["boot_lo"]), "boot_hi_event_study": hi1,
            "frozen_boot_hi": float(fz["boot_hi"]), "n_events": r1["n_events"], "frozen_n_events": int(fz["n_events"]), "n_households": r1["n_households"],
            "frozen_n_households": int(fz["n_households"])}
    for a, b in (("tau2_event_study", "frozen_tau2"), ("se_event_study", "frozen_se"), ("boot_lo_event_study", "frozen_boot_lo"), ("boot_hi_event_study", "frozen_boot_hi")):
        if not np.isclose(log_[a], log_[b], rtol=1e-9, atol=1e-7):
            raise AssertionError(f"{clock}/{label}: {a} {log_[a]!r} does not reproduce {log_[b]!r}")
    if (r1["n_events"], r1["n_households"]) != (log_["frozen_n_events"], log_["frozen_n_households"]):
        raise AssertionError(f"{clock}/{label}: counts differ from results.csv")
    if clock == "T" and (r1["n_events"], r1["n_households"]) != TARGETS_T[label]:
        raise AssertionError(f"T/{label}: counts differ from the reference counts")
    tw = two_way_scalar(r1["scores"] - r0["scores"], r1["household"], r1["block"], r1["k"])
    tw1 = two_way_scalar(r1["scores"], r1["household"], r1["block"], r1["k"])
    tw0 = two_way_scalar(r0["scores"], r0["household"], r0["block"], r0["k"])
    log_["se_event_study_scores"] = tw1["se"]
    if not r1["psd_corrected"] and not np.isclose(tw1["se"], r1["se_block"], rtol=1e-8):
        raise AssertionError(f"{clock}/{label}: score SE {tw1['se']} differs from the original {r1['se_block']}")
    gl, gh, gv = pct(gap)
    both_valid = int(np.sum(np.isfinite(d1) & np.isfinite(d0)))
    cut1, cut0 = ctx.cutoffs_for(s1), ctx.cutoffs_for(s0)
    common = dict(clock=clock, sample=label, horizon={"ff5mom_h10": "10", "ff5mom_h60": "60"}.get(s1.outcome, "20"), weighting=WEIGHT_RULE[s1.weights],
                  controls="log(1+n_A) in both periods" if s1.residualize_activity and s1.activity_control else "none (constant only, both periods)",
                  scale="winsorized" if s1.winsorized else "raw")
    src = f"specification {name} (estimators.specs.analysis_specs) and its factor-only counterpart"
    for bench, r, lo, hi, nv, cut, twb in (("event_study", r1, lo1, hi1, v1, cut1, tw1), ("factor_only", r0, lo0, hi0, v0, cut0, tw0)):
        b = dict(common, benchmark=bench, source=src)
        rows.add("S4_robustness", "covariance", r["tau2"], "bp2", term=label, inference="original estimator (cgm.tau2_i1 or no_control.tau2_no_control_i1)", **b)
        rows.add("S4_robustness", "se_two_way_conditional", r["se_block"], "bp2", term=label, inference="household x 20-session entry block CGM, period-A residual variance fixed (original se_block)", **b)
        rows.add("S4_robustness", "se_two_way_dof", r["dof"], "count", term=label, **b)
        rows.add("S4_robustness", "psd_corrected", float(r["psd_corrected"]), "indicator", term=label, **b)
        rows.add("S4_robustness", "variance_two_way_raw_scores", twb["variance_raw"], "bp4", term=label, inference="scalar score variance before any PSD adjustment", **b)
        rows.add("S4_robustness", "se_two_way_uncorrected_scalar", twb["se"], "bp2", term=label,
                 inference="sqrt of the scalar score variance (no eigenvalue correction); equals se_two_way_conditional unless psd_corrected = 1", **b)
        rows.add("S4_robustness", "slope", r["slope"], "slope", term=label, **b)
        rows.add("S4_robustness", "var_w_x_A", r["var_wx"], "bp2", term=label, **b)
        rows.add("S4_robustness", "boot_ci_low", lo, "bp2", term=label, inference=f"paired household bootstrap, {N_BOOT} draws, seed_from('{s1.name}', '{clock}', 'household bootstrap'); percentile", **b)
        rows.add("S4_robustness", "boot_ci_high", hi, "bp2", term=label, inference="paired household bootstrap percentile", **b)
        rows.add("S4_robustness", "boot_valid_draws", nv, "count", term=label, **b)
        rows.add("S4_robustness", "n_events", r["n_events"], "count", term=label, **b)
        rows.add("S4_robustness", "n_households", r["n_households"], "count", term=label, **b)
        rows.add("S4_robustness", "n_blocks", r["n_blocks"], "count", term=label, **b)
        for p, (lo_c, hi_c) in sorted((cut or {}).items()):
            rows.add("S4_robustness", f"cutoff_low_{p}", lo_c, "bp", term=label, inference="winsorization cutoff (benchmark-specific)", **b)
            rows.add("S4_robustness", f"cutoff_high_{p}", hi_c, "bp", term=label, inference="winsorization cutoff (benchmark-specific)", **b)
    b = dict(common, benchmark="event_study_minus_factor_only", source=src)
    rows.add("S4_robustness", "gap", r1["tau2"] - r0["tau2"], "bp2", term=label, **b)
    rows.add("S4_robustness", "gap_se_two_way_conditional", tw["se"], "bp2", term=label,
             inference="CGM household x block on the difference of the two benchmarks' event-level scores (same events), period-A variances fixed", **b)
    rows.add("S4_robustness", "gap_variance_two_way_raw", tw["variance_raw"], "bp4", term=label, **b)
    rows.add("S4_robustness", "gap_boot_ci_low", gl, "bp2", term=label, inference="paired household bootstrap: gap formed within each common draw", **b)
    rows.add("S4_robustness", "gap_boot_ci_high", gh, "bp2", term=label, **b)
    rows.add("S4_robustness", "gap_boot_valid_draws", gv, "count", term=label, **b)
    rows.add("S4_robustness", "draws_valid_both", both_valid, "count", term=label, **b)
    log(f"{clock}/{label}: ES {r1['tau2']:.3f} [{lo1:.1f}, {hi1:.1f}]  FO {r0['tau2']:.3f} [{lo0:.1f}, {hi0:.1f}]  gap {r1['tau2'] - r0['tau2']:.3f} [{gl:.1f}, {gh:.1f}]; reproduces results.csv")
    out = {"clock": clock, "row": label, "label": dict((a, c) for a, _, c in ROWS)[label], "spec": name, "n_events": r1["n_events"], "n_households": r1["n_households"],
           "n_blocks": r1["n_blocks"], "es_cov": r1["tau2"], "es_se": r1["se_block"], "es_boot_lo": lo1, "es_boot_hi": hi1, "fo_cov": r0["tau2"], "fo_se": r0["se_block"],
           "fo_boot_lo": lo0, "fo_boot_hi": hi0, "gap": r1["tau2"] - r0["tau2"], "gap_se_two_way": tw["se"], "gap_boot_lo": gl, "gap_boot_hi": gh,
           "es_psd": r1["psd_corrected"], "fo_psd": r0["psd_corrected"], "seed": seed, "es_cutoffs_bp": str(cut1), "fo_cutoffs_bp": str(cut0)}
    out["_design"] = {"p1": p1, "table": r1["table"], "hash": design_hash(p1, r1["table"], cut1), "mult": mult, "d1": d1, "d0": d0, "r1": r1, "r0": r0}
    return out, log_


def c14(ctx, results: dict, rows: cm.Rows, log) -> pd.DataFrame:
    """Baseline versus vendor duplicates retained: exact design identity, seeds, and intervals under shared selections."""
    recs = []
    for clock in ("T", "P"):
        a, b = results[(clock, "baseline")]["_design"], results[(clock, "vendor_duplicates")]["_design"]
        same = {k: a["hash"][k] == b["hash"][k] for k in a["hash"]}
        identical = all(same.values())
        seed_a = seed_from("tau/event_share", clock, "household bootstrap")
        seed_b = seed_from("robustness/keep_duplicates", clock, "household bootstrap")
        rec = {"clock": clock, "designs_identical": identical, **{f"identical_{k}": v for k, v in same.items()},
               "baseline_events": len(a["p1"]), "duplicates_events": len(b["p1"]), "baseline_households": len(a["table"].household_id),
               "duplicates_households": len(b["table"].household_id), "seed_baseline": seed_a, "seed_duplicates": seed_b,
               "baseline_ci": pct(a["d1"])[:2], "duplicates_ci_own_seed": pct(b["d1"])[:2]}
        if identical:
            d1 = draws_for(b["table"], a["mult"], True)
            d0 = draws_for(results[(clock, "vendor_duplicates")]["_design"]["r0"]["table"], a["mult"], True)
            rec["duplicates_ci_baseline_selections"] = pct(d1)[:2]
            rec["duplicates_fo_ci_baseline_selections"] = pct(d0)[:2]
            rec["gap_ci_baseline_selections"] = pct(d1 - d0)[:2]
            if rec["duplicates_ci_baseline_selections"] != rec["baseline_ci"]:
                raise AssertionError(f"{clock}: identical designs but intervals differ under shared selections")
            for bench, (lo, hi) in (("event_study", rec["duplicates_ci_baseline_selections"]), ("factor_only", rec["duplicates_fo_ci_baseline_selections"]),
                                    ("event_study_minus_factor_only", rec["gap_ci_baseline_selections"])):
                for stat, v in (("boot_ci_low_baseline_selections", lo), ("boot_ci_high_baseline_selections", hi)):
                    rows.add("C14_duplicates", stat, v, "bp2", clock=clock, scale="winsorized", term="vendor_duplicates", benchmark=bench, sample="vendor_duplicates",
                             inference="identical design: the baseline's 4,999 household selections (seed_from('tau/event_share', clock, 'household bootstrap')) reused",
                             source="robustness/keep_duplicates vs tau/event_share")
        else:
            ea, eb = set(a["p1"]["event_id"]), set(b["p1"]["event_id"])
            rec["events_only_in_duplicates"] = len(eb - ea)
            rec["events_only_in_baseline"] = len(ea - eb)
        rows.add("C14_duplicates", "designs_identical", float(identical), "indicator", clock=clock, scale="winsorized", term="vendor_duplicates",
                 inference="events, outcomes, periods, households, weights, activity and cutoff hashes all equal", source="robustness/keep_duplicates vs tau/event_share")
        rows.add("C14_duplicates", "seed_baseline", seed_a, "seed", clock=clock, scale="winsorized", term="vendor_duplicates", source="outcomes.seeds.seed_from")
        rows.add("C14_duplicates", "seed_duplicates_row", seed_b, "seed", clock=clock, scale="winsorized", term="vendor_duplicates", source="outcomes.seeds.seed_from")
        log(f"C14 {clock}: identical={identical}; {rec}")
        recs.append(rec)
    return pd.DataFrame(recs)


def decompositions(ctx, key: pd.Series, rows: cm.Rows, log) -> pd.DataFrame:
    """Sales and signed trades: K11 = K00 + KDD + K0D + KD0 on benchmark-paired household means, raw and winsorized;
    paired percentile intervals under the row's original selections and delete-one-household jackknife."""
    ss = pd.read_csv(cm.RESTRICTED / "sales_signed_means.csv")
    terms = ["K11", "K00", "KDD", "K0D", "KD0", "difference"]
    out = []
    for (clock, design, scale), g in ss.groupby(["clock", "design", "scale"]):
        g = g.sort_values("household_index").reset_index(drop=True)
        name = "robustness/sales" if design == "sales" else "robustness/signed_all"
        seed = seed_from(name, clock, "household bootstrap")
        M = multiplicities(seed, len(g))
        b = moments(g, M)
        point, deleted, cov, se, lo, hi = jackknife(g)
        G = len(g)
        for j, t in enumerate(terms):
            f = b[:, j][np.isfinite(b[:, j])]
            q = np.quantile(f, [0.025, 0.975])
            rec = {"clock": clock, "design": design, "scale": scale, "term": t, "estimate": point[j], "boot_lo": q[0], "boot_hi": q[1], "boot_valid": len(f),
                   "jackknife_se": se[j], "jackknife_lo": lo[j], "jackknife_hi": hi[j], "households": G, "events": int(g.n_A.sum() + g.n_B.sum()), "seed": seed}
            out.append(rec)
            base = dict(clock=clock, scale=scale, sample=design, term=f"{design}/{t}", benchmark="decomposition", weighting="sqrt(n_A n_B)", controls="log(1+n_A) in both periods",
                        source=f"data/v4/restricted/writing_sample/sales_signed_means.csv ({design}); selections seed_from('{name}', '{clock}', 'household bootstrap')")
            rows.add("S4_sales_signed_decomposition", "estimate", point[j], "bp2", inference="bilinear decomposition on household means", **base)
            rows.add("S4_sales_signed_decomposition", "boot_ci_low", q[0], "bp2", inference=f"paired household bootstrap percentile, {N_BOOT} draws, activity refit per draw", **base)
            rows.add("S4_sales_signed_decomposition", "boot_ci_high", q[1], "bp2", inference="paired household bootstrap percentile", **base)
            rows.add("S4_sales_signed_decomposition", "jackknife_se", se[j], "bp2", inference="delete-one household, both projections refit", **base)
            rows.add("S4_sales_signed_decomposition", "jackknife_ci_low", lo[j], "bp2", inference=f"t({G - 1}) jackknife interval", **base)
            rows.add("S4_sales_signed_decomposition", "jackknife_ci_high", hi[j], "bp2", inference=f"t({G - 1}) jackknife interval", **base)
        rows.add("S4_sales_signed_decomposition", "identity_gap", point[0] - point[1:5].sum(), "bp2", clock=clock, scale=scale, sample=design, term=f"{design}/identity",
                 source="K11 - (K00 + KDD + K0D + KD0)")
        log(f"{clock}/{design}/{scale}: K11 {point[0]:.3f} = K00 {point[1]:.3f} + KDD {point[2]:.3f} + K0D {point[3]:.3f} + KD0 {point[4]:.3f}")
    return pd.DataFrame(out)


def overlap(ctx, results: dict, rows: cm.Rows, log) -> pd.DataFrame:
    """Household overlap of the purchase, sale and signed cohorts, and a common-household sensitivity."""
    recs = []
    for clock in ("T", "P"):
        sets = {k: set(np.asarray(results[(clock, k)]["_design"]["table"].household_id).astype(str)) for k in ("baseline", "sales", "signed")}
        common = sets["baseline"] & sets["sales"]
        rec = {"clock": clock, "purchase_households": len(sets["baseline"]), "sale_households": len(sets["sales"]), "signed_households": len(sets["signed"]),
               "purchase_and_sale": len(common), "purchase_and_signed": len(sets["baseline"] & sets["signed"])}
        for k in ("baseline", "sales"):
            s1 = sp.analysis_specs(clock)[dict((a, b) for a, b, _ in ROWS)[k]]
            for bench, spec in (("event_study", s1), ("factor_only", pn.factor_only(s1))):
                p = ctx.panel(spec)
                p = p[p["household_id"].astype(str).isin(common)]
                t = tau_mod.household_table(p, spec.weights)
                if len(t.household_id) >= 3:
                    v = tau_mod.tau2_household_covariance(t)
                else:
                    v = np.nan
                rec[f"{k}_{bench}_common_households"] = v
                rows.add("S4_overlap", "covariance_common_households", v, "bp2", clock=clock, scale="winsorized", sample=k, benchmark=bench, term=f"{k}/common_purchase_sale",
                         inference=f"same specification restricted to the {len(common)} households in both the purchase and the sale cohorts; own events, cutoffs of the full row",
                         source=f"{s1.name}")
        for k, v in rec.items():
            if k != "clock" and "common_households" not in k:
                rows.add("S4_overlap", k, v, "count", clock=clock, scale="winsorized", term="cohort_overlap", source="household sets of the baseline, sales and signed rows")
        log(f"overlap {clock}: {rec}")
        recs.append(rec)
    return pd.DataFrame(recs)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    log = cm.logger("robustness")
    t0 = time.time()
    ctx, cal, ext, new = pn.load(log)
    del ext
    key = cm.household_key()
    rows = cm.Rows("data/v4/results.csv specifications; data/v4/restricted/writing_sample/fit_cache_extended.parquet")
    results, recon = {}, []
    for clock in ("T", "P"):
        for label, name, _ in ROWS:
            out, rec = run_row(ctx, clock, label, name, rows, log)
            results[(clock, label)] = out
            recon.append(rec)
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "_design"} for r in results.values()])
    cm.write_csv(table, OUT / "table_F5_both_benchmarks_full_precision.csv")
    cm.write_csv(pd.DataFrame(recon), OUT / "event_study_reproduction.csv")
    c = c14(ctx, results, rows, log)
    cm.write_csv(c, OUT / "c14_duplicates.csv")
    d = decompositions(ctx, key, rows, log)
    cm.write_csv(d, OUT / "sales_signed_decomposition.csv")
    o = overlap(ctx, results, rows, log)
    cm.write_csv(o, OUT / "cohort_overlap.csv")
    hashes, draws = [], []
    for (clock, label), r in results.items():
        hashes.append({"clock": clock, "row": label, **r["_design"]["hash"], "seed": r["seed"]})
        dd = r["_design"]
        draws.append(pd.DataFrame({"clock": clock, "row": label, "draw": np.arange(len(dd["d1"])), "K_event_study": dd["d1"], "K_factor_only": dd["d0"],
                                   "gap": dd["d1"] - dd["d0"], "distinct_households": (dd["mult"] > 0).sum(axis=1)}))
    cm.write_csv(pd.DataFrame(hashes), OUT / "specification_hashes.csv")
    (OUT / "paired_bootstrap_draws_F5.csv").unlink(missing_ok=True)
    pd.concat(draws, ignore_index=True).to_csv(OUT / "paired_bootstrap_draws_F5.csv.gz", index=False, float_format="%.17g", lineterminator="\n",
                                               compression={"method": "gzip", "mtime": 0})
    rows.write_component("robustness_both_benchmarks")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
