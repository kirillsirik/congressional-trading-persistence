"""Task C6 and Export 6: every two- and three-way CGM covariance behind a reported standard error, recomputed with the
original estimators, before and after eigenvalue truncation, joined to its frozen source row.

python -m trading_on_disclosure_v4.round2.variance_audit
Writes <run>/variance_audit/{variance_matrices_before_truncation.csv, variance_matrices_after_truncation.csv,
variance_specifications.csv, variance_audit_summary.csv}.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import no_control as nc
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.estimators import tests_t1_t4 as tt
from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.ext import inference as einf
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.round2 import cgm_recorder as cr
from trading_on_disclosure_v4.round2 import common as cm

OUT = cm.RUN_DIR / "variance_audit"
TINY = 1e-12  # |negative eigenvalue| / max |eigenvalue| at or below this: numerical noise, above: material
MATCH_TOL = 1e-8


class Audit:
    """Collects one specification per estimator call: the CGM records made during the call and the frozen SE it reproduces."""

    def __init__(self, rec: cr.Recorder):
        self.rec = rec
        self.specs: list[dict] = []

    def run(self, spec_id: str, fn, *, coef: int, scale_from=None, frozen=None, source: str = "", coef_names=(), notes: str = "", **meta):
        start = len(self.rec.records)
        out = fn()
        new = self.rec.records[start:]
        if len(new) != 1:
            raise AssertionError(f"{spec_id}: expected one CGM call, saw {len(new)}")
        r = new[0]
        scale = float(scale_from(out)) if scale_from else 1.0
        se_post = r["se_post"][coef] * scale
        se_pre = r["se_pre"][coef] * scale
        diag_pre = r["diag_pre"][coef] * scale * scale
        fz = float(frozen) if frozen is not None and np.isfinite(frozen) else np.nan
        match = bool(np.isfinite(fz) and abs(se_post - fz) <= MATCH_TOL * max(1.0, abs(fz)))
        rel_neg = (-r["min_eig_pre"] / r["max_abs_eig_pre"]) if r["min_eig_pre"] < 0 and r["max_abs_eig_pre"] > 0 else 0.0
        self.specs.append({"specification_id": spec_id, "record_id": r["record_id"], "coefficient_index": coef,
                           "coefficient_order": ",".join(coef_names) if coef_names else "", "n": r["n"], "k": r["k"],
                           "cluster_dimensions": r["dims"], "cluster_counts": json.dumps(r["cluster_counts"], ensure_ascii=False),
                           "correction_factors": json.dumps({d: (g / (g - 1)) * ((r["n"] - 1) / (r["n"] - r["k"])) for d, g in r["cluster_counts"].items() if g > 1}, ensure_ascii=False),
                           "dof": r["dof"], "min_eigenvalue_pre": r["min_eig_pre"], "max_abs_eigenvalue_pre": r["max_abs_eig_pre"],
                           "negative_eigenvalue_relative": rel_neg,
                           "truncation_class": "none" if r["min_eig_pre"] >= 0 else ("numerical" if rel_neg <= TINY else "material"),
                           "psd_corrected": r["psd_corrected"], "correction_rule": "eigenvalue truncation when the smallest eigenvalue < 0 (cgm3)",
                           "contrast_variance_pre": diag_pre, "se_pre_truncation": se_pre if diag_pre >= 0 else np.nan,
                           "se_post_truncation": se_post, "contrast_variance_changed": bool(abs(se_post - (se_pre if diag_pre >= 0 else 0.0)) > 1e-12 * max(1.0, se_post)),
                           "contrast_scale": scale, "frozen_se": fz, "frozen_source": source, "reproduces_frozen_se": match, "notes": notes, **meta})
        return out


def _results(clock, analysis, variant, statistic, column):
    try:
        return cm.frozen_value(clock, analysis, variant, statistic, column)
    except KeyError:
        return np.nan


def family_results(a: Audit, ctx) -> None:
    """results.csv: every covariance specification (block and day clusters) and the pooled purchase means."""
    for clock in ("T", "P"):
        for name, spec in sp.analysis_specs(clock).items():
            if spec.kind != "tau":
                continue
            analysis, variant = name.split("/", 1)
            panel = ctx.panel(spec)
            if panel.empty:
                continue
            pb = rs.block_cluster(panel, ctx.calendar)
            names = ("intercept", "x_A", "log_activity") if spec.activity_control else ("intercept", "x_A")
            for anchor, lab, col, stat in (("anchor_blk", "household x 20-session block", "se_block", "tau2"), ("anchor_date", "household x entry day", "value", "tau2_se_day")):
                if spec.residualize_activity:
                    fn = lambda pb=pb, spec=spec, anchor=anchor: cgm.tau2_i1(pb, spec.weights, anchor=anchor, activity_control=spec.activity_control)
                else:
                    fn = lambda pb=pb, spec=spec, anchor=anchor: nc.tau2_no_control_i1(pb, spec.weights, anchor=anchor)
                a.run(f"results/{clock}/{name}/{stat}", fn, coef=1, scale_from=lambda o: o["var_wx"], coef_names=names,
                      frozen=_results(clock, analysis, variant, stat, col), source=f"data/v4/results.csv {clock}/{analysis}/{variant}/{stat} {col}",
                      family="persistence covariance", clock=clock, clusters=lab, sample="initial")
        for name in ("purchase_mean", "purchase_mean/official_only", "purchase_mean/excl_two_vendor_households"):
            spec = sp.analysis_specs(clock)[name]
            pb = rs.block_cluster(ctx.panel(spec), ctx.calendar)
            prefix = "" if name == "purchase_mean" else name.split("/", 1)[1] + "_"
            for anchor, lab, stat, col in (("anchor_blk", "household x 20-session block", "mean", "se_block"), ("anchor_date", "household x entry day", "se_day", "value")):
                a.run(f"results/{clock}/{name}/{stat}", lambda pb=pb, anchor=anchor: tt.t1_clustered_mean(pb, anchor=anchor), coef=0, coef_names=("mean",),
                      frozen=_results(clock, "purchase_mean", prefix + "vs_zero", stat, col), source=f"data/v4/results.csv {clock}/purchase_mean/{prefix}vs_zero/{stat} {col}",
                      family="pooled purchase mean", clock=clock, clusters=lab, sample="initial")


def family_rerun(a: Audit) -> None:
    """existing/rerun_tables.csv: every spread / mean test on the original, deduplicated and the two extra samples."""
    calendar = rerun.load_calendar()
    signals = rerun.load_signals()
    dedup, _ = rerun.dedup_filter(signals, rerun.duplicate_signal_uids())
    frames = {"original": signals, "dedup": dedup}
    extra = rerun.extra_frames(dedup, rerun.two_vendor_household_ids())
    on_file = pd.read_csv(rerun.TABLES_PATH)
    for test in rerun.TESTS:
        fr = dict(frames)
        if test.test_id == rerun.PRIMARY_TEST_ID:
            fr.update(extra)
        for sample, frame in fr.items():
            for clusters in rerun.CLUSTERS:
                f = on_file[(on_file["test_id"] == test.test_id) & (on_file["sample"] == sample) & (on_file["clusters"] == clusters)]
                frozen = float(f["se"].iloc[0]) if len(f) == 1 else np.nan
                start = len(a.rec.records)
                r = rerun.run_test(frame, test, clusters, calendar)
                new = a.rec.records[start:]
                if len(new) != 1:
                    continue  # tests without a CGM covariance (none expected)
                a.rec.records = a.rec.records[:start]
                a.run(f"rerun/{test.test_id}/{sample}/{clusters}", lambda frame=frame, test=test, clusters=clusters: rerun.run_test(frame, test, clusters, calendar),
                      coef=new[0]["k"] - 1, scale_from=lambda o: rerun.BP, frozen=frozen, source=f"data/v4/existing/rerun_tables.csv {test.test_id}/{sample}/{clusters} se",
                      coef_names=("intercept", "purchase") if new[0]["k"] == 2 else ("mean",), family="initial spread and mean tests", clusters=clusters, sample=sample)


def family_ladder(a: Audit, ctx) -> None:
    """ext/inference_ladder.csv: the spread, the two purchase means and the covariance under every clustering, and the
    block-demeaned versions (the actual-data part of each function, called exactly as the module does)."""
    on_file = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "ext" / "inference_ladder.csv")
    calendar = ctx.calendar

    def frozen(clock, variant, statistic):
        f = on_file[(on_file["clock"] == clock) & (on_file["variant"] == variant) & (on_file["statistic"] == statistic)]
        return float(f["se_block"].iloc[0]) if len(f) == 1 else np.nan

    signals = rerun.load_signals()
    dedup, _ = rerun.dedup_filter(signals, rerun.duplicate_signal_uids())
    d = dedup.dropna(subset=["ar_E_h20", "follower_entry_date"]).reset_index(drop=True)
    y = d["ar_E_h20"].to_numpy(float) * einf.BP
    X = np.column_stack([np.ones(len(y)), (d["direction"] == "purchase").to_numpy(float)])
    blk = einf.date_clusters(d["follower_entry_date"], calendar)["block20"]
    y_dm = y - pd.Series(y).groupby(blk).transform("mean").to_numpy()
    for est, yy, keep in (("public_spread_dedup", y, None), ("public_spread_dedup_block_demeaned", y_dm, ("household_x_block20", "household_x_block20_x_security"))):
        for proc, clusters in _ladder_partitions(d["filer_uid"].to_numpy(), d["follower_entry_date"], d["permno"].to_numpy(), calendar):
            if keep and proc not in keep:
                continue
            a.run(f"ladder/P/{est}/{proc}", lambda yy=yy, clusters=clusters: einf.clustered(yy, X, None, clusters, 1), coef=1, coef_names=("intercept", "purchase"),
                  frozen=frozen("P", est, proc), source=f"data/v4/ext/inference_ladder.csv P/{est}/{proc} se_block", family="spread clustering ladder", clusters=proc, sample="initial")
    for clock in ("T", "P"):
        specs = sp.analysis_specs(clock)
        for est, spec in (("purchase_mean_all_households", specs["purchase_mean"]),
                          ("purchase_mean_persistence_sample", replace(sp.headline_spec(clock), name="purchase_mean_persistence", kind="mean", winsorized=False))):
            panel = ctx.panel(spec)
            yv = panel["y_h20"].to_numpy(float)
            for proc, clusters in _ladder_partitions(panel["household_id"].to_numpy(), panel["anchor_date"], panel["permno"].to_numpy(), calendar):
                a.run(f"ladder/{clock}/{est}/{proc}", lambda yv=yv, clusters=clusters: einf.clustered(yv, np.ones((len(yv), 1)), None, clusters, 0), coef=0, coef_names=("mean",),
                      frozen=frozen(clock, est, proc), source=f"data/v4/ext/inference_ladder.csv {clock}/{est}/{proc} se_block", family="mean clustering ladder", clusters=proc, sample="initial")
        panel = ctx.panel(sp.headline_spec(clock))
        table = tau_mod.household_table(panel, "ES")
        hh = pd.DataFrame({"household_id": table.household_id, "x": table.x, "log_n_a": table.log_n_a, "event_weight": table.w / table.n_b})
        b = panel.loc[panel["period"] == "B", ["household_id", "y_h20", "anchor_date", "permno"]].merge(hh, on="household_id", how="inner")
        Xb = np.column_stack([np.ones(len(b)), b["x"].to_numpy(float), b["log_n_a"].to_numpy(float)])
        v = tau_mod.var_w(table.x, table.w)
        for proc, clusters in _ladder_partitions(b["household_id"].to_numpy(), b["anchor_date"], b["permno"].to_numpy(), calendar):
            a.run(f"ladder/{clock}/tau2_headline/{proc}", lambda clusters=clusters: einf.clustered(b["y_h20"].to_numpy(float), Xb, b["event_weight"].to_numpy(float), clusters, 1),
                  coef=1, scale_from=lambda o, v=v: v, coef_names=("intercept", "x_A", "log_activity"), frozen=frozen(clock, "tau2_headline", proc),
                  source=f"data/v4/ext/inference_ladder.csv {clock}/tau2_headline/{proc} se_block", family="covariance clustering ladder", clusters=proc, sample="initial")
        # block-demeaned covariance (actual-data part of ext.inference.demeaned_tau2_rows)
        frame = ctx.frames[clock]
        base = sp.Spec(f"{einf.ANALYSIS}/demeaned", clock, "tau", winsorized=False)
        raw = sp.to_panel(frame, base, None)
        allp = frame.loc[(frame["direction"] == "purchase") & frame["period"].isin(["A", "B"]) & ~frame["dup_all_lines"].astype(bool)]
        allp = allp.loc[np.isfinite(pd.to_numeric(allp["y_h20"], errors="coerce"))]
        bm = pd.Series(pd.to_numeric(allp["y_h20"]).to_numpy(float)).groupby(einf.date_clusters(allp["anchor_date"], calendar)["block20"]).mean()
        raw = raw.assign(y_h20=raw["y_h20"].to_numpy(float) - bm.reindex(einf.date_clusters(raw["anchor_date"], calendar)["block20"]).to_numpy())
        pdm = sp.winsorize(raw, evf.sample_cutoffs(raw["y_h20"].to_numpy(), raw["period"].to_numpy()))
        pbd = rs.block_cluster(pdm, calendar)
        a.run(f"ladder/{clock}/tau2_headline_block_demeaned/household_x_block20", lambda: cgm.tau2_i1(pbd, "ES", anchor="anchor_blk"), coef=1,
              scale_from=lambda o: o["var_wx"], coef_names=("intercept", "x_A", "log_activity"),
              frozen=frozen(clock, "tau2_headline_block_demeaned", "household_x_block20"),
              source=f"data/v4/ext/inference_ladder.csv {clock}/tau2_headline_block_demeaned/household_x_block20 se_block",
              family="covariance clustering ladder", clusters="household_x_block20", sample="initial")


def _ladder_partitions(household, dates, permno, calendar):
    dc = einf.date_clusters(dates, calendar)
    out = [(f"household_x_{name}", pd.DataFrame({"household": household, name: col})) for name, col in dc.items()]
    out.append(("household_x_block20_x_security", pd.DataFrame({"household": household, "block20": dc["block20"], "security": permno})))
    out.append(("household_x_block60_x_security", pd.DataFrame({"household": household, "block60": dc["block60"], "security": permno})))
    return out


def family_extended(a: Audit, log) -> None:
    """final/spread_corrected.csv: the extended-sample spreads under every completion treatment and the ladder of the
    primary corrected spread; final/persistence_corrected.csv: the covariance rows on the extended sample."""
    from trading_on_disclosure_v4.ext import windows as w
    from trading_on_disclosure_v4.final import cumulative as cu
    from trading_on_disclosure_v4.outcomes import panel as pm

    lg = cm.logger("variance_audit_cumulative")
    u = cu.build_union(lg)
    uni = cu.public_universe_corrected(u, lg)
    base_uni = cu.baseline_universe(u)
    market = w.Market(u["calendar"], pm.load_benchmark(), pm.load_factors(), pm.daily_parquet_path())
    market.load(uni["permno"].unique())
    comp, led = cu.completions(uni, market, lg)
    start = len(a.rec.records)
    _, vectors = cu.spread_rows(uni, comp, led, base_uni, u["calendar"], lg)
    a.rec.records = a.rec.records[:start]
    on_file = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "final" / "spread_corrected.csv")
    f0 = on_file[(on_file["variant"] == "baseline_record") & (on_file["statistic"] == "spread")]
    a.run("extended/P/spread/baseline_record", lambda: w.spread_estimate(base_uni, base_uni["y_market_adjusted"].to_numpy(dtype=float)), coef=1,
          coef_names=("intercept", "purchase"), frozen=float(f0["se_block"].iloc[0]), source="data/v4/final/spread_corrected.csv baseline_record/spread se_block",
          family="extended spread", clusters="household x 20-session block", sample="initial (record)")
    for label in cu.SPREAD_VARIANTS[1:]:
        f = on_file[(on_file["variant"] == label) & (on_file["statistic"] == "spread")]
        a.run(f"extended/P/spread/{label}", lambda label=label: w.spread_estimate(uni, vectors[label]), coef=1, coef_names=("intercept", "purchase"),
              frozen=float(f["se_block"].iloc[0]) if len(f) == 1 else np.nan, source=f"data/v4/final/spread_corrected.csv {label}/spread se_block",
              family="extended spread", clusters="household x 20-session block", sample="extended")
    y = vectors[cu.PRIMARY_SPREAD]
    ok = np.isfinite(y)
    fr = uni.loc[ok]
    X = np.column_stack([np.ones(int(ok.sum())), (fr["direction"].to_numpy() == "purchase").astype(float)])
    lad = on_file[on_file["analysis"] == "spread_ladder"]
    for proc, clusters in _ladder_partitions(fr["household"].to_numpy(), fr["anchor_date"], fr["permno"].to_numpy(), u["calendar"]):
        f = lad[lad["statistic"] == proc]
        a.run(f"extended/P/spread_ladder/{proc}", lambda clusters=clusters: einf.clustered(y[ok], X, None, clusters, 1), coef=1, coef_names=("intercept", "purchase"),
              frozen=float(f["se_block"].iloc[0]) if len(f) == 1 else np.nan, source=f"data/v4/final/spread_corrected.csv spread_ladder/{proc} se_block",
              family="extended spread ladder", clusters=proc, sample="extended")
    log(f"extended-sample spreads audited ({len(uni)} signals)")


def summarize(specs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, g in specs.groupby("family", sort=False):
        rows.append({"family": fam, "n_specifications": int(len(g)), "n_reproducing_frozen_se": int(g["reproduces_frozen_se"].sum()),
                     "n_with_frozen_comparator": int(g["frozen_se"].notna().sum()), "n_negative_min_eigenvalue": int((g["min_eigenvalue_pre"] < 0).sum()),
                     "n_truncated": int(g["psd_corrected"].sum()), "n_material": int((g["truncation_class"] == "material").sum()),
                     "n_numerical": int((g["truncation_class"] == "numerical").sum()), "n_reported_se_changed": int(g["contrast_variance_changed"].sum())})
    tot = {"family": "all", **{k: int(sum(r[k] for r in rows)) for k in rows[0] if k != "family"}}
    return pd.DataFrame(rows + [tot])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Task C6: variance-truncation audit")
    ap.add_argument("--families", default="results,rerun,ladder,extended")
    args = ap.parse_args(argv)
    log = cm.logger("variance_audit")
    t0 = time.time()
    fams = set(args.families.split(","))
    ctx, _ = cm.load_context()
    with cr.recording() as rec:
        a = Audit(rec)
        if "results" in fams:
            family_results(a, ctx)
            log(f"results family: {len(a.specs)} specifications")
        if "rerun" in fams:
            family_rerun(a)
            log(f"through rerun family: {len(a.specs)}")
        if "ladder" in fams:
            family_ladder(a, ctx)
            log(f"through ladder family: {len(a.specs)}")
        if "extended" in fams:
            family_extended(a, log)
            log(f"through extended family: {len(a.specs)}")
        try:
            from trading_on_disclosure_v4.round2 import variance_audit_extra as vx
            vx.run(a, ctx, log)
        except ImportError:
            pass
        records = {r["record_id"]: r for r in rec.records}
    specs = pd.DataFrame(a.specs)
    pre, post = [], []
    for s in specs.itertuples(index=False):
        if s.record_id < 0:
            continue
        r = records[s.record_id]
        for i, row in enumerate(r["cov_pre"]):
            for j, v in enumerate(row):
                pre.append({"specification_id": s.specification_id, "coefficient_i": i, "coefficient_j": j, "value": v})
                post.append({"specification_id": s.specification_id, "coefficient_i": i, "coefficient_j": j, "value": r["cov_post"][i][j]})
    summ = summarize(specs)
    cm.write_csv(pd.DataFrame(pre), OUT / "variance_matrices_before_truncation.csv")
    cm.write_csv(pd.DataFrame(post), OUT / "variance_matrices_after_truncation.csv")
    cm.write_csv(specs, OUT / "variance_specifications.csv")
    cm.write_csv(summ, OUT / "variance_audit_summary.csv")
    eig = [{"specification_id": s.specification_id, "eigenvalues_pre": json.dumps(records[s.record_id]["eig_pre"])} for s in specs.itertuples(index=False) if s.record_id >= 0]
    cm.write_csv(pd.DataFrame(eig), OUT / "eigenvalues_before_truncation.csv")
    log(f"{len(specs)} specifications; {int(specs['psd_corrected'].sum())} truncated; {int(specs['reproduces_frozen_se'].sum())} reproduce the frozen SE; "
        f"{time.time() - t0:.0f} s")
    print(summ.to_string(index=False))
    bad = specs[specs["frozen_se"].notna() & ~specs["reproduces_frozen_se"]]
    if len(bad):
        print(bad[["specification_id", "se_post_truncation", "frozen_se"]].to_string(index=False))


if __name__ == "__main__":
    main()
