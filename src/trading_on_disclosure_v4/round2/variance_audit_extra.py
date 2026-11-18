"""Further C6 families: extended-sample covariances, overlap exclusions, both reconstruction rounds and the historical
score regressions. Each original function is called under the recorder; its CGM calls are paired with the SEs it
reports in the order the function makes them, and every pair is checked against the frozen file."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.round2 import common as cm

MATCH_TOL = 1e-8


def add(a, spec_id: str, record: dict, coef: int, reported: float, frozen: float, source: str, **meta) -> None:
    """Register one pre/post record whose reported SE (from the original function) is `reported`."""
    se_rec = record["se_post"][coef]
    scale = reported / se_rec if se_rec > 0 else np.nan
    diag_pre = record["diag_pre"][coef] * scale * scale
    se_pre = record["se_pre"][coef] * scale
    rel_neg = (-record["min_eig_pre"] / record["max_abs_eig_pre"]) if record["min_eig_pre"] < 0 and record["max_abs_eig_pre"] > 0 else 0.0
    fz = float(frozen) if frozen is not None and np.isfinite(frozen) else np.nan
    a.specs.append({"specification_id": spec_id, "record_id": record["record_id"], "coefficient_index": coef, "coefficient_order": "",
                    "n": record["n"], "k": record["k"], "cluster_dimensions": record["dims"],
                    "cluster_counts": json.dumps(record["cluster_counts"], ensure_ascii=False),
                    "correction_factors": json.dumps({d: (g / (g - 1)) * ((record["n"] - 1) / (record["n"] - record["k"])) for d, g in record["cluster_counts"].items() if g > 1}, ensure_ascii=False),
                    "dof": record["dof"], "min_eigenvalue_pre": record["min_eig_pre"], "max_abs_eigenvalue_pre": record["max_abs_eig_pre"],
                    "negative_eigenvalue_relative": rel_neg, "truncation_class": "none" if record["min_eig_pre"] >= 0 else ("numerical" if rel_neg <= 1e-12 else "material"),
                    "psd_corrected": record["psd_corrected"], "correction_rule": "eigenvalue truncation when the smallest eigenvalue < 0 (cgm3)",
                    "contrast_variance_pre": diag_pre, "se_pre_truncation": se_pre if diag_pre >= 0 else np.nan, "se_post_truncation": reported,
                    "contrast_variance_changed": bool(abs(reported - (se_pre if diag_pre >= 0 else 0.0)) > 1e-12 * max(1.0, reported)),
                    "contrast_scale": scale, "frozen_se": fz, "frozen_source": source,
                    "reproduces_frozen_se": bool(np.isfinite(fz) and abs(reported - fz) <= MATCH_TOL * max(1.0, abs(fz))), "notes": "", **meta})


def family_persistence_corrected(a, log) -> None:
    from trading_on_disclosure_v4.final import cumulative as cu

    lg = cm.logger("variance_audit_cumulative")
    u = cu.build_union(lg)
    cache = cu.merged_cache(["permno", "anchor_idx", "resid_sd", *evf.CACHE_Y.values()])
    on_file = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "final" / "persistence_corrected.csv")
    for clock, builder in (("T", cu.transaction_frames), ("P", cu.public_frames)):
        fb, fcx, _ = builder(u, cache, lg)
        start = len(a.rec.records)
        rows, panels = cu.persistence_rows(clock, {"baseline": fb, "corrected": fcx}, evf.pooled_cutoffs, u["calendar"], 0, lg)
        recs = a.rec.records[start:]
        labels = list(panels)
        expected = 2 * len(labels) + 2
        if len(recs) != expected:
            raise AssertionError(f"persistence_rows {clock}: {len(recs)} CGM calls, expected {expected}")
        get = {(r["variant"], r["statistic"]): r for r in rows}
        for i, lab in enumerate(labels):
            for j, (stat, col) in enumerate((("tau2", "se_block"), ("tau2_se_day", "value"))):
                rr = get[(lab, stat)]
                f = on_file[(on_file["clock"] == clock) & (on_file["variant"] == lab) & (on_file["statistic"] == stat)]
                add(a, f"extended/{clock}/persistence/{lab}/{stat}", recs[2 * i + j], 1, float(rr[col]), float(f[col].iloc[0]) if len(f) == 1 else np.nan,
                    f"data/v4/final/persistence_corrected.csv {clock}/{lab}/{stat} {col}", family="extended-sample covariance", clock=clock,
                    clusters="household x 20-session block" if j == 0 else "household x entry day", sample="extended" if lab != "baseline" else "initial (record)")
        for i, lab in enumerate(("baseline", "corrected")):
            rr = get[(lab, "purchase_mean_raw")]
            f = on_file[(on_file["clock"] == clock) & (on_file["variant"] == lab) & (on_file["statistic"] == "purchase_mean_raw")]
            add(a, f"extended/{clock}/persistence/{lab}/purchase_mean_raw", recs[2 * len(labels) + i], 0, float(rr["se_block"]),
                float(f["se_block"].iloc[0]) if len(f) == 1 else np.nan, f"data/v4/final/persistence_corrected.csv {clock}/{lab}/purchase_mean_raw se_block",
                family="extended-sample pooled mean", clock=clock, clusters="household x 20-session block", sample="extended" if lab != "baseline" else "initial (record)")
        log(f"persistence_corrected {clock}: {len(recs)} covariances audited")


def family_overlap(a, ctx, log) -> None:
    from trading_on_disclosure_v4.ext import decompose as dc
    from trading_on_disclosure_v4.ext import overlap as ov

    on_file = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "ext" / "overlap_exclusion.csv")
    for clock in ("T", "P"):
        base = dc.multi_outcome_panel(ctx, clock)
        cuts = {"Y1": evf.pooled_cutoffs(clock, "ff5mom_h20"), "Y0": evf.pooled_cutoffs(clock, "ff5mom_noalpha_h20")}
        flags = ov.overlap_flags(base, ctx.calendar)
        full_hh = pd.Index(sorted(flags["household_id"].unique()))
        for exclusion in ov.EXCLUSIONS:
            keep, hh_after = ov.restricted_panel(flags, exclusion)
            common = full_hh.intersection(hh_after)
            keep_c = keep.loc[keep["household_id"].isin(common)].reset_index(drop=True)
            full_c = flags.loc[flags["household_id"].isin(common)].reset_index(drop=True)
            for label, p in (("full_common", full_c), ("restricted", keep_c)):
                start = len(a.rec.records)
                out = ov.tau_rows(p, clock, ctx.calendar, cuts, label)
                recs = a.rec.records[start:]
                if len(recs) != 2:
                    raise AssertionError(f"overlap tau_rows: {len(recs)} CGM calls")
                for rec_, name in zip(recs, ("headline", "no_alpha"), strict=True):
                    stat = f"tau2_{name}_{label}"
                    f = on_file[(on_file["clock"] == clock) & (on_file["variant"] == exclusion) & (on_file["statistic"] == stat)]
                    fz = float(f["se_block"].iloc[0]) if len(f) == 1 and pd.notna(f["se_block"].iloc[0]) else np.nan
                    add(a, f"overlap/{clock}/{exclusion}/{stat}", rec_, 1, float(out[f"{stat}_se"]), fz, f"data/v4/ext/overlap_exclusion.csv {clock}/{exclusion}/{stat} se_block",
                        family="overlap-exclusion covariance", clock=clock, clusters="household x 20-session block", sample=f"overlap exclusion {exclusion}")
        log(f"overlap {clock} audited")


def family_history(a, ctx, log) -> None:
    from trading_on_disclosure_v4.data import events as evm
    from trading_on_disclosure_v4.data import load
    from trading_on_disclosure_v4.data import vendor_only_status as vs
    from trading_on_disclosure_v4.ext import history_recovery as hr
    from trading_on_disclosure_v4.ext import ledgers as lg
    from trading_on_disclosure_v4.final import history_recovery_final as hrf
    from trading_on_disclosure_v4.final import history_reread as rr
    from trading_on_disclosure_v4.outcomes import panel as pm

    matched = load.load_matched_transactions()
    households = vs.select_vendor_only_households(matched, 2)
    labels = {f"household_{i}": h for i, h in enumerate(households, start=1)}
    snapshot = pd.read_parquet(vs.dropbox_project_root() / vs.VENDOR_SNAPSHOT, columns=["issuer_name", "ticker_clean"])
    dictionary, ambiguous = hr.name_dictionary(matched, snapshot)
    by_sym = hr.symbol_ranges(lg.build_security_ranges())
    timed = load.load_timed_transactions()
    chamber = str(timed.loc[timed["filer_uid"].isin(households), "chamber"].mode().iloc[0])
    calendar = pm.v2_trading_calendar()
    frame_old = ctx.frames["T"]
    ev_old, _ = evm.build_events_T(timed, calendar)
    store: dict = {}

    def loader():
        if "daily" not in store:
            store["daily"], store["factors"] = pm.load_local_daily(), pm.load_factors()
        return store["daily"], store["factors"]

    rounds = (("single_orientation", pd.read_parquet(hr.ROWS), cm.REPO_ROOT / "data" / "v4" / "ext" / "history_recovery_estimates.csv"),
              ("rotation_assisted", hrf.admissible_rows(pd.read_parquet(rr.ROWS_PARQUET), pd.read_parquet(rr.FILINGS_PARQUET)),
               cm.REPO_ROOT / "data" / "v4" / "final" / "history_recovery_final_estimates.csv"))
    for name, rows, frozen_path in rounds:
        on_file = pd.read_csv(frozen_path)
        rec_ = hr.reconcile(hr.resolve(hr.candidates(rows), dictionary, ambiguous, by_sym))
        for route, routes in hr.ROUTES.items():
            tf = hr.timed_frame(rec_.loc[rec_["route"].isin(routes)], labels, chamber)
            union = pd.concat([timed, tf[[c for c in tf.columns if c in timed.columns]]], ignore_index=True)
            ev_new, _ = evm.build_events_T(union, calendar)
            new_ev = ev_new.loc[~ev_new["event_id"].isin(set(ev_old["event_id"]))].copy()
            new_ev, _ = hr.factor_outcomes(new_ev, calendar, loader)
            start = len(a.rec.records)
            t_new, _, _ = hr.expanded_estimate(ctx, calendar, frame_old, new_ev)
            recs = a.rec.records[start:]
            if len(recs) != 1:
                raise AssertionError("expanded_estimate made more than one CGM call")
            key = f"expanded_cohort_{route}"
            if "variant" in on_file and (on_file["variant"] == key).any():
                f = on_file[(on_file["variant"] == key) & (on_file["statistic"] == "tau2_headline")]
            else:
                f = on_file[(on_file["statistic"] == key) & (on_file["variant"] == "tau2_headline")]
            add(a, f"history/{name}/{route}", recs[0], 1, float(t_new["var_wx"] * t_new["se_b"]), float(f["se_block"].iloc[0]) if len(f) == 1 else np.nan,
                f"{frozen_path.relative_to(cm.REPO_ROOT)} {key} se_block", family="reconstruction covariance", clock="T", clusters="household x 20-session block",
                sample=f"provisional reconstruction ({name})")
        log(f"history {name} audited")


def family_forward(a, log) -> None:
    from trading_on_disclosure_v4.ext import forward as fw
    from trading_on_disclosure_v4.outcomes import panel as pm

    calendar = pm.v2_trading_calendar()
    frame = evf.load_event_frame(fw.CLOCK, calendar=calendar, directions=("purchase",), periods=("A", "B", "X"))
    for rule, path in (("historical", "forward_prediction_historical.csv"), ("period", "forward_prediction.csv")):
        on_file = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "ext" / path)
        start = len(a.rec.records)
        res = fw.run(frame, calendar, evf.pooled_cutoffs(fw.CLOCK), n_draws=2, log=None, cutoff_rule=rule)
        recs = a.rec.records[start:]
        targets = [r for r in res["rows"] if pd.notna(r.get("se_block")) and np.isfinite(r["se_block"])]
        used = set()
        for t in targets:
            hits = [r for r in recs if r["record_id"] not in used and abs(r["se_post"][1] - t["se_block"]) <= 1e-12 * max(1.0, t["se_block"])]
            if not hits:
                raise AssertionError(f"forward {rule}: no CGM record for {t['variant']}/{t['statistic']}")
            used.add(hits[0]["record_id"])
            f = on_file[(on_file["variant"] == t["variant"]) & (on_file["statistic"] == t["statistic"])]
            add(a, f"forward/{rule}/{t['variant']}/{t['statistic']}", hits[0], 1, float(t["se_block"]), float(f["se_block"].iloc[0]) if len(f) == 1 else np.nan,
                f"data/v4/ext/{path} {t['variant']}/{t['statistic']} se_block", family="historical-score regressions", clock=fw.CLOCK,
                clusters="household x 20-session block", sample=f"forward prediction ({rule} cutoffs)")
        log(f"forward {rule}: {len(targets)} reported SEs from {len(recs)} CGM calls")


def run(a, ctx, log) -> None:
    family_persistence_corrected(a, log)
    family_overlap(a, ctx, log)
    family_history(a, ctx, log)
    family_forward(a, log)
    run_found(a, ctx, log)


# ---------------------------------------------------------------- the families found by the manuscript enumeration

def value_match(a, records: list[dict], targets: list[dict], family: str, scales=(1.0,)) -> int:
    """Pair each reported SE with the CGM record (any coefficient, candidate scale) that reproduces it."""
    used, n = set(), 0
    for t in targets:
        val = float(t["se"])
        hit = None
        for allow_reuse in (False, True):  # an unused record first; a reused one only for a repeated identical value
            for r in records:
                if r["record_id"] in used and not allow_reuse:
                    continue
                for c in range(r["k"]):
                    for s in scales:
                        if abs(r["se_post"][c] * s - val) <= 1e-9 * max(1.0, val):
                            hit = (r, c, s)
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                break
        if hit is None:
            add_unmatched(a, t, family)
            continue
        r, c, s = hit
        used.add(r["record_id"])
        add(a, t["spec_id"], r, c, r["se_post"][c] * s, t.get("frozen", val), t["source"], family=family, clock=t.get("clock", "P"),
            clusters=t.get("clusters", ""), sample=t.get("sample", ""))
        n += 1
    return n


def add_unmatched(a, t: dict, family: str) -> None:
    a.specs.append({"specification_id": t["spec_id"], "record_id": -1, "family": family, "frozen_se": float(t.get("frozen", t["se"])), "frozen_source": t["source"],
                    "reproduces_frozen_se": False, "notes": "no CGM record reproduces this SE", "psd_corrected": False, "min_eigenvalue_pre": np.nan,
                    "truncation_class": "unmatched", "contrast_variance_changed": False})


def family_sensitivities(a, log) -> None:
    """Missing-outcome, endpoint-completion and delisting-path treatments of the spread (initial and extended)."""
    from trading_on_disclosure_v4.ext import delisting as dl
    from trading_on_disclosure_v4.ext import endpoint as ep
    from trading_on_disclosure_v4.ext import windows as w
    from trading_on_disclosure_v4.final import cumulative as cu
    from trading_on_disclosure_v4.outcomes import panel as pm

    lg = cm.logger("variance_audit_cumulative")
    calendar = pm.v2_trading_calendar()
    # extended sample: missingness_rows
    u = cu.build_union(lg)
    uni = cu.public_universe_corrected(u, lg)
    base_uni = cu.baseline_universe(u)
    market = w.Market(u["calendar"], pm.load_benchmark(), pm.load_factors(), pm.daily_parquet_path())
    market.load(uni["permno"].unique())
    comp, led = cu.completions(uni, market, lg)
    start = len(a.rec.records)
    _, vectors = cu.spread_rows(uni, comp, led, base_uni, u["calendar"], lg)
    a.rec.records = a.rec.records[:start]
    _, sens = cu.missingness_rows(uni, vectors, base_uni)
    recs = a.rec.records[start:]
    fz = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "final" / "missingness_sensitivity_corrected.csv")
    tg = [{"spec_id": f"extended/P/missingness/{s['treatment']}", "se": s["se_block"], "frozen": _sens_frozen(fz, s), "source": f"data/v4/final/missingness_sensitivity_corrected.csv {s['treatment']} se_block",
           "clusters": "household x 20-session block", "sample": "extended"} for s in sens if pd.notna(s["se_block"])]
    n = value_match(a, recs, tg, "extended spread sensitivity")
    log(f"missingness (extended): {n} of {len(tg)} reported SEs matched")
    # initial sample: endpoint completion and delisting paths
    ui, _ = w.public_universe(calendar)
    ui["blk"] = ui["anchor_idx"] // 20
    mk = w.Market(calendar, pm.load_benchmark(), pm.load_factors(), pm.daily_parquet_path())
    mk.load(ui["permno"].unique())
    y = ui["y_market_adjusted"].to_numpy(dtype=float)
    trunc = ui.loc[ui["status_market_adjusted"].to_numpy() == "end_of_sample_truncation"]
    dl_idx = mk.delist.reindex(ui["permno"].to_numpy()).to_numpy(dtype=float)
    has_dl = w.delisting_in_window(ui["anchor_idx"].to_numpy(), dl_idx)
    sessions, rows_j, bench_j = ep.july_slice(set(trunc["permno"].astype(int)))
    cal2, bench2, grids = ep.extended_grid(mk, sessions, rows_j, bench_j)
    cmp_ = ep.complete_windows(trunc, mk, cal2, bench2, grids)
    dlt = ep.delisting_treatments(ui.loc[has_dl & ~np.isfinite(y)], mk)
    start = len(a.rec.records)
    out = []
    base = w.spread_estimate(ui, y)
    out.append((w.T_OBSERVED, base["se"]))
    y6 = y.copy()
    done = cmp_.loc[cmp_["completed"]].set_index("event_id")["y_completed_bp"]
    f6 = ui["event_id"].map(done)
    m6 = ~np.isfinite(y6) & f6.notna().to_numpy()
    y6[m6] = f6.to_numpy(dtype=float)[m6]
    out.append((ep.T_COMPLETED, w.spread_estimate(ui, y6)["se"]))
    for label, col in ((ep.T_CASH, "y_cash_bp"), (ep.T_BENCH, "y_benchmark_reinvested_bp")):
        yv = y6.copy()
        f2 = ui["event_id"].map(dlt.set_index("event_id")[col])
        m = ~np.isfinite(yv) & f2.notna().to_numpy()
        yv[m] = f2.to_numpy(dtype=float)[m]
        out.append((label, w.spread_estimate(ui, yv)["se"]))
    recs = a.rec.records[start:]
    fz = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "ext" / "missing_return_sensitivity_endpoint.csv")
    tg = [{"spec_id": f"initial/P/endpoint/{lab}", "se": se, "frozen": _sens_frozen(fz, {"treatment": lab}), "source": f"data/v4/ext/missing_return_sensitivity_endpoint.csv {lab} se_block",
           "clusters": "household x 20-session block", "sample": "initial"} for lab, se in out]
    n = value_match(a, recs, tg, "initial spread sensitivity")
    log(f"endpoint (initial): {n} of {len(tg)} matched")
    led_p = dl.path_ledger(ui.loc[has_dl & ~np.isfinite(y)], mk, "P")
    cpath = cm.REPO_ROOT / "data" / "v4" / "restricted" / "ext" / "endpoint" / "completed_windows.parquet"
    completed = None
    if cpath.exists():
        c = pd.read_parquet(cpath)
        completed = c.loc[c["completed"]].set_index("event_id")["y_completed_bp"]
    start = len(a.rec.records)
    sens = dl.spread_rows(ui, led_p, completed)
    recs = a.rec.records[start:]
    fz = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "ext" / "missing_return_sensitivity_delisting.csv")
    tg = [{"spec_id": f"initial/P/delisting/{s['treatment']}", "se": s["se_block"], "frozen": _sens_frozen(fz, s), "source": f"data/v4/ext/missing_return_sensitivity_delisting.csv {s['treatment']} se_block",
           "clusters": "household x 20-session block", "sample": "initial"} for s in sens if pd.notna(s["se_block"])]
    n = value_match(a, recs, tg, "initial spread sensitivity")
    log(f"delisting (initial): {n} of {len(tg)} matched")


def _sens_frozen(fz: pd.DataFrame, s: dict) -> float:
    f = fz[fz["treatment"] == s["treatment"]]
    return float(f["se_block"].iloc[0]) if len(f) == 1 else np.nan


def family_factor_delisting(a, log) -> None:
    from trading_on_disclosure_v4.estimators import results as rs
    from trading_on_disclosure_v4.estimators import specs as sp
    from trading_on_disclosure_v4.ext import windows as w
    from trading_on_disclosure_v4.final import cumulative as cu
    from trading_on_disclosure_v4.final import factor_delisting as fd
    from trading_on_disclosure_v4.inference import cgm
    from trading_on_disclosure_v4.outcomes import panel as pm

    calendar = pm.v2_trading_calendar()
    market = w.Market(calendar, pm.load_benchmark(), pm.load_factors(), pm.daily_parquet_path())
    spec = sp.headline_spec("T")
    cuts = evf.pooled_cutoffs("T", spec.outcome)
    on_file = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "final" / "delisting_factor_estimates.csv")
    frames = {"baseline": evf.load_event_frame("T", calendar=calendar), "corrected": pd.read_parquet(cu.RESTRICTED / "frame_T_corrected.parquet")}
    for label, frame in frames.items():
        panel = sp.to_panel(frame, spec, cuts)
        tg = fd.targets(frame, set(panel["household_id"].astype(str)), market, calendar)
        recs_ = []
        for r in tg.itertuples(index=False):
            rec = fd.factor_path(market, int(r.permno), r.anchor_date, int(r.anchor_idx))
            rec.update({"event_id": r.event_id, "household_id": str(r.household_id), "period": r.period, "anchor_date": r.anchor_date})
            recs_.append(rec)
        paths = pd.DataFrame(recs_)
        calls = [("tau2_observed_only", lambda: cgm.tau2_i1(rs.block_cluster(panel, calendar), "ES", anchor="anchor_blk"))]
        for col, tag in (("y_reinvest_factor_bp", "tau2_reinvested_at_fitted_expected_return"), ("y_cash_factor_bp", "tau2_proceeds_in_cash"), ("y_partial_factor_bp", "tau2_partial_window_comparison")):
            calls.append((tag, lambda col=col: fd.completed_tau(frame, panel, paths, col, cuts, calendar)))
        for tag, fn in calls:
            start = len(a.rec.records)
            r = fn()
            rec_ = a.rec.records[start:]
            f = on_file[(on_file["variant"] == label) & (on_file["statistic"] == tag)]
            add(a, f"extended/T/delisting_factor/{label}/{tag}", rec_[0], 1, float(r["var_wx"] * r["se_b"]), float(f["se_block"].iloc[0]) if len(f) == 1 else np.nan,
                f"data/v4/final/delisting_factor_estimates.csv {label}/{tag} se_block", family="delisting-factor covariance", clock="T",
                clusters="household x 20-session block", sample="extended" if label == "corrected" else "initial (record)")
    log("factor delisting audited")


def family_units(a, log) -> None:
    from trading_on_disclosure_v4.existing import rerun
    from trading_on_disclosure_v4.existing import units

    signals_raw = pd.read_csv(rerun.SIGNALS_PATH, low_memory=False)
    signals_raw[units.FOLLOWER_ENTRY_COL] = pd.to_datetime(signals_raw[units.FOLLOWER_ENTRY_COL])
    signals = rerun.load_signals()
    dup = rerun.duplicate_signal_uids()
    dedup, _ = rerun.dedup_filter(signals, dup)
    calendar = rerun.load_calendar()
    on_file = pd.read_csv(rerun.TABLES_PATH)
    for sample, frame in (("original", signals), ("dedup", dedup)):
        for test in units.CLUSTERED_TESTS:
            for clusters in rerun.CLUSTERS:
                start = len(a.rec.records)
                r = units.clustered_row(frame, test, clusters, calendar)
                rec_ = a.rec.records[start:]
                f = on_file[(on_file["test_id"] == test) & (on_file["sample"] == sample) & (on_file["clusters"] == clusters)]
                add(a, f"units/{test}/{sample}/{clusters}", rec_[0], rec_[0]["k"] - 1, float(r["se"]), float(f["se"].iloc[0]) if len(f) == 1 else np.nan,
                    f"data/v4/existing/rerun_tables.csv {test}/{sample}/{clusters} se", family="initial spread and mean tests (units)", clusters=clusters, sample=sample)
    log("unit tests audited")


def family_pyun(a, log) -> None:
    from trading_on_disclosure_v4.pyun import bridge
    from trading_on_disclosure_v4.pyun.estimate import Inputs

    start = len(a.rec.records)
    table = bridge.run_bridge(Inputs())
    recs = a.rec.records[start:]
    on_file = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "pyun" / "bridge_table.csv")
    tg = []
    for r in table.itertuples(index=False):
        se = getattr(r, "se_headline", np.nan)
        if pd.notna(se) and r.quantity:
            f = on_file[(on_file["step"].astype(str) == str(r.step)) & (on_file["quantity"] == r.quantity)]
            tg.append({"spec_id": f"pyun/bridge/{r.step}/{r.quantity}", "se": float(se), "frozen": float(f["se_headline"].iloc[0]) if len(f) == 1 else np.nan,
                       "source": f"data/v4/pyun/bridge_table.csv step {r.step} {r.quantity} se_headline", "clusters": "household x filing month or 20-session block", "sample": "Pyun bridge"})
    n = value_match(a, recs, tg, "Pyun bridge", scales=(1.0, 1e4, 100.0))
    log(f"Pyun bridge: {n} of {len(tg)} reported SEs matched from {len(recs)} CGM calls")


def family_lag(a, ctx, log) -> None:
    """The 180- and 90-day rows of the new filing-lag sensitivity (the 400-day rows are the reconstruction family)."""
    from trading_on_disclosure_v4.data import events as evm
    from trading_on_disclosure_v4.data import load
    from trading_on_disclosure_v4.data import vendor_only_status as vs
    from trading_on_disclosure_v4.ext import history_recovery as hr
    from trading_on_disclosure_v4.ext import ledgers as lg
    from trading_on_disclosure_v4.final import history_recovery_final as hrf
    from trading_on_disclosure_v4.final import history_reread as rr
    from trading_on_disclosure_v4.outcomes import panel as pm

    on_file = pd.read_csv(cm.RUN_DIR / "lag_sensitivity" / "ocr_lag_sensitivity.csv")
    rows = hrf.admissible_rows(pd.read_parquet(rr.ROWS_PARQUET), pd.read_parquet(rr.FILINGS_PARQUET))
    matched = load.load_matched_transactions()
    households = vs.select_vendor_only_households(matched, 2)
    labels = {f"household_{i}": h for i, h in enumerate(households, start=1)}
    snapshot = pd.read_parquet(vs.dropbox_project_root() / vs.VENDOR_SNAPSHOT, columns=["issuer_name", "ticker_clean"])
    dictionary, ambiguous = hr.name_dictionary(matched, snapshot)
    by_sym = hr.symbol_ranges(lg.build_security_ranges())
    timed = load.load_timed_transactions()
    chamber = str(timed.loc[timed["filer_uid"].isin(households), "chamber"].mode().iloc[0])
    calendar = pm.v2_trading_calendar()
    ev_old, _ = evm.build_events_T(timed, calendar)
    store: dict = {}

    def loader():
        if "daily" not in store:
            store["daily"], store["factors"] = pm.load_local_daily(), pm.load_factors()
        return store["daily"], store["factors"]

    saved = hr.MAX_LAG_DAYS
    try:
        for lag in (180, 90):
            hr.MAX_LAG_DAYS = lag
            rec_ = hr.reconcile(hr.resolve(hr.candidates(rows), dictionary, ambiguous, by_sym))
            for route, routes in hr.ROUTES.items():
                tf = hr.timed_frame(rec_.loc[rec_["route"].isin(routes)], labels, chamber)
                union = pd.concat([timed, tf[[c for c in tf.columns if c in timed.columns]]], ignore_index=True)
                ev_new, _ = evm.build_events_T(union, calendar)
                new_ev = ev_new.loc[~ev_new["event_id"].isin(set(ev_old["event_id"]))].copy()
                new_ev, _ = hr.factor_outcomes(new_ev, calendar, loader)
                start = len(a.rec.records)
                t_new, _, _ = hr.expanded_estimate(ctx, calendar, ctx.frames["T"], new_ev)
                recs = a.rec.records[start:]
                f = on_file[(on_file["max_lag_days"] == lag) & (on_file["route"] == route)]
                add(a, f"history/lag_{lag}/{route}", recs[0], 1, float(t_new["var_wx"] * t_new["se_b"]), float(f["se_block_bp2"].iloc[0]),
                    f"data/v4/round2/closeout/closeout_2026-11-10/lag_sensitivity/ocr_lag_sensitivity.csv {lag}/{route} se_block_bp2",
                    family="reconstruction covariance", clock="T", clusters="household x 20-session block", sample=f"provisional reconstruction, {lag}-day lag")
    finally:
        hr.MAX_LAG_DAYS = saved
    log("lag rows audited")


def run_found(a, ctx, log) -> None:
    family_sensitivities(a, log)
    family_factor_delisting(a, log)
    family_units(a, log)
    family_pyun(a, log)
    family_lag(a, ctx, log)
