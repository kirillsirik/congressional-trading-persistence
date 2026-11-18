"""Shifted-date reference datasets for the corrected event sets and the reference probabilities of the corrected
persistence estimates (closeout plan section 4).

The generator of the plan (null.block_shift: common signed shifts of 40-250 sessions per 20-session anchor block,
infeasible windows dropped with the 10% retry rule, cap 50) is run on the corrected purchase events of a clock
with the released forward cache joined to its extension; 2,000 draws, timing-only (N1) and centered (N2)
flavours, seeds derived from the new set name; drop and retry logs, cap hits and overlap flags written with the
set. Every statistic of the corrected panels is then recomputed per draw with the same panel transformation
(eligibility reassessed per draw, winsorization at the committed cutoffs) and the add-one upper-tail rank of the
observed level is reported. Restricted: data/v4/restricted/final/panels/<clock>/; committed:
data/v4/final/persistence_corrected_reference.csv and reference_sets.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.final import cache_extension as cx
from trading_on_disclosure_v4.final import common as fc
from trading_on_disclosure_v4.final import cumulative as cu
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.inference.placebo import placebo_p_value
from trading_on_disclosure_v4.null import block_shift as bs
from trading_on_disclosure_v4.null import run_null_panels as rnp
from trading_on_disclosure_v4.outcomes import forward_cache as fcache
from trading_on_disclosure_v4.outcomes import panel as pm

PANELS_ROOT = fc.RESTRICTED / "panels"
OUT_REF = fc.OUT_DIR / "persistence_corrected_reference.csv"
OUT_SETS = fc.OUT_DIR / "reference_sets.csv"
N_DRAWS = 2000
BATCH = 100
SET_TAG = "final"
SPECS = {"headline": lambda c: sp.headline_spec(c), "equal_household": lambda c: sp.headline_spec(c, "EH"), "pair": lambda c: sp.headline_spec(c, "PW"),
         "no_alpha": lambda c: sp.Spec("robustness/no_alpha", c, "tau", outcome="ff5mom_noalpha_h20"),
         "market_adjusted": lambda c: sp.Spec("robustness/market_adjusted", c, "tau", outcome="mkt_h20"),
         "raw_outcome": lambda c: sp.Spec("robustness/raw_outcome", c, "tau", winsorized=False)}


def set_name(clock: str) -> str:
    return bs.test_name_for(clock, "purchase", "A,B", 20, "both", True, False, False, "all_lines", SET_TAG)


def corrected_events(clock: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(purchase events of periods A and B for the generator, the corrected frame) of a clock from the cumulative run."""
    frame = pd.read_parquet(cu.RESTRICTED / f"frame_{clock}_corrected.parquet")
    ev = frame.loc[(frame["direction"] == "purchase") & frame["period"].isin(["A", "B"]) & ~frame["dup_all_lines"].astype(bool),
                   ["event_id", "household_id", "permno", "clock", "anchor_date", "period"]].copy()
    ev["anchor_date"] = pd.to_datetime(ev["anchor_date"])
    return ev.reset_index(drop=True), frame


def cache_grid(cfg: bs.ShiftConfig) -> tuple[bs.CacheGrid, str]:
    cols = ["permno", "anchor_idx", "anchor_date", "reason_h20"] + [bs.Y_COLUMNS[o] for o in cfg.outcomes] + [bs.CARRY_COLUMNS[c] for c in cfg.carry]
    cache = cu.merged_cache(cols)
    sha = fc.config_key({"released": json.loads(fcache.MANIFEST_PATH.read_text())["sha256"]}, [fc.file_entry(cx.OUT_CACHE)])
    return bs.CacheGrid(cache, outcomes=cfg.outcomes, carry=cfg.carry), sha


def generate(clock: str, ev: pd.DataFrame, log) -> tuple[Path, Path, dict]:
    name = set_name(clock)
    cfg = bs.ShiftConfig(test_name=name, outcomes=rnp.FULL_OUTCOMES, carry=("resid_sd",))
    grid, cache_sha = cache_grid(cfg)
    evb = bs.assign_blocks(ev, pm.v2_trading_calendar(), cfg)
    n1 = rnp.set_dir(clock, "N1", name, PANELS_ROOT)
    n2 = rnp.set_dir(clock, "N2", name, PANELS_ROOT)
    log_path = fc.LOG_DIR / f"reference_panels_{clock}.log"
    gen = rnp.generate_n1(evb, grid, cfg, N_DRAWS, n1, BATCH, cache_sha, log_path)
    summary, means = rnp.finalize_n1(n1, cfg, evb)
    summary["seconds"] = round(gen["seconds"], 2)
    extra = {"flavour": "N1", "clock": clock, "batch_size": BATCH, "cache_sha256": cache_sha, "n_households_in": int(evb["household_id"].nunique()),
             "events_source": str((cu.RESTRICTED / f"frame_{clock}_corrected.parquet").relative_to(fc.REPO_ROOT))}
    rnp.write_manifest(n1, summary, extra)
    w_ = rnp.write_n2(n1, n2, means, cfg, log_path)
    check = rnp.n2_centering_check(n2, cfg)
    manifest = rnp.write_manifest(n2, summary, {**extra, "flavour": "N2", "source_n1_dir": str(n1), "centering_sha256": w_["centering_sha256"],
                                                "n2_centering_check_max_abs": check, "seconds_n2": round(w_["seconds"], 2)})
    cmax = max(v for k, v in check.items() if k != "n_household_period_cells")
    log(f"{name}: {manifest['n_draws']} draws, {manifest['n_events_in']} events, drop rate {manifest['drop_rate_overall']:.4f}, redraws {manifest['redraws_total']}, "
        f"cap hits {manifest['cap_hit_blocks']}, N2 centering check max |mean| {cmax:.2e}")
    return n1, n2, manifest


def draw_frames(set_dir: Path, flavour: str, meta: pd.DataFrame, n: int = N_DRAWS):
    """(draw, event frame) for the stored draws of a set, joined to the corrected event metadata (the loader of the
    results run, pointed at the closeout set)."""
    prefix = flavour.lower()
    m = meta.drop(columns=[c for c in ("anchor_date", "clock", "period", "household_id") if c in meta.columns])
    for p in sorted(set_dir.glob(f"{prefix}_batch_*.parquet")):
        avail = pq.read_schema(p).names
        cols = ["draw", "event_id", "household_id", "period", "shifted_anchor_date"] + [c for c in evf.CACHE_Y if c in avail] + [c for c in ("sigma_event", "resid_sd") if c in avail]
        batch = pd.read_parquet(p, columns=cols)
        batch = batch[batch["draw"] < n]
        f = batch.rename(columns={"shifted_anchor_date": "anchor_date"})
        for c in ("event_id", "household_id", "period"):
            f[c] = f[c].astype(str)
        for y in evf.CACHE_Y:
            if y in f.columns:
                f[y] = f[y].astype(float) * fc.BP
        if "sigma_event" in f.columns:
            f["sigma_event"] = f["sigma_event"].astype(float) * fc.BP
        f["clock"] = meta["clock"].iloc[0]
        f = f.merge(m, on="event_id", how="left").reset_index(drop=True)
        for draw, g in f.groupby("draw", sort=True):
            yield int(draw), g.reset_index(drop=True)


def reference_rows(clock: str, frame: pd.DataFrame, n1: Path, n2: Path, cutoffs, calendar: pd.DatetimeIndex, log) -> tuple[list[dict], pd.DataFrame]:
    """Per-draw covariance levels for the corrected and common-cohort panels under every spec, both flavours, and the
    reference probabilities of the observed values."""
    meta = frame[["event_id", "household_id", "clock", "period", "anchor_date", "anchor_year", "window_end_date", "record_source", "permno", "direction", "dup_any_line", "dup_all_lines"]].copy()
    base_hh = set(sp.to_panel(evf.load_event_frame(clock, calendar=calendar), sp.headline_spec(clock), cutoffs(clock, "ff5mom_h20"))["household_id"].astype(str))
    specs = {k: f(clock) for k, f in SPECS.items()}
    observed: dict[tuple[str, str], tuple[float, int, int]] = {}
    for key, spec in specs.items():
        cuts = cutoffs(clock, spec.outcome) if spec.winsorized else None
        for cohort, fr in (("corrected", frame), ("common_cohort", frame.loc[frame["household_id"].astype(str).isin(base_hh)])):
            panel = sp.to_panel(fr, spec, cuts)
            r = cgm.tau2_i1(rs.block_cluster(panel, calendar), spec.weights, anchor="anchor_blk")
            observed[(key, cohort)] = (float(r["tau2_hat"]), int(r["n_events_a"] + r["n_events_b"]), int(r["n_households"]))
    per_draw: list[dict] = []
    for flavour, d in (("N2", n2), ("N1", n1)):
        t0 = time.time()
        n_seen = 0
        for draw, g in draw_frames(d, flavour, meta):
            for key, spec in specs.items():
                cuts = cutoffs(clock, spec.outcome) if spec.winsorized else None
                for cohort in ("corrected", "common_cohort"):
                    gg = g if cohort == "corrected" else g.loc[g["household_id"].astype(str).isin(base_hh)]
                    try:
                        panel = sp.to_panel(gg, spec, cuts, draw=draw)
                        v = tau_mod.tau2_household_covariance(tau_mod.household_table(panel, spec.weights))
                    except (ValueError, KeyError):
                        v = np.nan
                    per_draw.append({"clock": clock, "flavour": flavour, "spec": key, "cohort": cohort, "draw": draw, "tau2": v})
            n_seen += 1
            if n_seen % 500 == 0:
                log(f"{clock}/{flavour}: {n_seen} draws, {time.time() - t0:.0f} s")
        log(f"{clock}/{flavour}: {n_seen} draws in {time.time() - t0:.0f} s")
    pdf = pd.DataFrame(per_draw)
    rows = []
    for (key, cohort), (obs, n_ev, n_hh) in observed.items():
        for flavour in ("N2", "N1"):
            vals = pdf.loc[(pdf["flavour"] == flavour) & (pdf["spec"] == key) & (pdf["cohort"] == cohort), "tau2"].to_numpy(dtype=float)
            pl = placebo_p_value(obs, vals, "greater")
            rows.append(rs.row(clock, "persistence_reference", f"{cohort}|{key}", f"tau2_placebo_p_{'centered' if flavour == 'N2' else 'timing'}", pl["p"], "p",
                               placebo_p=pl["p"], boot_lo=pl["null_mean"], boot_hi=pl["null_q95"], n_events=n_ev, n_households=n_hh,
                               notes=f"observed tau2 {obs:.2f} bp2; {'centered' if flavour == 'N2' else 'timing-only'} shifted-date datasets regenerated on the corrected event set, {pl['n_null']} draws; "
                                     f"placebo mean in boot_lo, 95th percentile in boot_hi; add-one upper-tail rank of the level"))
        rows.append(rs.row(clock, "persistence_reference", f"{cohort}|{key}", "tau2_observed", obs, "bp2", n_events=n_ev, n_households=n_hh))
    return rows, pdf


def set_row(clock: str, man: dict) -> dict:
    """One reference_sets.csv row from a set's N2 manifest (the centering check excludes the cell count)."""
    return {"clock": clock, "set_name": set_name(clock), "n_draws": man["n_draws"], "n_events_in": man["n_events_in"], "n_households_in": man["n_households_in"],
            "n_blocks": man["n_blocks"], "rows_per_draw_mean": man["rows_per_draw_mean"], "drop_rate_overall": man["drop_rate_overall"],
            "drop_rate_A": man["drop_rate_by_period"].get("A"), "drop_rate_B": man["drop_rate_by_period"].get("B"), "redraws_total": man["redraws_total"],
            "cap_hit_blocks": man["cap_hit_blocks"], "blocks_with_redraw_share": man["blocks_with_redraw_share"], "shift_abs_mean": man["shift_abs_mean"],
            "shift_positive_share": man["shift_positive_share"], "overlaps_actual_share": man["n1_flag_shares"].get("overlaps_actual"),
            "overlaps_cross_period_share": man["n1_flag_shares"].get("overlaps_cross_period"), "n2_centering_check_max_abs": max(v for k, v in man["n2_centering_check_max_abs"].items() if k != "n_household_period_cells"),
            "seed_scheme": man["seed_scheme"], "seconds_n1": man["seconds"], "seconds_n2": man["seconds_n2"]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="shifted-date reference datasets and probabilities for the corrected persistence estimates")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--summary-only", action="store_true", help="rebuild reference_sets.csv from the existing set manifests without regenerating draws")
    args = ap.parse_args(argv)
    log = fc.logger("reference")
    t0 = time.time()
    if args.summary_only:
        sets = []
        for clock in args.clocks.split(","):
            man = json.loads((PANELS_ROOT / clock / "N2" / set_name(clock) / "manifest.json").read_text())
            sets.append(set_row(clock, man))
        fc.write_csv(pd.DataFrame(sets), OUT_SETS)
        log(f"reference_sets.csv rebuilt from the N2 set manifests for clocks {args.clocks}; no draws regenerated")
        return
    calendar = pm.v2_trading_calendar()
    rows, sets = [], []
    entries = [fc.file_entry(cx.OUT_CACHE), fc.file_entry(fcache.MANIFEST_PATH)]
    for clock in args.clocks.split(","):
        ev, frame = corrected_events(clock)
        entries.append(fc.file_entry(cu.RESTRICTED / f"frame_{clock}_corrected.parquet", rows=len(frame)))
        n1, n2, man = generate(clock, ev, log)
        sets.append(set_row(clock, man))
        r, pdf = reference_rows(clock, frame, n1, n2, evf.pooled_cutoffs, calendar, log)
        rows += r
        pdf.to_parquet(fc.RESTRICTED / f"reference_draw_statistics_{clock}.parquet", index=False)
    fc.write_rows(rows, OUT_REF)
    fc.write_csv(pd.DataFrame(sets), OUT_SETS)
    fc.write_manifest("reference", entries, config={"n_draws": N_DRAWS, "batch": BATCH, "set_tag": SET_TAG, "block_length": 20, "shift": [40, 250],
                                                     "max_infeasible_share": 0.10, "max_redraws": 50, "specs": list(SPECS)},
                      extra={"seconds": round(time.time() - t0, 1)})
    pd.set_option("display.width", 250)
    out = rs.results_frame(rows)
    print(out[out["statistic"] != "tau2_observed"][["clock", "variant", "statistic", "placebo_p", "boot_lo", "boot_hi", "n_events", "n_households"]].round(4).to_string(index=False))
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
