"""Overlap between first-period outcome windows and second-period loading estimation (extension plan addendum
2026-10-19, section C): two exclusions of second-period purchases, the headline and no-intercept tau2 and the
decomposition terms on common households, and the centered-placebo p of the restricted headline.
Writes data/v4/ext/overlap_exclusion.csv.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.outcomes import factor_models as fm
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.ext import common as cm
from trading_on_disclosure_v4.ext import decompose as dc
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.inference.placebo import placebo_p_value

ANALYSIS = "overlap_exclusion"
H = 20
EST_SPAN = fm.ESTIMATION_GAP + fm.ESTIMATION_WINDOW  # the estimation window starts this many sessions before the anchor
PERIOD_A_END = pd.Timestamp("2020-12-31")
EXCLUSIONS = ("calendar_overlap", "own_overlap")
OUT_CSV = cm.OUT_DIR / "overlap_exclusion.csv"
LOG = cm.LOG_DIR / "overlap.log"


def overlap_flags(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Per row: est_start / est_last (calendar positions of the loading window), calendar_overlap (period-B rows whose
    window starts on or before the last session of 2020) and own_overlap (period-B rows whose window intersects the
    outcome window of a period-A purchase of the same security by the same household)."""
    p = panel.copy()
    idx = calendar.get_indexer(pd.DatetimeIndex(p["anchor_date"]))
    if (idx < 0).any():
        raise ValueError("anchor not on the calendar")
    p["anchor_idx"] = idx
    p["est_start"] = idx - EST_SPAN
    p["est_last"] = idx - fm.ESTIMATION_GAP - 1
    a_end_idx = int(calendar.searchsorted(PERIOD_A_END, side="right") - 1)
    is_b = (p["period"] == "B").to_numpy()
    p["calendar_overlap"] = is_b & (p["est_start"].to_numpy() <= a_end_idx)
    a = p.loc[p["period"] == "A", ["household_id", "permno", "anchor_idx"]].rename(columns={"anchor_idx": "a_idx"})
    b = p.loc[is_b, ["event_id", "household_id", "permno", "est_start", "est_last"]]
    m = b.merge(a, on=["household_id", "permno"], how="inner")
    hit = m[(m["a_idx"] + 1 <= m["est_last"]) & (m["a_idx"] + H >= m["est_start"])]["event_id"].unique()
    p["own_overlap"] = p["event_id"].isin(set(hit)) & is_b
    return p


def restricted_panel(p: pd.DataFrame, exclusion: str) -> tuple[pd.DataFrame, pd.Index]:
    """Rows after the exclusion with the headline eligibility rule re-applied; returns (rows, eligible households)."""
    keep = p.loc[~p[exclusion].to_numpy()]
    hh = sp.eligible_households(keep, "R1")
    return keep.loc[keep["household_id"].isin(hh)].reset_index(drop=True), hh


def tau_rows(p: pd.DataFrame, clock: str, calendar: pd.DatetimeIndex, cuts: dict[str, dict], label: str) -> dict:
    """Headline and no-intercept tau2 (winsorized at the committed cutoffs, block clusters) and the raw and
    winsorized decomposition terms on one panel."""
    out = {}
    for name, col in (("headline", "Y1"), ("no_alpha", "Y0")):
        q = pd.DataFrame({"event_id": p["event_id"], "household_id": p["household_id"], "period": p["period"], "anchor_date": p["anchor_date"],
                          "y_h20": p[col].to_numpy(dtype=float)})
        q = sp.winsorize(q, cuts[col])
        r = cgm.tau2_i1(rs.block_cluster(q, calendar), "ES", anchor="anchor_blk")
        out[f"tau2_{name}_{label}"] = r["tau2_hat"]
        out[f"tau2_{name}_{label}_se"] = r["var_wx"] * r["se_b"]
    hh = dc.household_means(p, ["Y1", "Y0"])
    w_ = tau_mod.household_weights(hh["n_A"].to_numpy(), hh["n_B"].to_numpy(), "ES")
    t = dc.terms(hh, "Y1", "Y0", w_)
    for k in dc.TERMS:
        out[f"term_{k}_raw_{label}"] = t[k]
    pw = p.assign(Y1w=dc.winsorized_column(p, "Y1", clock), Y0w=dc.winsorized_column(p, "Y0", clock))
    hhw = dc.household_means(pw, ["Y1w", "Y0w"])
    ww = tau_mod.household_weights(hhw["n_A"].to_numpy(), hhw["n_B"].to_numpy(), "ES")
    tw = dc.terms(hhw, "Y1w", "Y0w", ww)
    for k in dc.TERMS:
        out[f"term_{k}_wins_{label}"] = tw[k]
    out[f"n_events_{label}"], out[f"n_households_{label}"] = int(len(p)), int(p["household_id"].nunique())
    out[f"n_events_B_{label}"] = int((p["period"] == "B").sum())
    return out


def placebo_restricted_tau2(clock: str, exclusion: str, cuts: dict, calendar: pd.DatetimeIndex, n_placebo: int, log) -> np.ndarray:
    """The restricted headline tau2 on every centered placebo draw, with the exclusion recomputed on the draw's shifted anchors."""
    spec_raw = replace(sp.headline_spec(clock), winsorized=False)
    meta = evf.event_metadata(clock)
    vals = []
    t0 = time.time()
    for _, batch in evf.placebo_event_frames(clock, "N2", meta, dup="default", batches=True):
        for draw, g in batch.groupby("draw", sort=True):
            if draw >= n_placebo:
                continue
            g = g.reset_index(drop=True)
            q = sp.to_panel(g, spec_raw, None, draw=int(draw))
            if q.empty:
                vals.append(np.nan)
                continue
            q = q.rename(columns={"y_h20": "Y1"})
            full_hh = set(q["household_id"])
            flags = overlap_flags(q, calendar)
            keep, hh = restricted_panel(flags, exclusion)
            keep = keep.loc[keep["household_id"].isin(full_hh)]
            if keep["household_id"].nunique() < 3:
                vals.append(np.nan)
                continue
            qq = sp.winsorize(keep.rename(columns={"Y1": "y_h20"}), cuts["Y1"])
            try:
                vals.append(tau_mod.tau2_household_covariance(tau_mod.household_table(qq, "ES")))
            except ValueError:
                vals.append(np.nan)
        log(f"placebo {clock}/{exclusion}: {len(vals)} draws, {time.time() - t0:.0f} s")
    return np.asarray(vals, dtype=float)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="overlap between first-period outcomes and second-period loading estimation")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--n-placebo", type=int, default=2000)
    a = ap.parse_args(argv)
    clocks = tuple(a.clocks.split(","))
    log = lambda m: cm.log_line(LOG, m)  # noqa: E731
    t0 = time.time()
    ctx, calendar = cm.load_headline_context(clocks, with_placebo=False)
    rows: list[dict] = []
    for clock in clocks:
        base = dc.multi_outcome_panel(ctx, clock)
        cuts = {"Y1": evf.pooled_cutoffs(clock, "ff5mom_h20"), "Y0": evf.pooled_cutoffs(clock, "ff5mom_noalpha_h20")}
        flags = overlap_flags(base, calendar)
        full_hh = pd.Index(sorted(flags["household_id"].unique()))
        for exclusion in EXCLUSIONS:
            keep, hh_after = restricted_panel(flags, exclusion)
            common = full_hh.intersection(hh_after)
            keep_c = keep.loc[keep["household_id"].isin(common)].reset_index(drop=True)
            full_c = flags.loc[flags["household_id"].isin(common)].reset_index(drop=True)
            n_b = int((flags["period"] == "B").sum())
            n_excl = int(flags[exclusion].sum())
            stats = {"n_events_B_full": n_b, "n_events_B_excluded": n_excl, "share_B_excluded": n_excl / n_b if n_b else np.nan,
                     "n_households_full": int(len(full_hh)), "n_households_eligible_after": int(len(hh_after)), "n_households_common": int(len(common))}
            stats.update(tau_rows(full_c, clock, calendar, cuts, "full_common"))
            stats.update(tau_rows(keep_c, clock, calendar, cuts, "restricted"))
            vals = placebo_restricted_tau2(clock, exclusion, cuts, calendar, a.n_placebo, log)
            pl = placebo_p_value(stats["tau2_headline_restricted"], vals, "greater")
            note = (f"{exclusion}: second-period purchases {'whose loading window starts on or before 31 December 2020' if exclusion == 'calendar_overlap' else 'whose loading window intersects the outcome window of a first-period purchase of the same security by the same household'} "
                    f"excluded; eligibility re-applied; common households = eligible with and without the exclusion; winsorized at the committed cutoffs")
            for k, v in stats.items():
                unit = "bp2" if k.startswith(("tau2", "term")) else ("share" if k.startswith("share") else "count")
                extra = {}
                if k == "tau2_headline_restricted":
                    extra = dict(placebo_p=pl["p"], boot_lo=pl["null_q95"], se_block=stats["tau2_headline_restricted_se"])
                elif k == "tau2_headline_full_common":
                    extra = dict(se_block=stats["tau2_headline_full_common_se"])
                elif k.endswith("_se"):
                    continue
                rows.append(rs.row(clock, ANALYSIS, exclusion, k, v, unit, n_events=stats["n_events_restricted"], n_households=stats["n_households_common"],
                                   notes=note + (f"; centered placebo over {pl['n_null']} draws with the same exclusion (placebo mean {pl['null_mean']:.1f}, 95th percentile in boot_lo)" if k == "tau2_headline_restricted" else "")
                                   , **extra))
            log(f"{clock}/{exclusion}: B excluded {n_excl}/{n_b}, households {len(full_hh)} -> {len(hh_after)} (common {len(common)}); "
                f"tau2 headline full-common {stats['tau2_headline_full_common']:.0f} restricted {stats['tau2_headline_restricted']:.0f} (placebo p {pl['p']:.3f}); "
                f"no-alpha {stats['tau2_no_alpha_full_common']:.0f} -> {stats['tau2_no_alpha_restricted']:.0f}")
    out = cm.write_rows(rows, OUT_CSV)
    cm.write_manifest("overlap", [cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "restricted" / "n" / "forward_cache_manifest.json"),
                                  cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "events_T.parquet"),
                                  cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "events_P.parquet")], {"seconds": round(time.time() - t0, 1)})
    log(f"written {OUT_CSV} ({len(out)} rows) in {time.time() - t0:.0f} s")
    return out


if __name__ == "__main__":
    main()
