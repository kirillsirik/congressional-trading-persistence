"""Reproduction and verification of the supplied results (closeout plan sections 2 and 4).

(1) The de-duplicated public-clock spread and the headline persistence covariance recomputed from the
estimators and compared with the files on record (the extension module's checks, re-run here and written
under data/v4/final/). (2) The public-clock entry rule verified on every signal: the entry is the first
trading close strictly after the assigned public date and the outcome's first return day is the following
session. (3) The equivalence of the event-level weighted regression and the household covariance.
(4) The benchmark decomposition generated from code on the baseline panels: the four components and the
identity residual for raw and winsorized outcomes (D = winsor(Y1) - winsor(Y0)), compared with the
decomposition file on record. Writes reproduction.csv, verification.csv, decomposition_identity.csv.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.timing import return_window_dates
from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.ext import common as ec
from trading_on_disclosure_v4.ext import decompose as dc
from trading_on_disclosure_v4.ext import reproduce as rp
from trading_on_disclosure_v4.final import common as fc
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import panel as pm

OUT_REPRO = fc.OUT_DIR / "reproduction.csv"
OUT_VERIFY = fc.OUT_DIR / "verification.csv"
OUT_DECOMP = fc.OUT_DIR / "decomposition_identity.csv"
ON_FILE = fc.REPO_ROOT / "data" / "v4" / "ext" / "decomposition.csv"


def entry_rule_check(calendar: pd.DatetimeIndex) -> list[dict]:
    """Every public-clock signal: follower entry = first session strictly after the assigned public date; the
    20-session outcome window starts on the session after the entry and holds 20 sessions when complete."""
    s = load.load_signals()
    s = s.loc[s["follower_entry_date"].notna() & s["analysis_public_date"].notna()]
    pos = calendar.searchsorted(pd.DatetimeIndex(s["analysis_public_date"]), side="right")
    inside = pos < len(calendar)
    expected = pd.DatetimeIndex(np.where(inside, calendar.to_numpy()[np.minimum(pos, len(calendar) - 1)], np.datetime64("NaT")))
    observed = pd.DatetimeIndex(s["follower_entry_date"])
    same = (expected == observed) | (expected.isna() & observed.isna())
    strictly_after = (observed > pd.DatetimeIndex(s["analysis_public_date"])).sum()
    sample = s.sample(min(500, len(s)), random_state=20261106)
    first_after = 0
    full = 0
    for e in pd.DatetimeIndex(sample["follower_entry_date"]):
        w = return_window_dates(pd.Timestamp(e), calendar, 20)
        if len(w):
            first_after += int(w[0] == calendar[calendar.searchsorted(e, side="right")])
            full += int(len(w) == 20)
    return [{"check": "public_entry_equals_first_session_strictly_after_public_date", "n": int(len(s)), "n_pass": int(same.sum()), "n_fail": int((~same).sum())},
            {"check": "public_entry_strictly_after_public_date", "n": int(len(s)), "n_pass": int(strictly_after), "n_fail": int(len(s) - strictly_after)},
            {"check": "outcome_window_first_return_day_is_session_after_entry (sample)", "n": int(len(sample)), "n_pass": first_after, "n_fail": int(len(sample) - first_after)},
            {"check": "outcome_window_has_20_sessions_when_complete (sample)", "n": int(len(sample)), "n_pass": full, "n_fail": int(len(sample) - full)}]


def equivalence_check(ctx: rs.Context, calendar: pd.DatetimeIndex) -> list[dict]:
    """Event-level WLS slope x Var_w(x) against the household covariance, both clocks."""
    out = []
    for clock in ("T", "P"):
        panel = ctx.panel(sp.headline_spec(clock))
        r = cgm.tau2_i1(rs.block_cluster(panel, calendar), "ES", anchor="anchor_blk")
        cov = tau_mod.tau2_household_covariance(tau_mod.household_table(panel, "ES"))
        ev = tau_mod.estimate_tau2(panel, "ES")
        out.append({"check": f"event_wls_equals_household_covariance_{clock}", "n": int(len(panel)), "n_pass": int(abs(r["tau2_hat"] - cov) < 1e-6),
                    "n_fail": int(abs(r["tau2_hat"] - cov) >= 1e-6), "value_a": r["tau2_hat"], "value_b": cov, "abs_diff": abs(r["tau2_hat"] - cov),
                    "note": f"estimate_tau2 (no cluster) {ev.tau2_hat:.6f}"})
    return out


def decomposition_identity(ctx: rs.Context, log) -> pd.DataFrame:
    """The four components and the identity residual on the baseline panels for the three alternatives, raw and
    winsorized (D defined on the winsorized outcomes), with the difference from decomposition.csv on record."""
    on_file = pd.read_csv(ON_FILE) if ON_FILE.exists() else None
    rows = []
    for clock in ("T", "P"):
        base = dc.multi_outcome_panel(ctx, clock)
        ref = dc.refit_outcomes(base, ctx.calendar, log)
        base = base.merge(ref, on="event_id", how="left")
        for variant, yx in dc.ALTERNATIVES.items():
            ok = np.isfinite(base[["Y1", yx]].to_numpy(dtype=float)).all(axis=1)
            panel = base.loc[ok].reset_index(drop=True)
            hh = dc.household_means(panel, ["Y1", yx])
            w = tau_mod.household_weights(hh["n_A"].to_numpy(), hh["n_B"].to_numpy(), "ES")
            t_raw = dc.terms(hh, "Y1", yx, w)
            pw_ = panel.copy()
            pw_["Y1w"] = dc.winsorized_column(panel, "Y1", clock)
            pw_[f"{yx}w"] = dc.winsorized_column(panel, yx, clock)
            hhw = dc.household_means(pw_, ["Y1w", f"{yx}w"])
            ww = tau_mod.household_weights(hhw["n_A"].to_numpy(), hhw["n_B"].to_numpy(), "ES")
            t_w = dc.terms(hhw, "Y1w", f"{yx}w", ww)
            for kind, t in (("raw", t_raw), ("winsorized", t_w)):
                for k in dc.TERMS + ("identity_gap",):
                    stat = f"tau2_{k}_{'raw' if kind == 'raw' else 'wins'}" if k != "identity_gap" else f"identity_gap_{'raw' if kind == 'raw' else 'wins'}"
                    val_file = np.nan
                    if on_file is not None:
                        m = on_file[(on_file.clock == clock) & (on_file.variant == variant) & (on_file.statistic == stat)]
                        if len(m) == 1:
                            val_file = float(m["value"].iloc[0])
                    rows.append({"clock": clock, "alternative": variant, "outcome": kind, "term": k, "value_bp2": float(t[k]), "on_file_bp2": val_file,
                                 "abs_diff_from_file": abs(float(t[k]) - val_file) if np.isfinite(val_file) else np.nan,
                                 "n_events": int(len(panel)), "n_households": int(panel["household_id"].nunique()),
                                 "sum_of_four": float(t["base"] + t["adj"] + t["cross_base_adj"] + t["cross_adj_base"]) if k == "total" else np.nan})
        log(f"{clock}: decomposition identity regenerated on {len(base)} events")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="reproduction and verification of the supplied results")
    ap.parse_args(argv)
    log = fc.logger("reproduce")
    t0 = time.time()
    repro = pd.DataFrame(rp.spread_rows() + rp.tau_rows())
    fc.write_csv(repro, OUT_REPRO)
    summary = {"max_abs_diff": float(repro["abs_diff"].max()), "n_checks": int(repro["abs_diff"].notna().sum())}
    log(f"reproduction: {summary}")
    calendar = pm.v2_trading_calendar()
    ctx, _ = ec.load_headline_context(("T", "P"), with_placebo=False)
    ver = pd.DataFrame(entry_rule_check(calendar) + equivalence_check(ctx, calendar))
    fc.write_csv(ver, OUT_VERIFY)
    log("verification rows: " + "; ".join(f"{r.check}: {r.n_pass}/{r.n}" for r in ver.itertuples(index=False)))
    dec = decomposition_identity(ctx, log)
    fc.write_csv(dec, OUT_DECOMP)
    gap = dec.loc[dec["term"] == "identity_gap", "value_bp2"].abs().max()
    diff_file = dec["abs_diff_from_file"].max()
    log(f"decomposition identity: max |residual| {gap:.2e} bp2; max |difference from decomposition.csv| {diff_file:.2e} bp2")
    entries = [fc.file_entry(rp.RESULTS), fc.file_entry(rp.RERUN), fc.file_entry(ON_FILE),
               fc.file_entry(fc.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "events_T.parquet"),
               fc.file_entry(fc.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "events_P.parquet"),
               fc.file_entry(fc.REPO_ROOT / "data" / "v4" / "restricted" / "n" / "forward_cache_manifest.json"),
               fc.file_entry(fc.REPO_ROOT / "outputs" / "frames" / "signals.csv"), fc.file_entry(evf.POOLED_CSV)]
    fc.write_manifest("reproduce", entries, config={"spread_test": rp.rerun.PRIMARY_TEST_ID, "weights": "ES", "clocks": ["T", "P"]},
                      extra={"seconds": round(time.time() - t0, 1), **summary, "decomposition_max_abs_identity_gap_bp2": float(gap),
                             "decomposition_max_abs_diff_from_file_bp2": float(diff_file)})
    (OUT_REPRO.with_suffix(".json")).write_text(json.dumps({**summary, "seconds": round(time.time() - t0, 1)}, indent=2))
    pd.set_option("display.width", 250)
    print(ver.to_string(index=False))
    print(dec[dec.term.isin(("total", "identity_gap"))].round(3).to_string(index=False))
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
