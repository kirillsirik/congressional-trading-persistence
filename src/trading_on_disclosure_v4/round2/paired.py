"""Task C1: validated run of the supplied paired engine, invalid-resample audit, and cross-checks against the original
estimator functions and the frozen marginal intervals.

python -m trading_on_disclosure_v4.round2.paired
Reads <run>/inputs/principal_household_means.csv; the engine writes <run>/paired/; this module adds
<run>/paired/{input_validation.json, resample_audit.csv, selection_multiplicities_<clock>.csv, crosscheck_original.csv,
marginal_reproduction.csv, sign_frequencies.csv, summary_with_denominators.csv}.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.ext import decompose as dc
from trading_on_disclosure_v4.inference import block_bootstrap as bb
from trading_on_disclosure_v4.outcomes.seeds import seed_from
from trading_on_disclosure_v4.round2 import common as cm

ENGINE = Path(__file__).resolve().with_name("paired_engine.py")
INPUT = cm.INPUTS / "principal_household_means.csv"
OUT = cm.RUN_DIR / "paired"
N_BOOT = 4999
EXPECTED = {"T": (38, 7964), "P": (30, 6095)}
TERMS = ("K11", "K00", "KDD", "K0D", "KD0", "difference")


def multiplicities(clock: str, n: int, n_boot: int = N_BOOT) -> np.ndarray:
    """The original household-bootstrap selections: seed_from('tau/event_share', clock, 'household bootstrap')."""
    rng = np.random.default_rng(seed_from("tau/event_share", clock, "household bootstrap"))
    return np.array([bb._household_draw(rng, n) for _ in range(n_boot)], dtype=np.int64)


def validate_input(frame: pd.DataFrame, log) -> dict:
    out = {}
    for clock, (n_h, n_p) in EXPECTED.items():
        for scale in ("raw", "winsorized"):
            g = frame[(frame["clock"] == clock) & (frame["scale"] == scale)].sort_values("household_index")
            keys = g["household_index"].to_numpy()
            if len(g) != n_h or len(np.unique(keys)) != n_h or not np.issubdtype(keys.dtype, np.integer):
                raise AssertionError(f"{clock}/{scale}: {len(g)} rows, keys not {n_h} unique integers")
            if int(g["n_A"].sum() + g["n_B"].sum()) != n_p or (g[["n_A", "n_B"]] < 1).any().any():
                raise AssertionError(f"{clock}/{scale}: purchase counts do not total {n_p}")
            C = tau_mod.activity_design(g["n_A"].to_numpy(float))
            rank = int(np.linalg.matrix_rank(C))
            if rank != 2:
                raise AssertionError(f"{clock}/{scale}: activity design rank {rank}")
            out[f"{clock}/{scale}"] = {"households": n_h, "purchases": n_p, "keys_unique_integers": True, "activity_design_rank": rank,
                                       "distinct_n_A": int(g["n_A"].nunique())}
        a = frame[(frame["clock"] == clock) & (frame["scale"] == "raw")].sort_values("household_index")
        b = frame[(frame["clock"] == clock) & (frame["scale"] == "winsorized")].sort_values("household_index")
        for f in ("household_index", "n_A", "n_B"):
            if not np.array_equal(a[f].to_numpy(), b[f].to_numpy()):
                raise AssertionError(f"{clock}: raw and winsorized differ in {f}")
    log("input validation passed: " + "; ".join(f"{k} {v['households']} households / {v['purchases']} purchases" for k, v in out.items()))
    return out


def baseline_agreement(points: pd.DataFrame, log) -> list[dict]:
    """Engine point estimates against the frozen decomposition rows at full precision."""
    rows = []
    stat = {"K11": "tau2_total", "K00": "tau2_base", "KDD": "tau2_adj", "K0D": "tau2_cross_base_adj", "KD0": "tau2_cross_adj_base"}
    for r in points.itertuples(index=False):
        tag = "raw" if r.scale == "raw" else "wins"
        for term, name in stat.items():
            frozen = cm.frozen_value(r.clock, "decomposition", "no_alpha_omit", f"{name}_{tag}", path=cm.FROZEN_DECOMP)
            val = float(getattr(r, term))
            gap = abs(val - frozen)
            if gap > 1e-8 * max(1.0, abs(frozen)):
                raise AssertionError(f"{r.clock}/{r.scale} {term}: engine {val} vs frozen {frozen}")
            rows.append({"clock": r.clock, "scale": r.scale, "term": term, "engine_point": val, "frozen": frozen, "abs_diff": gap,
                         "frozen_source": f"data/v4/ext/decomposition.csv no_alpha_omit/{name}_{tag}"})
    log(f"engine point estimates agree with decomposition.csv: max |diff| {max(r['abs_diff'] for r in rows):.2e} bp2")
    return rows


def resample_audit(frame: pd.DataFrame, clock: str, mult: np.ndarray) -> pd.DataFrame:
    """Per resample and scale: households selected, marginal validity of each benchmark under the original rule
    (_tau2_from_sums: at least three households and positive weighted period-A residual variance), joint validity
    under the engine's rule, and the reason for any invalid resample."""
    rows = []
    for scale in ("raw", "winsorized"):
        g = frame[(frame["clock"] == clock) & (frame["scale"] == scale)].sort_values("household_index").reset_index(drop=True)
        n_a, n_b = g["n_A"].to_numpy(float), g["n_B"].to_numpy(float)
        w0 = np.sqrt(n_a * n_b)
        for i, m in enumerate(mult):
            w = w0 * m
            keep = w > 0
            rec = {"clock": clock, "scale": scale, "resample": i, "n_distinct_households": int(keep.sum()), "max_multiplicity": int(m.max())}
            for k in ("Y1", "Y0"):
                if keep.sum() < bb.MIN_HOUSEHOLDS_PER_DRAW:
                    rec[f"valid_{k}"], rec[f"reason_{k}"], rec[f"sxx_{k}"] = False, "fewer_than_three_households", np.nan
                    continue
                x = tau_mod._wls_residual(g[f"{k}_A"].to_numpy(float)[keep], tau_mod.activity_design(n_a[keep]), w[keep])
                sxx = float(np.sum(w[keep] * x * x))
                rec[f"sxx_{k}"] = sxx
                rec[f"valid_{k}"] = bool(sxx > 0)
                rec[f"reason_{k}"] = "" if sxx > 0 else "nonpositive_period_A_residual_variance"
            rec["valid_joint"] = bool(rec["valid_Y1"] and rec["valid_Y0"])
            rec["masks_coincide"] = bool(rec["valid_Y1"] == rec["valid_Y0"])
            rows.append(rec)
    return pd.DataFrame(rows)


def original_panels(ctx, clock: str) -> dict[str, pd.DataFrame]:
    """The four purchase-level panels of the paired comparison in the original estimator's panel form."""
    base = dc.multi_outcome_panel(ctx, clock)
    cols = ["event_id", "household_id", "period", "anchor_date"]
    out = {}
    for k, col in (("Y1", "Y1"), ("Y0", "Y0")):
        out[f"raw/{k}"] = base[cols].assign(y_h20=base[col].to_numpy(float))
        out[f"winsorized/{k}"] = base[cols].assign(y_h20=dc.winsorized_column(base, col, clock))
    return out


def crosscheck_original(ctx, clock: str, draws: pd.DataFrame, log) -> list[dict]:
    """Every engine draw of K11 and K00 against the original household bootstrap (tau2_household_bootstrap) run on the
    original purchase panels with the principal seed: identical selections must give identical draws."""
    seed = seed_from("tau/event_share", clock, "household bootstrap")
    rows = []
    for key, panel in original_panels(ctx, clock).items():
        scale, k = key.split("/")
        res = bb.tau2_household_bootstrap(panel, "ES", N_BOOT, seed)
        orig = np.asarray(res["statistics"], float)
        term = "K11" if k == "Y1" else "K00"
        eng = draws[(draws["clock"] == clock) & (draws["scale"] == scale)].set_index("resample")[term]
        aligned = orig[eng.index.to_numpy()]
        diff = np.abs(aligned - eng.to_numpy())
        rows.append({"clock": clock, "scale": scale, "benchmark": k, "term": term, "original_function": "inference.block_bootstrap.tau2_household_bootstrap",
                     "seed": seed, "n_draws_original": int(len(orig)), "n_valid_original": int(res["n_valid"]),
                     "n_engine_valid": int(len(eng)), "max_abs_diff_bp2": float(np.nanmax(diff)) if len(diff) else np.nan,
                     "original_percentile_low": float(res["tau2_low"]), "original_percentile_high": float(res["tau2_high"])})
        log(f"{clock} {scale} {k}: original bootstrap vs engine, max |diff| {rows[-1]['max_abs_diff_bp2']:.2e} over {len(eng)} common valid draws")
    return rows


def crosscheck_terms(frame: pd.DataFrame, clock: str, mult: np.ndarray, draws: pd.DataFrame, picks: list[int]) -> list[dict]:
    """Selected resamples: all five components from the original decomposition function (ext.decompose.terms) with
    weights w_h x multiplicity, against the engine's resample-level output."""
    rows = []
    for scale in ("raw", "winsorized"):
        g = frame[(frame["clock"] == clock) & (frame["scale"] == scale)].sort_values("household_index").reset_index(drop=True)
        hh = pd.DataFrame({"n_A": g["n_A"].astype(float), "n_B": g["n_B"].astype(float), "Y1_A": g["Y1_A"], "Y1_B": g["Y1_B"],
                           "Y0_A": g["Y0_A"], "Y0_B": g["Y0_B"]})
        w0 = np.sqrt(hh["n_A"] * hh["n_B"]).to_numpy()
        eng = draws[(draws["clock"] == clock) & (draws["scale"] == scale)].set_index("resample")
        for i in picks:
            m = mult[i]
            keep = m > 0
            t = dc.terms(hh.loc[keep].reset_index(drop=True), "Y1", "Y0", (w0 * m)[keep])
            mapping = {"K11": t["total"], "K00": t["base"], "KDD": t["adj"], "K0D": t["cross_base_adj"], "KD0": t["cross_adj_base"]}
            for term, val in mapping.items():
                e = float(eng.loc[i, term]) if i in eng.index else np.nan
                rows.append({"clock": clock, "scale": scale, "resample": int(i), "term": term, "original_decompose_terms": float(val),
                             "engine": e, "abs_diff": abs(float(val) - e)})
    return rows


def reproduce_marginals(ctx, clock: str, draws: pd.DataFrame, log) -> list[dict]:
    """The frozen Table 3 marginal intervals from their own seeds, and the paired engine's marginal percentiles
    (principal seed) beside them."""
    specs = sp.analysis_specs(clock)
    rows = []
    cases = (("principal winsorized K11", "tau/event_share", "tau", "event_share", "winsorized", "K11"),
             ("intercept-omitted winsorized K00", "robustness/no_alpha", "robustness", "no_alpha", "winsorized", "K00"),
             ("principal raw K11", "robustness/raw_outcome", "robustness", "raw_outcome", "raw", "K11"))
    for label, name, analysis, variant, scale, term in cases:
        spec = specs[name]
        panel = ctx.panel(spec)
        seed = seed_from(spec.name, clock, "household bootstrap")
        res = bb.tau2_household_bootstrap(panel, spec.weights, N_BOOT, seed)
        fl, fh = cm.frozen_value(clock, analysis, variant, "tau2", "boot_lo"), cm.frozen_value(clock, analysis, variant, "tau2", "boot_hi")
        d = draws[(draws["clock"] == clock) & (draws["scale"] == scale)][term].to_numpy(float)
        lo, hi = np.quantile(d, [0.025, 0.975])
        rows.append({"clock": clock, "case": label, "frozen_results_key": f"{analysis}/{variant}/tau2", "original_seed_name": spec.name, "original_seed": seed,
                     "frozen_boot_lo": fl, "frozen_boot_hi": fh, "reproduced_boot_lo": float(res["tau2_low"]), "reproduced_boot_hi": float(res["tau2_high"]),
                     "reproduced_n_valid": int(res["n_valid"]), "abs_diff_lo": abs(float(res["tau2_low"]) - fl), "abs_diff_hi": abs(float(res["tau2_high"]) - fh),
                     "paired_engine_seed": seed_from("tau/event_share", clock, "household bootstrap"), "paired_engine_lo": float(lo), "paired_engine_hi": float(hi),
                     "paired_engine_n_valid": int(len(d)), "same_selections_as_frozen": spec.name == "tau/event_share"})
        log(f"{clock} {label}: frozen [{fl:.3f}, {fh:.3f}] reproduced [{res['tau2_low']:.3f}, {res['tau2_high']:.3f}]; paired engine [{lo:.3f}, {hi:.3f}]")
    return rows


def sign_frequencies(draws: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (clock, scale), d in draws.groupby(["clock", "scale"], sort=False):
        a = audit[(audit["clock"] == clock) & (audit["scale"] == scale)]
        n_joint = int(a["valid_joint"].sum())
        if n_joint != len(d):
            raise AssertionError(f"{clock}/{scale}: engine kept {len(d)} resamples, audit counts {n_joint} jointly valid")
        rows.append({"clock": clock, "scale": scale, "n_resamples": N_BOOT, "n_valid_joint": n_joint, "n_invalid_joint": N_BOOT - n_joint,
                     "n_valid_Y1_marginal": int(a["valid_Y1"].sum()), "n_valid_Y0_marginal": int(a["valid_Y0"].sum()),
                     "masks_coincide_all": bool(a["masks_coincide"].all()),
                     "n_opposite_sign": int(((d["K11"] * d["K00"]) < 0).sum()), "share_opposite_sign": float(((d["K11"] * d["K00"]) < 0).mean()),
                     "n_K11_pos_K00_neg": int(((d["K11"] > 0) & (d["K00"] < 0)).sum()), "share_K11_pos_K00_neg": float(((d["K11"] > 0) & (d["K00"] < 0)).mean()),
                     "n_K11_neg_K00_pos": int(((d["K11"] < 0) & (d["K00"] > 0)).sum()),
                     "n_difference_positive": int((d["difference"] > 0).sum()), "share_difference_positive": float((d["difference"] > 0).mean()),
                     "denominator": "jointly valid common resamples",
                     "interpretation": "descriptive resampling frequency; not a posterior probability or calibrated p-value"})
    return pd.DataFrame(rows)


def identity_check(draws: pd.DataFrame, points: pd.DataFrame) -> dict:
    allr = pd.concat([draws.assign(kind="resample"), points.assign(kind="point", resample=-1)], ignore_index=True)
    gap = allr["K11"] - (allr["K00"] + allr["KDD"] + allr["K0D"] + allr["KD0"])
    rel = gap.abs() / np.maximum(1.0, allr["K11"].abs())
    return {"n_checked": int(len(allr)), "max_abs_gap_bp2": float(gap.abs().max()), "max_rel_gap": float(rel.max()), "tolerance_rel": 1e-8,
            "passed": bool((rel < 1e-8).all())}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Task C1: validated paired engine run and audit")
    ap.add_argument("--skip-engine", action="store_true", help="reuse an existing engine output (still validated)")
    a = ap.parse_args(argv)
    log = cm.logger("paired")
    t0 = time.time()
    frame = pd.read_csv(INPUT)
    validation = validate_input(frame, log)
    py = sys.executable
    if not a.skip_engine:
        for args in (["--self-check"], ["--input", str(INPUT.relative_to(cm.REPO_ROOT)), "--output", str(OUT.relative_to(cm.REPO_ROOT))]):
            cmd = [py, str(ENGINE.relative_to(cm.REPO_ROOT)), *args]
            shown = [cm.rel(Path(py)) if Path(py).is_relative_to(cm.REPO_ROOT) else Path(py).name, *cmd[1:]]
            cm.record_command(" ".join(shown), "Task C1 supplied engine")
            r = subprocess.run(cmd, cwd=cm.REPO_ROOT, capture_output=True, text=True)
            log(f"engine {' '.join(args[:1])}: exit {r.returncode}")
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-2000:])
    points = pd.read_csv(OUT / "point_statistics.csv")
    draws = pd.read_csv(OUT / "paired_resamples.csv")
    agreement = baseline_agreement(points, log)
    ctx, _ = cm.load_context()
    audits, cross, cross_terms, marg = [], [], [], []
    for clock, (n_h, _) in EXPECTED.items():
        mult = multiplicities(clock, n_h)
        keys = frame[(frame["clock"] == clock) & (frame["scale"] == "raw")].sort_values("household_index")["household_index"].to_numpy()
        mfile = cm.write_csv(pd.DataFrame(mult, columns=[f"hh_{k}" for k in keys]).assign(resample=np.arange(N_BOOT))[["resample", *[f"hh_{k}" for k in keys]]],
                             OUT / f"selection_multiplicities_{clock}.csv")
        log(f"{clock}: selection matrix sha256 {hashlib.sha256(mult.tobytes()).hexdigest()[:16]} ({mfile.name})")
        audits.append(resample_audit(frame, clock, mult))
        cross += crosscheck_original(ctx, clock, draws, log)
        picks = sorted(set([0, 1, 2, 1000, 2500, N_BOOT - 1] + list(np.random.default_rng(20261110).choice(N_BOOT, 10, replace=False))))
        cross_terms += crosscheck_terms(frame, clock, mult, draws, picks)
        marg += reproduce_marginals(ctx, clock, draws, log)
    audit = pd.concat(audits, ignore_index=True)
    signs = sign_frequencies(draws, audit)
    ident = identity_check(draws, points)
    summary = pd.read_csv(OUT / "paired_summary.csv")
    summary = summary.merge(signs[["clock", "scale", "n_valid_joint"]], on=["clock", "scale"], how="left")
    summary["quantile_convention"] = "linear interpolation (pandas/numpy default), 2.5th and 97.5th percentiles over jointly valid resamples"
    summary["se_convention"] = "standard deviation over jointly valid resamples, ddof=1"
    cm.write_csv(audit, OUT / "resample_audit.csv")
    cm.write_csv(pd.DataFrame(cross), OUT / "crosscheck_original.csv")
    cm.write_csv(pd.DataFrame(cross_terms), OUT / "crosscheck_decompose_terms.csv")
    cm.write_csv(pd.DataFrame(marg), OUT / "marginal_reproduction.csv")
    cm.write_csv(signs, OUT / "sign_frequencies.csv")
    cm.write_csv(summary, OUT / "summary_with_denominators.csv")
    cm.write_csv(pd.DataFrame(agreement), OUT / "baseline_agreement.csv")
    cm.write_json({"input": cm.file_entry(INPUT), "validation": validation, "decomposition_identity": ident,
                   "max_abs_diff_original_bootstrap": float(pd.DataFrame(cross)["max_abs_diff_bp2"].max()),
                   "max_abs_diff_decompose_terms": float(pd.DataFrame(cross_terms)["abs_diff"].max()),
                   "seed_convention": "seed_from('tau/event_share', clock, 'household bootstrap'); numpy default_rng; one bincount(integers(0, n, n)) per resample",
                   "household_order": "ascending household_index = ascending original household id", "n_boot": N_BOOT,
                   "environment": cm.environment(), "seconds": round(time.time() - t0, 1)}, OUT / "input_validation.json")
    log(f"identity: {ident}; done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
