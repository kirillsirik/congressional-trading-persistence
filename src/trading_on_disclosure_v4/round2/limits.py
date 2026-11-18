"""Task C4 and Export 4: exact concentrated-gain limits (household and household x entry-block clustering) and the
all-gain bound of Proposition E.1 on the actual initial cohort and on every reference dataset of the pooled-mean
experiment (centered draws 0-499, most-active recipient, ties by sorted household id, unwinsorized returns).

python -m trading_on_disclosure_v4.round2.limits
Writes <run>/limits/ (dataset-level limits, summaries, verification) and the cell-level Export 4 under the restricted run
folder (household_block_sums.csv.gz, recipients.csv).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import power_common as pc
from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tests_t1_t4 as tt
from trading_on_disclosure_v4.ext import activity_power as ap_mod
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.round2 import common as cm

OUT = cm.RUN_DIR / "limits"
N_DRAWS = 500
BLOCK = 20
COND_TOL = 1e-10  # det(J) / (v00 v11) below this: the all-gain bound is reported as ill-conditioned
MU_GRID = (50.0, 100.0, 200.0, 400.0, 800.0, 1600.0)
LARGE_MU = (1e3, 1e4, 1e5, 1e6)


def lam(g: int) -> float:
    return g / (g - 1.0) if g > 1 else np.nan


def dataset_quantities(hh_codes: np.ndarray, blk: np.ndarray, y: np.ndarray, hh_names: np.ndarray) -> dict:
    """Every quantity of Propositions 3 and E.1 from purchase-level codes and unwinsorized returns of one dataset."""
    N = len(y)
    G = len(hh_names)
    n_h = np.bincount(hh_codes, minlength=G).astype(float)
    S_h = np.bincount(hh_codes, weights=y, minlength=G)
    order = np.argsort(-n_h, kind="stable")  # most active; ties by sorted household id (inject_k_by_activity)
    star = int(order[0])
    tie = bool(G > 1 and n_h[order[1]] == n_h[star])
    omega = n_h / N
    w_star = float(omega[star])
    m0 = float(y.mean())
    zeta = S_h - n_h * m0
    ind = (np.arange(G) == star).astype(float)
    xi = n_h * (ind - w_star)
    lg = lam(G)
    L_hh = float((lg * ((1 - w_star) ** 2 + np.sum(np.delete(omega, star) ** 2))) ** -0.5)
    B_share = float((G - 1) / (G * (1 - w_star)))
    v00, v01, v11 = (lg * float(np.sum(a * b)) / N**2 for a, b in ((zeta, zeta), (zeta, xi), (xi, xi)))
    det = v00 * v11 - v01**2
    rel = det / (v00 * v11) if v00 > 0 and v11 > 0 else np.nan
    applicable = bool(np.isfinite(rel) and rel > COND_TOL)
    B_t2 = float((v11 * m0**2 - 2 * v01 * m0 * w_star + v00 * w_star**2) / det) if applicable else np.nan
    # gain at which the household-clustered |t| attains the bound: direction J^{-1} b, b = (m0, omega_star)
    if applicable:
        z = np.linalg.solve(np.array([[v00, v01], [v01, v11]]), np.array([m0, w_star]))
        mu_star = float(z[1] / z[0]) if z[0] != 0 else np.inf
    else:
        mu_star = np.nan
    # household x block cells
    blk_codes, blk_names = pd.factorize(blk, sort=True)
    Gb = len(blk_names)
    cell = hh_codes.astype(np.int64) * Gb + blk_codes
    cell_codes, cell_names = pd.factorize(cell, sort=True)
    Gc = len(cell_names)
    n_hj = np.bincount(cell_codes, minlength=Gc).astype(float)
    S_hj = np.bincount(cell_codes, weights=y, minlength=Gc)
    cell_h = (cell_names // Gb).astype(int)
    cell_j = (cell_names % Gb).astype(int)
    xi_hj = n_hj * ((cell_h == star).astype(float) - w_star)
    zeta_hj = S_hj - n_hj * m0
    xi_j = np.bincount(cell_j, weights=xi_hj, minlength=Gb)
    zeta_j = np.bincount(cell_j, weights=zeta_hj, minlength=Gb)
    lb, lc = lam(Gb), lam(Gc)
    A2 = (lg * np.sum(xi**2) + lb * np.sum(xi_j**2) - lc * np.sum(xi_hj**2)) / N**2
    c00 = (lg * np.sum(zeta**2) + lb * np.sum(zeta_j**2) - lc * np.sum(zeta_hj**2)) / N**2
    c01 = (lg * np.sum(zeta * xi) + lb * np.sum(zeta_j * xi_j) - lc * np.sum(zeta_hj * xi_hj)) / N**2
    L_2way = float(w_star / np.sqrt(A2)) if A2 > 0 else np.nan
    return {"N": N, "G": G, "G_blk": Gb, "G_cap": Gc, "recipient_code": star, "recipient_id": str(hh_names[star]), "recipient_n": int(n_h[star]),
            "recipient_tie": tie, "omega_star": w_star, "pooled_mean_bp": m0, "L_hh": L_hh, "B_share": B_share, "v00": v00, "v01": v01, "v11": v11,
            "det_J": det, "det_J_relative": rel, "all_gain_bound_applicable": applicable, "B_t2": B_t2,
            "B_abs_t": float(np.sqrt(B_t2)) if applicable else np.nan, "mu_at_bound_bp": mu_star,
            "bound_attained_at_nonnegative_gain": bool(applicable and np.isfinite(mu_star) and mu_star >= 0),
            "A_2way": float(A2), "A_2way_positive": bool(A2 > 0), "L_2way": L_2way, "c00_2way": float(c00), "c01_2way": float(c01),
            "t_hh_at_zero": float(m0 / np.sqrt(v00)) if v00 > 0 else np.nan,
            "t_2way_at_zero": float(m0 / np.sqrt(c00)) if c00 > 0 else np.nan,
            "cells": pd.DataFrame({"hh_code": cell_h, "entry_block": blk_names[cell_j].astype(np.int64), "n_purchases": n_hj.astype(np.int64), "sum_return_bp": S_hj})}


def t_quadratic(q: dict, mu: float, two_way: bool) -> tuple[float, float]:
    """(estimate, SE) of the pooled mean after adding mu to the recipient, from the stable quadratic form."""
    est = q["pooled_mean_bp"] + mu * q["omega_star"]
    if two_way:
        v = q["c00_2way"] + 2 * mu * q["c01_2way"] + mu * mu * q["A_2way"]
    else:
        v = q["v00"] + 2 * mu * q["v01"] + mu * mu * q["v11"]
    return est, float(np.sqrt(v)) if v > 0 else np.nan


def actual_panel(ctx, clock: str, r1: set[str]) -> pd.DataFrame:
    """Actual purchases of the R1 households with finite unwinsorized outcomes (the experiment's common sample on actual dates)."""
    spec = replace(sp.headline_spec(clock), winsorized=False, rule="all")
    p = sp.to_panel(ctx.frames[clock], spec, None)
    return p.loc[p["household_id"].isin(r1)].reset_index(drop=True)


def codes_for(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hh_codes, hh_names = pd.factorize(panel["household_id"].astype(str), sort=True)
    pos = calendar.get_indexer(pd.DatetimeIndex(panel["anchor_date"]))
    if (pos < 0).any():
        raise ValueError("entry sessions off the calendar")
    return hh_codes, pos // BLOCK, panel["y_h20"].to_numpy(float), np.asarray(hh_names)


def stored_t1(clock: str) -> pd.DataFrame:
    """The experiment's stored per-draw pooled-mean statistics (most active, one household) at every simulated gain."""
    rows = []
    for mu in MU_GRID:
        spec = {"kind": "sparse_k_activity", "k": 1, "mu_bp": mu, "selection": "most_active", "base": "R1"}
        key = f"{clock}_wins_{rc.cell_id(spec)}".replace("|", "_").replace("=", "-")
        d = pd.read_parquet(rc.unit_path(ap_mod.GRID, key), columns=["draw", "T1raw_I1blk20_est", "T1raw_I1blk20_se", "T1raw_n_events", "inj_affected_share"])
        rows.append(d.assign(mu_bp=mu))
    return pd.concat(rows, ignore_index=True)


def direct_check(panel: pd.DataFrame, calendar, q: dict, dataset: str, clock: str) -> list[dict]:
    """Original clustered-mean function (two-way household x block; one-way household) at large gains vs the formulas."""
    rows = []
    p = rc.add_block_clusters(panel, calendar, (BLOCK,))
    hit = p["household_id"].astype(str).to_numpy() == q["recipient_id"]
    for mu in LARGE_MU:
        pi = p.assign(y_h20=p["y_h20"].to_numpy(float) + mu * hit)
        r2 = tt.t1_clustered_mean(pi, anchor=f"anchor_blk{BLOCK}")
        r1 = cgm.clustered_slope(pi["y_h20"].to_numpy(float), np.ones((len(pi), 1)), np.ones(len(pi)),
                                 pd.DataFrame({"household_id": pi["household_id"].to_numpy()}), coef_index=0)
        e2, s2 = t_quadratic(q, mu, True)
        e1, s1 = t_quadratic(q, mu, False)
        rows.append({"clock": clock, "dataset": dataset, "mu_bp": mu, "t_2way_original": r2["t_stat"], "t_2way_formula": e2 / s2, "L_2way": q["L_2way"],
                     "t_hh_original": r1.t_stat, "t_hh_formula": e1 / s1, "L_hh": q["L_hh"],
                     "abs_diff_2way": abs(r2["t_stat"] - e2 / s2), "abs_diff_hh": abs(r1.t_stat - e1 / s1),
                     "gap_to_limit_2way": abs(e2 / s2 - q["L_2way"]), "gap_to_limit_hh": abs(e1 / s1 - q["L_hh"]),
                     "hh_bound_respected": bool(not q["all_gain_bound_applicable"] or (e1 / s1) ** 2 <= q["B_t2"] * (1 + 1e-12))})
    return rows


def summarize(lim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ref = lim[lim["dataset"] != "actual"]
    for clock, g in ref.groupby("clock", sort=False):
        for col in ("omega_star", "L_hh", "B_share", "B_abs_t", "L_2way", "A_2way", "N", "G_blk", "G_cap"):
            v = g[col].to_numpy(float)
            f = v[np.isfinite(v)]
            act = float(lim[(lim["clock"] == clock) & (lim["dataset"] == "actual")][col].iloc[0])
            rows.append({"clock": clock, "quantity": col, "actual": act, "n_reference": int(len(v)), "n_valid": int(len(f)),
                         "median": float(np.median(f)), "p05": float(np.quantile(f, 0.05)), "p95": float(np.quantile(f, 0.95)),
                         "min": float(f.min()), "max": float(f.max()), "mean": float(f.mean())})
        rows.append({"clock": clock, "quantity": "n_bound_applicable", "actual": float(lim[(lim["clock"] == clock) & (lim["dataset"] == "actual")]["all_gain_bound_applicable"].iloc[0]),
                     "n_reference": int(len(g)), "n_valid": int(g["all_gain_bound_applicable"].sum())})
        rows.append({"clock": clock, "quantity": "n_A_2way_positive", "actual": float(lim[(lim["clock"] == clock) & (lim["dataset"] == "actual")]["A_2way_positive"].iloc[0]),
                     "n_reference": int(len(g)), "n_valid": int(g["A_2way_positive"].sum())})
        rows.append({"clock": clock, "quantity": "n_recipient_ties", "actual": float(lim[(lim["clock"] == clock) & (lim["dataset"] == "actual")]["recipient_tie"].iloc[0]),
                     "n_reference": int(len(g)), "n_valid": int(g["recipient_tie"].sum())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Task C4 and Export 4")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    a = ap.parse_args(argv)
    log = cm.logger("limits")
    t0 = time.time()
    ctx, calendar = cm.load_context()
    key = pd.read_csv(cm.CROSSWALK).set_index("household_id")["household_index"]
    lims, cells, recips, checks, stored_cmp = [], [], [], [], []
    for clock in ("T", "P"):
        r1 = rc.r1_households(clock)
        act = actual_panel(ctx, clock, r1)
        head = sp.to_panel(ctx.frames[clock], replace(sp.headline_spec(clock), winsorized=False), None)
        same = set(act["household_id"]) == set(head["household_id"]) and len(act) == len(head)
        log(f"{clock}: R1 set {len(r1)} households; actual common sample {len(act)} purchases / {act['household_id'].nunique()} households; "
            f"equals the raw principal panel: {same}")
        datasets = [("actual", act)]
        for p in rc.iter_draws(clock, 0, a.n_draws, set_name=rc.DEFAULT_SET[clock]):
            datasets.append((f"reference_{int(p['draw'].iloc[0]):04d}", pc.restrict(p, r1)))
        stored = stored_t1(clock)
        for name, panel in datasets:
            hc, blk, y, names = codes_for(panel, calendar)
            q = dataset_quantities(hc, blk, y, names)
            c = q.pop("cells")
            c["household_index"] = key.reindex(names[c["hh_code"].to_numpy()]).to_numpy()
            cells.append(c.drop(columns="hh_code").assign(clock=clock, dataset_id=name))
            rid = q.pop("recipient_id")
            recips.append({"clock": clock, "dataset_id": name, "recipient_household_index": int(key[rid]), "recipient_n": q["recipient_n"],
                           "recipient_tie": q["recipient_tie"], "rule": "most purchases in the dataset among R1 households; ties by sorted household id"})
            lims.append({"clock": clock, "dataset": name, **{k: v for k, v in q.items() if k != "recipient_code"},
                         "recipient_household_index": int(key[rid])})
            if name in ("actual", "reference_0000", "reference_0001", "reference_0249", f"reference_{a.n_draws - 1:04d}"):
                checks += direct_check(panel, calendar, {**q, "recipient_id": rid}, name, clock)
            if name != "actual":
                draw = int(name.split("_")[1])
                s = stored[stored["draw"] == draw]
                for r in s.itertuples(index=False):
                    e, se = t_quadratic(q, r.mu_bp, True)
                    stored_cmp.append({"clock": clock, "draw": draw, "mu_bp": r.mu_bp, "est_formula": e, "est_stored": r.T1raw_I1blk20_est,
                                       "se_formula": se, "se_stored": r.T1raw_I1blk20_se, "n_formula": q["N"], "n_stored": int(r.T1raw_n_events),
                                       "share_formula": q["omega_star"], "share_stored": r.inj_affected_share})
        log(f"{clock}: {len(datasets)} datasets processed ({time.time() - t0:.0f} s)")
    lim = pd.DataFrame(lims)
    chk = pd.DataFrame(checks)
    cmp_ = pd.DataFrame(stored_cmp)
    cmp_["abs_diff_est"] = (cmp_["est_formula"] - cmp_["est_stored"]).abs()
    cmp_["abs_diff_se"] = (cmp_["se_formula"] - cmp_["se_stored"]).abs()
    cmp_["abs_diff_share"] = (cmp_["share_formula"] - cmp_["share_stored"]).abs()
    summ = summarize(lim)
    cell = pd.concat(cells, ignore_index=True)[["clock", "dataset_id", "household_index", "entry_block", "n_purchases", "sum_return_bp"]]
    cm.write_csv(lim, OUT / "dataset_limits.csv")
    cm.write_csv(summ, OUT / "limits_summary.csv")
    cm.write_csv(chk, OUT / "large_gain_verification.csv")
    agg = cmp_.groupby("clock").agg(n=("draw", "size"), max_abs_diff_est=("abs_diff_est", "max"), max_abs_diff_se=("abs_diff_se", "max"),
                                    max_abs_diff_share=("abs_diff_share", "max"), n_mismatch_n=("n_formula", lambda s: int((s != cmp_.loc[s.index, "n_stored"]).sum()))).reset_index()
    cm.write_csv(agg, OUT / "stored_experiment_reproduction.csv")
    restricted = cm.RESTRICTED / "inputs"
    restricted.mkdir(parents=True, exist_ok=True)
    cell_path = restricted / "household_block_sums.csv.gz"
    cell.to_csv(cell_path, index=False, float_format="%.17g", compression="gzip")
    cm.write_csv(pd.DataFrame(recips), restricted / "recipients.csv")
    cm.write_csv(pd.DataFrame(recips), OUT / "recipients.csv")
    cm.write_json({"household_block_sums": {**cm.file_entry(cell_path, rows=len(cell)), "location": "restricted", "columns": list(cell.columns),
                                            "definition": "one row per nonempty household x 20-session entry block cell; unwinsorized purchase returns (bp); "
                                                          "reference datasets are the centered shifted-date draws 0-499 used in the pooled-mean experiment, "
                                                          "with the stored centering; entry_block = calendar position of the (shifted) entry session // 20"},
                   "recipient_rule": "most purchases among R1 households in the dataset; ties by sorted household id (ext.activity_power.inject_k_by_activity)",
                   "sample": "R1 (persistence-eligible) households' purchases, periods A and B, unwinsorized 20-session factor outcome",
                   "cluster_correction": "g/(g-1) per inclusion-exclusion term; (N-1)/(N-k) = 1 with k = 1",
                   "bound_condition": f"all-gain bound reported when det(J)/(v00 v11) > {COND_TOL:g}; otherwise inapplicable",
                   "n_draws": a.n_draws, "seconds": round(time.time() - t0, 1)}, OUT / "limits_manifest.json")
    log(f"stored experiment reproduction: {agg.to_dict('records')}")
    log(f"large-gain verification max |t_original - t_formula|: 2way {chk['abs_diff_2way'].max():.2e}, hh {chk['abs_diff_hh'].max():.2e}")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
