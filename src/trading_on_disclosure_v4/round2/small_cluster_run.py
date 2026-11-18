"""Task C5: bias-reduced (CR2) household-clustered inference with Imbens-Kolesar degrees of freedom and the
Carter-Schnepel-Steigerwald effective number of clusters for the principal covariance regressions.

python -m trading_on_disclosure_v4.round2.small_cluster_run
Reads the restricted purchase-level design (export 5) and writes <run>/small_cluster/small_cluster_inference.csv.
The covariance-scale SE is the slope SE times the original weighted period-A residual variance (conditional).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.round2 import common as cm
from trading_on_disclosure_v4.round2 import small_cluster as sc

DESIGN = cm.RESTRICTED / "inputs" / "covariance_regression_design.csv"
OUT = cm.RUN_DIR / "small_cluster"
RESULTS_KEY = {"principal": ("tau", "event_share"), "intercept_omitted": ("robustness", "no_alpha")}


def run_one(g: pd.DataFrame, working: str) -> dict:
    y = g["y_B_bp"].to_numpy(float)
    X = np.column_stack([np.ones(len(g)), g["x_A_bp"].to_numpy(float), g["activity_log"].to_numpy(float)])
    v = g["event_weight"].to_numpy(float)
    cl = g["household_index"].to_numpy()
    r = sc.cr2_inference(y, X, v, cl, coef_index=1, working=working)
    c1 = sc.css_effective_clusters(X, v, cl, coef_index=1, rho=1.0, working=working)
    c0 = sc.css_effective_clusters(X, v, cl, coef_index=1, rho=0.0, working=working)
    return {"r": r, "css1": c1, "css0": c0}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Task C5")
    ap.parse_args(argv)
    log = cm.logger("small_cluster")
    t0 = time.time()
    d = pd.read_csv(DESIGN)
    rows = []
    for (clock, spec), g in d.groupby(["clock", "specification"], sort=False):
        var_x = float(g["var_w_x_A"].iloc[0])
        analysis, variant = RESULTS_KEY[spec]
        kappa = cm.frozen_value(clock, analysis, variant, "tau2")
        se_two_way = cm.frozen_value(clock, analysis, variant, "tau2", "se_block")
        for working in ("inverse_weights", "identity"):
            o = run_one(g, working)
            r, c1, c0 = o["r"], o["css1"], o["css0"]
            b = r["estimate"]
            if abs(b * var_x - kappa) > 1e-8 * max(1.0, abs(kappa)):
                raise AssertionError(f"{clock} {spec}: slope x Var_w(x) {b * var_x} differs from frozen {kappa}")
            rows.append({"clock": clock, "specification": spec, "working_model": working, "n_purchases_B": int(r["n_obs"]), "n_households": int(r["n_clusters"]),
                         "slope": b, "covariance_bp2": b * var_x, "var_w_x_A_bp2": var_x,
                         "se_slope_cr2": r["se_cr2"], "se_cov_cr2_bp2": r["se_cr2"] * var_x,
                         "se_slope_cr1_household": r["se_cr1_cgm"], "se_cov_cr1_household_bp2": r["se_cr1_cgm"] * var_x,
                         "se_cov_original_two_way_bp2": se_two_way, "original_two_way": "household x 20-session entry block CGM (results.csv se_block)",
                         "dof_ik": r["dof_ik"], "dof_bm": r["dof_bm"], "crit_ik": r["crit_ik"],
                         "ci_slope_low_ik": r["ci_low_ik"], "ci_slope_high_ik": r["ci_high_ik"],
                         "ci_cov_low_ik_bp2": r["ci_low_ik"] * var_x, "ci_cov_high_ik_bp2": r["ci_high_ik"] * var_x,
                         "rho_hat_ik": r["rho_hat"], "sigma2_hat_ik": r["sigma2_hat"], "icc_hat_ik": r.get("icc_hat", np.nan),
                         "ik_truncated": bool(r.get("ik_truncated", False)), "ik_omega_psd": bool(r.get("ik_omega_psd", True)),
                         "G_star_rho1": c1["G_star"], "Gamma_rho1": c1["Gamma"], "G_star_rho0": c0["G_star"], "Gamma_rho0": c0["Gamma"],
                         "G_star_rho09999": c1.get("by_rho", {}).get(0.9999, {}).get("G_star", np.nan) if isinstance(c1.get("by_rho"), dict) else np.nan,
                         "working_model_description": r["working_model_description"],
                         "assumptions": "household clusters only; conditional on the estimated benchmarks, cutoffs and period-A residuals; "
                                        "does not allow for shocks shared across households (the two-way SE does)"})
            log(f"{clock} {spec} {working}: CR2 SE (cov) {r['se_cr2'] * var_x:.1f} vs CR1 {r['se_cr1_cgm'] * var_x:.1f} vs two-way {se_two_way:.1f}; "
                f"IK dof {r['dof_ik']:.2f}, BM dof {r['dof_bm']:.2f}; G* rho=1 {c1['G_star']:.2f}, rho=0 {c0['G_star']:.2f}")
    out = pd.DataFrame(rows)
    cm.write_csv(out, OUT / "small_cluster_inference.csv")
    cm.write_json({"design": {**cm.file_entry(DESIGN), "location": "restricted"}, "module": "src/trading_on_disclosure_v4/round2/small_cluster.py",
                   "methods": "docs/methods/small_cluster_inference.md", "principal_working_model": "inverse_weights (sqrt(v)-transformed rows)",
                   "seconds": round(time.time() - t0, 1)}, OUT / "small_cluster_manifest.json")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
