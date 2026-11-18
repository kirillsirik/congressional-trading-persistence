"""Fractional-intercept sensitivity, kept separate from the affine interpolation curve (scripts/writing_sample/figure3.py).

python -m trading_on_disclosure_v4.writing_sample.fractional
Z_i(c) = 10^4 [prod(1 + r) - prod(1 + rf + c*alpha + beta'f)] over the 20 sessions after entry, with the forward
cache's own fitted alpha and loadings (fixed), computed with the cache's arithmetic so that Z(0) and Z(1) equal the
stored factor-only and event-study outcomes bit for bit. For each c on 0, 0.01, ..., 1 (and at the trade-date raw
purchase continuation estimate lambda_hat when it lies in [0, 1]) the persistence covariance is computed on the
fixed persistence cohort (events, w = sqrt(n_A n_B), activity log(1+n_A) refit in both periods), raw and
winsorized. Winsorized cutoffs for each c: p1/p99 by period of Z(c) over the pooled purchase population of the
committed cutoff table (same rounding), fixed within resamples; at c = 0, 1 they equal the committed cutoffs.
Pointwise bands use the original 4,999 common household selections. The joint plug-in interval re-estimates
lambda (entry-block FE regression with household multiplicities as pseudo-household weights) and the covariance at
c = lambda* in each common selection, cutoffs held at the full-sample function of c.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy import optimize

from trading_on_disclosure_v4.outcomes import panel as panel_mod
from trading_on_disclosure_v4.outcomes import pooled
from trading_on_disclosure_v4.writing_sample import common as cm
from trading_on_disclosure_v4.writing_sample import fits

OUT = cm.OUT / "figure3"
GRID = np.round(np.linspace(0.0, 1.0, 101), 10)
H = 20


class Paths:
    """Per-event window products: prod(1+r) and the expected-return legs, exactly as forward_cache computes them."""

    def __init__(self, events: pd.DataFrame, calendar: pd.DatetimeIndex, daily: dict, F: np.ndarray, rf: np.ndarray, coefs: pd.DataFrame):
        ev = events.reset_index(drop=True)
        c = ev[["permno", "anchor_idx"]].merge(coefs, on=["permno", "anchor_idx"], how="left")
        self.alpha = c["alpha_daily"].to_numpy(float)
        self.beta = c[fits.BETA_COLS].to_numpy(float)
        idx = ev["anchor_idx"].to_numpy(np.int64)
        self.prod_s = np.full(len(ev), np.nan)
        for perm, rows in ev.groupby("permno").indices.items():
            r = panel_mod.align_to_calendar(daily.get(int(perm), pd.Series(dtype=float)), calendar)
            filled = np.where(np.isfinite(r), r, 0.0)
            p = np.ones(len(rows))
            for k in range(1, H + 1):
                p = p * (1.0 + filled[idx[rows] + k])
            self.prod_s[rows] = p
        F0, rf0 = np.where(np.isfinite(F), F, 0.0), np.where(np.isfinite(rf), rf, 0.0)
        self.rf = np.stack([rf0[idx + k] for k in range(1, H + 1)], axis=1)
        full = np.column_stack([self.alpha, self.beta])  # the cache passes beta[:, 1:] of the (n, 7) coefficient array
        self.bf = np.stack([np.einsum("nj,nj->n", F0[idx + k], full[:, 1:]) for k in range(1, H + 1)], axis=1)

    def z(self, c: float) -> np.ndarray:
        """Z(c) in bp with the cache's operation order: e = rf + c*alpha + beta'f; prod *= 1 + e."""
        prod = np.ones(len(self.alpha))
        a = c * self.alpha
        for k in range(H):
            e = self.rf[:, k] + a + self.bf[:, k]
            prod = prod * (1.0 + e)
        return ((self.prod_s - 1.0) - (prod - 1.0)) * cm.BP


def cutoffs_at(z_pool: np.ndarray, period: np.ndarray) -> dict:
    """Committed-table convention: np.percentile 1/99 by period, stored at %.8g (decimal), returned in bp."""
    out = {}
    for p in ("A", "B"):
        v = z_pool[period == p] / cm.BP
        v = v[np.isfinite(v)]
        out[p] = tuple(float(f"{np.percentile(v, q):.8g}") * cm.BP for q in (1, 99))
    return out


def clip(z: np.ndarray, period: np.ndarray, cuts: dict) -> np.ndarray:
    out = z.copy()
    for p, (lo, hi) in cuts.items():
        m = period == p
        out[m] = np.clip(out[m], lo, hi)
    return out


class Cohort:
    """Household-period sums of the fixed persistence cohort and the covariance under household weights W."""

    def __init__(self, panel: pd.DataFrame):
        self.hh_codes, self.hh = pd.factorize(panel["household_index"], sort=True)
        self.is_a = panel["period"].to_numpy() == "A"
        G = len(self.hh)
        self.n_a = np.bincount(self.hh_codes[self.is_a], minlength=G).astype(float)
        self.n_b = np.bincount(self.hh_codes[~self.is_a], minlength=G).astype(float)
        self.w = np.sqrt(self.n_a * self.n_b)
        self.c = np.log1p(self.n_a)

    def means(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        G = len(self.hh)
        a = np.bincount(self.hh_codes[self.is_a], weights=z[self.is_a], minlength=G) / self.n_a
        b = np.bincount(self.hh_codes[~self.is_a], weights=z[~self.is_a], minlength=G) / self.n_b
        return a, b

    def cov(self, z: np.ndarray, M: np.ndarray | None = None) -> np.ndarray:
        a, b = self.means(z)
        W = (np.ones((1, len(self.hh))) if M is None else np.atleast_2d(M)) * self.w
        W = W / W.sum(axis=1, keepdims=True)
        ma, mb, mc = W @ a, W @ b, W @ self.c
        vc = W @ (self.c * self.c) - mc * mc
        out = W @ (a * b) - ma * mb - (W @ (a * self.c) - ma * mc) * (W @ (b * self.c) - mb * mc) / vc
        out[~(vc > 1e-12)] = np.nan
        return out


def weighted_fe_slope(y: np.ndarray, x: np.ndarray, block: np.ndarray, m: np.ndarray) -> float:
    g = pd.factorize(block)[0]
    sw = np.bincount(g, weights=m)
    keep = sw[g] > 0
    mx = np.bincount(g, weights=m * x) / np.where(sw > 0, sw, 1)
    my = np.bincount(g, weights=m * y) / np.where(sw > 0, sw, 1)
    xd, yd = (x - mx[g])[keep], (y - my[g])[keep]
    sxx = float(np.sum(m[keep] * xd * xd))
    return float(np.sum(m[keep] * xd * yd) / sxx) if sxx > 1e-12 else np.nan


def lambda_hat() -> float:
    t = pd.read_csv(cm.COMPONENTS / "consumer_A1_purchases.csv")
    m = t[(t.analysis == "A1_purchases") & (t.scale == "raw") & (t.term == "sample=cohort;subset=all;regression=Y0_on_x") & (t.statistic == "estimate")]
    if len(m) != 1:
        raise KeyError("raw cohort continuation estimate not found; run the A1_purchases consumer first")
    return float(m.value.iloc[0])


def run_clock(clock: str, cal, daily, F, rf, coefs, lam: float | None, rows: cm.Rows, log) -> dict:
    pp = pd.read_parquet(cm.RESTRICTED / f"purchase_panel_{clock}.parquet")
    coh = pp[pp["sample"] == "cohort"].reset_index(drop=True)
    events, _ = pooled.load_events(clock)
    pool = events.assign(anchor_idx=cal.get_indexer(pd.DatetimeIndex(events["anchor_date"])))
    ok = pool[["permno", "anchor_idx"]].merge(coefs[["permno", "anchor_idx", "reason_h20"]], on=["permno", "anchor_idx"], how="left")["reason_h20"].astype(str).eq("ok").to_numpy()
    pool = pool[ok].reset_index(drop=True)
    P_coh = Paths(coh, cal, daily, F, rf, coefs)
    P_pool = Paths(pool, cal, daily, F, rf, coefs)
    per_coh, per_pool = coh["period"].to_numpy(), pool["period"].to_numpy()
    # exactness at the endpoints
    z0, z1 = P_coh.z(0.0), P_coh.z(1.0)
    chk = {"max_abs_Z0_minus_Y0_bp": float(np.max(np.abs(z0 - coh["Y0_bp"].to_numpy()))), "max_abs_Z1_minus_Y1_bp": float(np.max(np.abs(z1 - coh["Y1_bp"].to_numpy())))}
    from trading_on_disclosure_v4.estimators import event_frames as evf
    for c, outcome in ((0.0, "ff5mom_noalpha_h20"), (1.0, "ff5mom_h20")):
        cut = cutoffs_at(P_pool.z(c), per_pool)
        com = evf.pooled_cutoffs(clock, outcome)
        chk[f"cutoffs_c{c:g}_equal_committed"] = bool(all(np.isclose(cut[p][i], com[p][i], rtol=0, atol=1e-9) for p in ("A", "B") for i in (0, 1)))
    if chk["max_abs_Z0_minus_Y0_bp"] > 1e-9 or chk["max_abs_Z1_minus_Y1_bp"] > 1e-9 or not all(v for k, v in chk.items() if k.startswith("cutoffs")):
        raise AssertionError(f"{clock}: endpoints not reproduced: {chk}")
    log(f"{clock}: Z(0)=Y0 and Z(1)=Y1 on {len(coh)} purchases (max |diff| {max(chk['max_abs_Z0_minus_Y0_bp'], chk['max_abs_Z1_minus_Y1_bp']):.1e} bp); endpoint cutoffs equal the committed table ({len(pool)} pooled purchases)")
    cohort = Cohort(coh)
    mfile = cm.CLOSEOUT / "paired" / f"selection_multiplicities_{clock}.csv"
    mult = pd.read_csv(mfile)
    if mult.columns[1:].tolist() != [f"hh_{i}" for i in cohort.hh]:
        raise AssertionError("selection order differs from the cohort household order")
    M = mult.iloc[:, 1:].to_numpy(float)
    grid = sorted(set(GRID.tolist()) | ({round(lam, 15)} if lam is not None and 0 <= lam <= 1 else set()))
    curve = []
    for c in grid:
        zc, zp = P_coh.z(c), P_pool.z(c)
        cuts = cutoffs_at(zp, per_pool)
        for scale, z in (("raw", zc), ("winsorized", clip(zc, per_coh, cuts))):
            k = float(cohort.cov(z)[0])
            kb = cohort.cov(z, M)
            f = kb[np.isfinite(kb)]
            curve.append({"clock": clock, "scale": scale, "c": c, "is_lambda_hat": bool(lam is not None and abs(c - lam) < 1e-14), "covariance": k,
                          "ci_low": float(np.quantile(f, 0.025)), "ci_high": float(np.quantile(f, 0.975)), "valid_draws": int(len(f)),
                          "cut_A_low": cuts["A"][0], "cut_A_high": cuts["A"][1], "cut_B_low": cuts["B"][0], "cut_B_high": cuts["B"][1]})
    curve = pd.DataFrame(curve)

    def kfun(c, scale):
        zc = P_coh.z(c)
        if scale == "winsorized":
            zc = clip(zc, per_coh, cutoffs_at(P_pool.z(c), per_pool))
        return float(cohort.cov(zc)[0])

    roots = {}
    for scale in ("raw", "winsorized"):
        g = curve[(curve.scale == scale) & curve.c.isin(GRID)].sort_values("c")
        v, cs = g.covariance.to_numpy(), g.c.to_numpy()
        found = []
        for i in range(len(cs) - 1):
            if v[i] == 0:
                found.append(cs[i])
            elif v[i] * v[i + 1] < 0:
                found.append(optimize.brentq(lambda c: kfun(c, scale), cs[i], cs[i + 1], xtol=1e-12))
        roots[scale] = found
    out = {"curve": curve, "roots": roots, "checks": chk, "cohort": coh, "P_coh": P_coh, "P_pool": P_pool, "per_coh": per_coh, "per_pool": per_pool,
           "cohort_obj": cohort, "M": M}
    src = f"data/v4/writing_sample/{cm.RUN_ID}/figure3/fractional_curve.csv"
    for _, r in curve.iterrows():
        for stat in ("covariance", "ci_low", "ci_high"):
            rows.add("figure3_fractional", {"covariance": "estimate"}.get(stat, stat), r[stat], "bp2", clock=clock, scale=r.scale, term="curve", lambda_value=r.c,
                     sample="persistence_cohort", benchmark="fractional_intercept_Z(c)", weighting="sqrt(n_A n_B)", controls="log(1+n_A), both periods",
                     inference="point: fixed events/loadings/weights; band: 4,999 original common household selections, pointwise percentile (not simultaneous)",
                     source=src, result_key=f"figure3_fractional/{clock}/{r.scale}/curve/{stat}/c={r.c!r}")
    for scale, rr in roots.items():
        rows.add("figure3_fractional", "roots_in_unit_interval", len(rr), "count", clock=clock, scale=scale, term="curve", benchmark="fractional_intercept_Z(c)", source=src)
        for j, v in enumerate(rr):
            rows.add("figure3_fractional", f"root_{j + 1}", v, "fraction", clock=clock, scale=scale, term="curve", benchmark="fractional_intercept_Z(c)", source=src,
                     inference="zero crossing of K_Z(c) (brentq on the exact function, cutoffs recomputed at each c)")
    log(f"{clock}: fractional curve K_Z(0)={kfun(0.0, 'winsorized'):.4f}, K_Z(1)={kfun(1.0, 'winsorized'):.4f} (winsorized); roots {roots}")
    return out


def joint_bootstrap(res: dict, cal, lam: float, rows: cm.Rows, log) -> pd.DataFrame:
    """lambda* and K_Z(lambda*) in each of the 4,999 common selections (trade date)."""
    coh, cohort, M = res["cohort"], res["cohort_obj"], res["M"]
    hh = cohort.hh_codes
    y0, x, blk = coh["Y0_bp"].to_numpy(float), coh["alpha20_bp"].to_numpy(float), coh["entry_block"].to_numpy()
    lam0 = weighted_fe_slope(y0, x, blk, np.ones(len(coh)))
    if not np.isclose(lam0, lam, rtol=1e-10, atol=1e-12):
        raise AssertionError(f"lambda reproduction {lam0} vs consumer {lam}")
    recs = []
    t0 = time.time()
    for d in range(len(M)):
        m = M[d][hh]
        lam_d = weighted_fe_slope(y0, x, blk, m)
        rec = {"draw": d, "lambda_star": lam_d, "households_distinct": int((M[d] > 0).sum())}
        if not np.isfinite(lam_d):
            rec.update(K_raw=np.nan, K_winsorized=np.nan, failed="no within-block regressor variation")
        else:
            zc = res["P_coh"].z(lam_d)
            cuts = cutoffs_at(res["P_pool"].z(lam_d), res["per_pool"])
            rec["K_raw"] = float(cohort.cov(zc, M[d])[0])
            rec["K_winsorized"] = float(cohort.cov(clip(zc, res["per_coh"], cuts), M[d])[0])
            rec["outside_unit_interval"] = bool(lam_d < 0 or lam_d > 1)
            rec["failed"] = "" if np.isfinite(rec["K_raw"]) and np.isfinite(rec["K_winsorized"]) else "covariance not finite"
        recs.append(rec)
        if (d + 1) % 1000 == 0:
            log(f"joint bootstrap {d + 1}/{len(M)} draws, {time.time() - t0:.0f} s")
    jb = pd.DataFrame(recs)
    src = f"data/v4/writing_sample/{cm.RUN_ID}/figure3/joint_bootstrap_draws.csv"
    lam_ok = jb["lambda_star"].dropna().to_numpy()
    base = dict(clock="T", sample="persistence_cohort", benchmark="fractional_intercept_Z(lambda)", weighting="sqrt(n_A n_B)", controls="log(1+n_A), both periods", source=src)
    rows.add("figure3_plugin", "lambda_hat", lam, "slope", term="plugin", inference="raw cohort continuation (A1_purchases, entry-block FE)", **base)
    for s, v in (("lambda_star_p2.5", np.percentile(lam_ok, 2.5)), ("lambda_star_p97.5", np.percentile(lam_ok, 97.5)), ("lambda_star_sd", lam_ok.std(ddof=1)),
                 ("lambda_star_share_outside_unit_interval", float(np.mean((lam_ok < 0) | (lam_ok > 1))))):
        rows.add("figure3_plugin", s, v, "slope" if "share" not in s else "share", term="plugin", inference="household bootstrap of lambda (4,999 common selections); conditional on fitted factor models", **base)
    for scale in ("raw", "winsorized"):
        f = jb[f"K_{scale}"].to_numpy(float)
        f = f[np.isfinite(f)]
        lo, hi = np.quantile(f, [0.025, 0.975])
        b = dict(base, scale=scale)
        rows.add("figure3_plugin", "joint_ci_low", lo, "bp2", term="plugin", inference="joint household bootstrap: lambda* and K_Z(lambda*) re-estimated in each common selection; activity refit; cutoffs fixed functions of c from the full sample; percentile", **b)
        rows.add("figure3_plugin", "joint_ci_high", hi, "bp2", term="plugin", inference="joint household bootstrap percentile", **b)
        rows.add("figure3_plugin", "joint_valid_draws", len(f), "count", term="plugin", **b)
        rows.add("figure3_plugin", "joint_failed_draws", int(len(jb) - len(f)), "count", term="plugin", inference="draws without a finite lambda* or covariance (dropped from the percentile)", **b)
    log(f"joint bootstrap: lambda* 95% range [{np.percentile(lam_ok, 2.5):.4f}, {np.percentile(lam_ok, 97.5):.4f}]; {int(jb['failed'].astype(bool).sum())} failed draws")
    return jb


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    log = cm.logger("fractional")
    t0 = time.time()
    cal = panel_mod.v2_trading_calendar()
    factors = panel_mod.load_factors()
    F = factors[list(panel_mod.FACTOR_COLS)].reindex(cal).to_numpy(float)
    rf = factors["rf"].reindex(cal).to_numpy(float)
    coefs = fits.load(["permno", "anchor_idx", "alpha_daily", *fits.BETA_COLS, "reason_h20"])
    coefs["reason_h20"] = coefs["reason_h20"].astype(str)
    d = panel_mod.load_local_daily()
    daily = {int(p): g.set_index("date")["total_ret"] for p, g in d.groupby("permno")}
    del d
    lam = lambda_hat()
    rows = cm.Rows(f"data/v4/writing_sample/{cm.RUN_ID}/figure3/")
    res = {c: run_clock(c, cal, daily, F, rf, coefs, lam if c == "T" else None, rows, log) for c in ("T", "P")}
    curve = pd.concat([r["curve"] for r in res.values()], ignore_index=True)
    cm.write_csv(curve, OUT / "fractional_curve.csv")
    plug = curve[(curve.clock == "T") & curve.is_lambda_hat]
    for _, r in plug.iterrows():
        b = dict(clock="T", scale=r.scale, sample="persistence_cohort", benchmark="fractional_intercept_Z(lambda)", weighting="sqrt(n_A n_B)", controls="log(1+n_A), both periods",
                 source=f"data/v4/writing_sample/{cm.RUN_ID}/figure3/fractional_curve.csv")
        rows.add("figure3_plugin", "K_at_lambda_hat", r.covariance, "bp2", term="plugin", inference="covariance of Z(lambda_hat) on the fixed cohort", **b)
        rows.add("figure3_plugin", "fixed_c_band_low_at_lambda_hat", r.ci_low, "bp2", term="plugin", inference="pointwise band at fixed c = lambda_hat; ignores estimation of lambda", **b)
        rows.add("figure3_plugin", "fixed_c_band_high_at_lambda_hat", r.ci_high, "bp2", term="plugin", inference="pointwise band at fixed c = lambda_hat; ignores estimation of lambda", **b)
    jb = joint_bootstrap(res["T"], cal, lam, rows, log)
    cm.write_csv(jb, OUT / "joint_bootstrap_draws.csv")
    checks = {c: r["checks"] for c, r in res.items()}
    roots = {c: r["roots"] for c, r in res.items()}
    cm.write_json({"checks": checks, "roots": roots, "lambda_hat": lam, "grid": "0, 0.01, ..., 1 plus lambda_hat (trade date)", "seconds": round(time.time() - t0, 1),
                   "selections": "data/v4/round2/closeout/closeout_2026-11-10/paired/selection_multiplicities_<clock>.csv (4,999 original common household selections)",
                   "environment": cm.environment()}, OUT / "fractional_manifest.json")
    rows.write_component("figure3_fractional")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
