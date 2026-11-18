"""Small-cluster inference for one WLS coefficient: CR2 with Imbens-Kolesar and Bell-McCaffrey dof, and the
Carter-Schnepel-Steigerwald effective number of clusters.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats

WORKING_MODELS = ("inverse_weights", "identity")
PAPER_RHO = 1.0
CLUSTEFF_RHO = 0.9999
REPORTED_RHOS = (0.0, CLUSTEFF_RHO, PAPER_RHO)

_CR2_DESCRIPTIONS = {
    "inverse_weights": (
        "WLS treated as OLS on sqrt(v)-transformed rows (y*, X*) = (sqrt(v) y, sqrt(v) X). CR2 = LZ2 of the "
        "transformed rows, A_s = (I - P*_ss)^(+1/2) with P* the hat matrix of X* (BM 2002 eq. 7 with U = I; "
        "IK 2016 sec. 4; equals PT 2018 eqs. 4, 7, 8 with W = V and Phi = V^-1 when v is constant within "
        "clusters). BM dof: (sum lambda)^2 / sum lambda^2 over eigenvalues of G'G, identity working covariance "
        "for the transformed errors. IK dof: eigenvalues of G' Omega G with Omega_ii = sigma2_hat and "
        "Omega_ij = rho_hat for i != j in the same cluster, estimated from the transformed residuals "
        "e*_i = sqrt(v_i) e_i: sigma2_hat = sum e*^2 / N, rho_hat = (sum_s (sum_{i in s} e*_i)^2 - sum e*^2) / "
        "(sum_s N_s^2 - N) (IK 2016 p. 21); idiosyncratic part max(sigma2_hat - rho_hat, 0) as in dfadjust; "
        "rho_hat floored at 0 only when ik_rho_nonnegative (dfadjust rho0). No multiplicative small-sample factor "
        "on CR2; t quantiles with real-valued dof."
    ),
    "identity": (
        "WLS with weights W = V and working covariance Phi = I for the original-scale errors of y. CR2 per "
        "PT 2018 eqs. 4, 7, 8: A_s = [(I - H)_s (I - H)_s']^(+1/2), H = X (X'VX)^-1 X'V. BM dof: PT 2018 "
        "eq. 11 with Phi = I. IK dof: PT 2018 eq. 11 with Phi replaced by the IK random-effects Omega_hat "
        "estimated, with the IK 2016 p. 21 estimators, from the original-scale residuals e; idiosyncratic part "
        "max(sigma2_hat - rho_hat, 0); rho_hat floored at 0 only when ik_rho_nonnegative. No multiplicative "
        "small-sample factor on CR2; t quantiles with real-valued dof."
    ),
}

_CSS_DESCRIPTIONS = {
    "inverse_weights": (
        "gamma_g = a'(X*'X*)^-1 X*_g' Omega_g X*_g (X*'X*)^-1 a on the sqrt(v)-transformed rows X* = sqrt(v) X "
        "(CSS 2017 eq. 5 and LS 2018 p. 448 applied to the transformed OLS regression, i.e. clusteff run on "
        "sqrt(v)-scaled regressors with noconstant), Omega_g = (1 - rho) I + rho 11' (clusteff covariance(rho)), "
        "a = e_k; Gamma = G^-1 sum_g ((gamma_g - gamma_bar) / gamma_bar)^2, G* = G / (1 + Gamma). The paper's "
        "feasible G*_A uses rho = 1 (matrix of ones); clusteff's documented default covariance(1) is run as "
        "rho = 0.9999; LS 2018 p. 456 gives rho = 0 (identity) as the less conservative variant."
    ),
    "identity": (
        "gamma_g = a'(X'VX)^-1 X_g' V_g Omega_g V_g X_g (X'VX)^-1 a, the CSS 2017 eq. 5 contribution of cluster "
        "g to Var(a' beta_hat) for the WLS estimator, with Omega_g = (1 - rho) I + rho 11' for the original-scale "
        "errors of y, a = e_k; Gamma and G* as in CSS 2017 sec. 3. rho = 1 is the paper's G*_A, 0.9999 the "
        "clusteff run value, 0 the LS 2018 p. 456 identity variant."
    ),
}


def _inputs(X, v, cluster, y=None):
    """Validate shapes, finiteness and positive weights; return float arrays, cluster codes and labels."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    n, k = X.shape
    v = np.asarray(v, dtype=float).reshape(-1)
    cl = np.asarray(cluster).reshape(-1)
    if len(v) != n or len(cl) != n:
        raise ValueError("X, v and cluster must have the same number of rows")
    if y is not None:
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != n:
            raise ValueError("y and X must have the same number of rows")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite")
    if not (np.all(np.isfinite(v)) and np.all(v > 0)):
        raise ValueError("weights v must be finite and strictly positive")
    if n <= k:
        raise ValueError("need more observations than regressors")
    labels, codes = np.unique(cl, return_inverse=True)
    if len(labels) < 2:
        raise ValueError("need at least two clusters")
    return y, X, v, codes.reshape(-1), labels


def _check_rank(Xt):
    """Raise instead of regularizing when the weighted design is rank deficient."""
    rank = int(np.linalg.matrix_rank(Xt))
    if rank < Xt.shape[1]:
        raise ValueError(
            f"design matrix is rank deficient (rank {rank} < {Xt.shape[1]} columns); drop collinear regressors"
        )


def _check_working(working):
    """Reject unknown working-model names."""
    if working not in WORKING_MODELS:
        raise ValueError(f"working must be one of {WORKING_MODELS}, got {working!r}")


def _groups(codes, n_clusters):
    """Row indices of each cluster in cluster-code order."""
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes, minlength=n_clusters))[:-1])


def _inv_sqrt_apply(Z, C, z, tol):
    """Apply (I + Z C Z')^(+1/2), the symmetric root of the Moore-Penrose inverse, to z; count dropped directions."""
    U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    if sv.size == 0 or sv[0] == 0.0:
        return z.copy(), 0
    keep = sv > sv[0] * 1e-12
    U, sv, Vt = U[:, keep], sv[keep], Vt[keep]
    T = (sv[:, None] * Vt) @ C @ (Vt.T * sv[None, :])
    lam, E = np.linalg.eigh(0.5 * (T + T.T))
    U = U @ E
    b = 1.0 + lam
    cut = tol * max(1.0, float(b.max()))
    f = np.where(b > cut, 1.0 / np.sqrt(np.maximum(b, cut)), 0.0)
    return z + U @ ((f - 1.0) * (U.T @ z)), int(np.sum(b <= cut))


def _satterthwaite(mat):
    """(sum of eigenvalues)^2 / sum of squared eigenvalues of a symmetric matrix."""
    ss = float(np.sum(mat * mat))
    return float(np.trace(mat)) ** 2 / ss if ss > 0 else np.nan


def _cr2_core(y, X, w, groups, ell, tol):
    """CR2 with identity working covariance in the scale of (y, X) and weights w; S x S matrices for the dof."""
    n, k = X.shape
    sw = np.sqrt(w)
    Xt = X * sw[:, None]
    _check_rank(Xt)
    Q, R = np.linalg.qr(Xt)
    beta = linalg.solve_triangular(R, Q.T @ (sw * y))
    e = y - X @ beta
    Rinv = linalg.solve_triangular(R, np.eye(k))
    M = Rinv @ Rinv.T
    m = M @ ell
    K = M @ ((X * (w * w)[:, None]).T @ X) @ M
    C_general = np.block([[K, -M], [-M, np.zeros((k, k))]])
    S = len(groups)
    se_term, cr0_term, qq, d = (np.zeros(S) for _ in range(4))
    sum_e, sum_e2, lev = (np.zeros(S) for _ in range(3))
    A, B, F = (np.zeros((S, k)) for _ in range(3))
    sizes = np.array([len(idx) for idx in groups], dtype=float)
    n_pinv = 0
    for s, idx in enumerate(groups):
        Xs, ws, es, Qs = X[idx], w[idx], e[idx], Q[idx]
        lev[s] = float(np.linalg.eigvalsh(Qs.T @ Qs).max())
        if ws.max() - ws.min() <= 1e-12 * ws.max():
            Z, C = Xs, K - 2.0 * float(ws.mean()) * M
        else:
            Z, C = np.hstack([Xs, Xs * ws[:, None]]), C_general
        z = ws * (Xs @ m)
        q, dropped = _inv_sqrt_apply(Z, C, z, tol)
        n_pinv += dropped > 0
        se_term[s] = q @ es
        cr0_term[s] = z @ es
        qq[s] = q @ q
        A[s] = Xs.T @ q
        B[s] = Xs.T @ (ws * q)
        d[s] = q.sum()
        F[s] = Xs.T @ ws
        sum_e[s] = es.sum()
        sum_e2[s] = es @ es
    P = np.diag(qq) - B @ M @ A.T - A @ M @ B.T + A @ K @ A.T
    T = np.diag(d) - F @ M @ A.T
    sigma2 = float(sum_e2.sum() / n)
    den = float(np.sum(sizes**2) - n)
    rho = float((np.sum(sum_e**2) - sum_e2.sum()) / den) if den > 0 else 0.0
    return {
        "beta": beta, "M": M, "K": K, "P": P, "PJ": T.T @ T, "se_term": se_term, "cr0_term": cr0_term,
        "sigma2": sigma2, "rho": rho, "leverage": lev, "sizes": sizes, "n_pinv": int(n_pinv),
    }


def _ik_omega(sigma2, rho, n_max, rho_nonnegative):
    """IK (2016) random-effects working covariance as sigma2_eps * I + rho * (same-cluster ones), dfadjust truncations."""
    rho_used = max(rho, 0.0) if rho_nonnegative else rho
    sigma2_eps = max(sigma2 - rho_used, 0.0)
    return rho_used, sigma2_eps, bool(sigma2_eps + rho_used * n_max >= 0)


def _t_interval(est, se, dof, level):
    """Two-sided t critical value, confidence limits and p-value for est/se with real-valued dof."""
    if not (np.isfinite(dof) and dof > 0 and se > 0):
        return np.nan, np.nan, np.nan, np.nan
    crit = float(stats.t.ppf(0.5 + level / 2.0, df=dof))
    p = float(2.0 * stats.t.sf(abs(est / se), df=dof))
    return crit, est - crit * se, est + crit * se, p


def cr2_inference(
    y, X, v, cluster, coef_index=1, level=0.95, *, working="inverse_weights", ik_rho_nonnegative=False, tol=1e-9
):
    """CR2 (BM 2002) clustered SE for one WLS coefficient with IK (2016) and BM dof and the project CGM CR1 SE."""
    _check_working(working)
    y, X, v, codes, labels = _inputs(X, v, cluster, y)
    n, k = X.shape
    if not 0 <= coef_index < k:
        raise ValueError(f"coef_index {coef_index} out of range for {k} columns")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    G = len(labels)
    groups = _groups(codes, G)
    ell = np.zeros(k)
    ell[coef_index] = 1.0
    if working == "inverse_weights":
        sw = np.sqrt(v)
        core = _cr2_core(sw * y, X * sw[:, None], np.ones(n), groups, ell, tol)
    else:
        core = _cr2_core(y, X, v, groups, ell, tol)
    est = float(core["beta"][coef_index])
    var_cr0 = float(np.sum(core["cr0_term"] ** 2))
    cgm_factor = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    se_cr1 = float(np.sqrt(cgm_factor * var_cr0))
    se_cr2 = float(np.sqrt(np.sum(core["se_term"] ** 2)))
    dof_bm = _satterthwaite(core["P"])
    rho_used, sigma2_eps, omega_psd = _ik_omega(core["sigma2"], core["rho"], core["sizes"].max(), ik_rho_nonnegative)
    dof_ik = _satterthwaite(sigma2_eps * core["P"] + rho_used * core["PJ"])
    crit_ik, lo_ik, hi_ik, p_ik = _t_interval(est, se_cr2, dof_ik, level)
    crit_bm, lo_bm, hi_bm, p_bm = _t_interval(est, se_cr2, dof_bm, level)
    crit_cgm, lo_cgm, hi_cgm, p_cgm = _t_interval(est, se_cr1, float(G - 1), level)
    return {
        "estimate": est,
        "se_cr2": se_cr2,
        "se_cr1_cgm": se_cr1,
        "se_cr0": float(np.sqrt(var_cr0)),
        "t_stat_cr2": est / se_cr2 if se_cr2 > 0 else np.nan,
        "dof_ik": dof_ik,
        "dof_bm": dof_bm,
        "dof_cgm": G - 1,
        "crit_ik": crit_ik,
        "ci_low_ik": lo_ik,
        "ci_high_ik": hi_ik,
        "p_value_ik": p_ik,
        "crit_bm": crit_bm,
        "ci_low_bm": lo_bm,
        "ci_high_bm": hi_bm,
        "p_value_bm": p_bm,
        "crit_cgm": crit_cgm,
        "ci_low_cgm": lo_cgm,
        "ci_high_cgm": hi_cgm,
        "p_value_cgm": p_cgm,
        "cgm_factor": cgm_factor,
        "rho_hat": core["rho"],
        "sigma2_hat": core["sigma2"],
        "icc_hat": core["rho"] / core["sigma2"] if core["sigma2"] > 0 else np.nan,
        "ik_rho_used": rho_used,
        "ik_sigma2_eps": sigma2_eps,
        "ik_truncated": bool(core["sigma2"] - rho_used < 0),
        "ik_rho_nonnegative": bool(ik_rho_nonnegative),
        "ik_omega_psd": omega_psd,
        "n_obs": n,
        "n_clusters": G,
        "n_params": k,
        "coef_index": coef_index,
        "level": level,
        "cluster_size_min": int(core["sizes"].min()),
        "cluster_size_max": int(core["sizes"].max()),
        "max_cluster_leverage": float(core["leverage"].max()),
        "n_clusters_pinv": core["n_pinv"],
        "working": working,
        "working_model_description": _CR2_DESCRIPTIONS[working],
    }


def _gamma(sum_b, sum_b2, rho):
    """Per-cluster gamma_g under Omega_g = (1 - rho) I + rho 11', from within-cluster sums of b and b^2."""
    return (1.0 - rho) * sum_b2 + rho * sum_b**2


def _gstar(gamma):
    """CSS (2017) Gamma (squared coefficient of variation, 1/G normalization) and G* = G / (1 + Gamma)."""
    gbar = float(gamma.mean())
    if not gbar > 0:
        raise ValueError("all gamma_g are zero: the tested coefficient does not load on any cluster")
    Gamma = float(np.mean(((gamma - gbar) / gbar) ** 2))
    return Gamma, len(gamma) / (1.0 + Gamma)


def css_effective_clusters(X, v, cluster, coef_index=1, rho=PAPER_RHO, *, working="inverse_weights"):
    """CSS (2017) / clusteff effective number of clusters G* for one WLS coefficient, with results for rho in {0, 0.9999, 1}."""
    _check_working(working)
    _, X, v, codes, labels = _inputs(X, v, cluster)
    n, k = X.shape
    if not 0 <= coef_index < k:
        raise ValueError(f"coef_index {coef_index} out of range for {k} columns")
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    G = len(labels)
    sw = np.sqrt(v)
    Xt = X * sw[:, None]
    _check_rank(Xt)
    Q, R = np.linalg.qr(Xt)
    a = np.zeros(k)
    a[coef_index] = 1.0
    Ma = linalg.solve_triangular(R, linalg.solve_triangular(R, a, trans="T"))
    b = (sw if working == "inverse_weights" else v) * (X @ Ma)
    sum_b = np.bincount(codes, weights=b, minlength=G)
    sum_b2 = np.bincount(codes, weights=b * b, minlength=G)
    gamma = _gamma(sum_b, sum_b2, rho)
    Gamma, G_star = _gstar(gamma)
    by_rho = {}
    for r in sorted(set(REPORTED_RHOS) | {rho}):
        Gm, Gs = _gstar(_gamma(sum_b, sum_b2, r))
        by_rho[r] = {"G_star": Gs, "Gamma": Gm}
    return {
        "G_star": G_star,
        "Gamma": Gamma,
        "gamma": gamma,
        "gamma_share": gamma / gamma.sum(),
        "gamma_bar": float(gamma.mean()),
        "max_gamma_share": float(gamma.max() / gamma.sum()),
        "clusters": labels,
        "n_clusters": G,
        "n_obs": n,
        "rho": rho,
        "by_rho": by_rho,
        "documented_default_rho": PAPER_RHO,
        "clusteff_run_rho": CLUSTEFF_RHO,
        "coef_index": coef_index,
        "working": working,
        "assumption": _CSS_DESCRIPTIONS[working],
    }
