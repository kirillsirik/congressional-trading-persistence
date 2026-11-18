"""A5: normal-normal empirical Bayes with prior mean linear in log(1+n_h) (plan §7.3).

Ybar_h ~ N(theta_h, s_h^2), theta_h ~ N(a + b log(1+n_h), tau_EB^2). (a, b, tau_EB^2) by marginal
maximum likelihood with tau_EB^2 >= 0 (zero-boundary flagged). Posterior means are never standardized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class EBFit:
    a: float
    b: float
    tau2: float
    tau: float
    at_zero_boundary: bool
    loglik: float
    n: int
    prior_mean: np.ndarray
    shrinkage: np.ndarray
    posterior_mean: np.ndarray
    posterior_var: np.ndarray


def _design(n_h: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(n_h)), np.log1p(np.asarray(n_h, dtype=float))])


def _profile(tau2: float, y: np.ndarray, Z: np.ndarray, s2: np.ndarray) -> tuple[float, np.ndarray]:
    v = tau2 + s2
    w = 1.0 / v
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Z * sw[:, None], y * sw, rcond=None)
    r = y - Z @ beta
    ll = -0.5 * float(np.sum(np.log(2 * np.pi * v) + r * r / v))
    return ll, beta


def fit_eb(ybar: np.ndarray, se: np.ndarray, n_h: np.ndarray, boundary_tol: float = 1e-10) -> EBFit:
    """Marginal ML over tau2 in [0, upper]; upper = 10 x the total variance of ybar (fallback 1)."""
    y = np.asarray(ybar, dtype=float)
    s2 = np.asarray(se, dtype=float) ** 2
    Z = _design(n_h)
    if len(y) < 3:
        raise ValueError("need at least three households")
    upper = max(10.0 * float(np.var(y)), 1.0)
    obj = lambda t2: -_profile(t2, y, Z, s2)[0]  # noqa: E731
    res = optimize.minimize_scalar(obj, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-10})
    tau2 = float(res.x)
    if obj(0.0) <= obj(tau2):
        tau2 = 0.0
    ll, beta = _profile(tau2, y, Z, s2)
    mu = Z @ beta
    shrink = tau2 / (tau2 + s2) if tau2 > 0 else np.zeros_like(s2)
    post = mu + shrink * (y - mu)
    return EBFit(
        a=float(beta[0]), b=float(beta[1]), tau2=tau2, tau=float(np.sqrt(tau2)),
        at_zero_boundary=bool(tau2 <= boundary_tol), loglik=ll, n=int(len(y)),
        prior_mean=mu, shrinkage=shrink, posterior_mean=post, posterior_var=shrink * s2,
    )


def posterior_mean(ybar: np.ndarray, se: np.ndarray, prior_mean: np.ndarray, tau2: float) -> np.ndarray:
    """Posterior mean for given hyperparameters (used for the limit tests and post-tag placement)."""
    s2 = np.asarray(se, dtype=float) ** 2
    if tau2 <= 0:
        return np.asarray(prior_mean, dtype=float).copy()
    shrink = tau2 / (tau2 + s2)
    return prior_mean + shrink * (np.asarray(ybar, dtype=float) - prior_mean)


def ols_hc1(x: np.ndarray, y: np.ndarray) -> dict:
    """y = c + slope*x with HC1 standard errors (household-level regressions, one row per household)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        raise ValueError("need at least three observations")
    X = np.column_stack([np.ones(n), x])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    u = y - X @ beta
    meat = (X * u[:, None]).T @ (X * u[:, None])
    V = bread @ meat @ bread * n / (n - 2)
    se = np.sqrt(np.diag(V))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(u * u)) / ss_tot if ss_tot > 0 else np.nan
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "se_intercept": float(se[0]),
            "se_slope": float(se[1]), "t_slope": float(beta[1] / se[1]) if se[1] > 0 else np.nan, "r2": r2, "n": n}


def forecast_regression(posterior_mean_a: np.ndarray, ybar_b: np.ndarray) -> dict:
    """Ybar_hB on the UNSTANDARDIZED period-A posterior mean; slope 1 = calibrated shrinkage."""
    out = ols_hc1(posterior_mean_a, ybar_b)
    out["regressor"] = "posterior_mean_A_unstandardized"
    return out


def reliability_regression(ybar_a: np.ndarray, ybar_b: np.ndarray) -> dict:
    """Ybar_hB on raw Ybar_hA (used when tau_EB = 0 and the posterior means are degenerate)."""
    out = ols_hc1(ybar_a, ybar_b)
    out["regressor"] = "ybar_A_raw"
    return out
