"""Continuation into the second period regressed on first-period performance, given activity.

The persistence estimate is conditional on trading in both periods, so the obvious worry is selection on
performance: households that did well in 2015-2020 may be the ones that keep trading in 2021-2025. This
module takes the population BEFORE the second-period condition — every household meeting the first-period
half of the persistence rule, i.e. every household with a first-period mean — and regresses two
continuation indicators on that mean and on log(1 + n_hA):

- `any_purchase`: the household makes at least one purchase in the second period;
- `eligibility_rule`: the household also meets the second-period half of the rule (the estimate's sample).

One row per household, so the standard errors are heteroskedasticity-robust (HC1), not clustered. The
first-period mean enters per 100 bp so the coefficient is a probability change per percentage point of
first-period abnormal return. Both a linear probability model and a logit are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.eligibility import RULES
from .specs import period_activity

OUTCOMES = ("any_purchase", "eligibility_rule")
#: the first-period mean is expressed per 100 bp (one percentage point)
MEAN_SCALE = 100.0
REGRESSORS = ("mean_a", "log_activity_a")


def continuation_table(panel: pd.DataFrame, rule: str = "R1", y: str = "y_h20",
                       household: str = "household_id", period: str = "period") -> pd.DataFrame:
    """One row per household meeting the first-period half of `rule`: its first-period mean and count,
    and the two continuation indicators. Household ids stay the index and are never written out."""
    na, qa, sa, nb, qb = RULES[rule]
    ab = panel.loc[panel[period].isin(["A", "B"])]
    act = period_activity(ab)
    a = act.xs("A", level="period") if "A" in act.index.get_level_values("period") else act.iloc[:0].droplevel("period")
    ok_a = a.index[(a["n"] >= na) & (a["q"] >= qa) & (a["s"] >= sa)]
    if len(ok_a) == 0:
        raise ValueError(f"no household meets the first-period half of rule {rule!r}")
    b = (act.xs("B", level="period") if "B" in act.index.get_level_values("period")
         else act.iloc[:0].droplevel("period")).reindex(ok_a).fillna(0)
    first = ab.loc[(ab[period] == "A") & ab[household].isin(ok_a)].groupby(household)[y].agg(["count", "mean"]).reindex(ok_a)
    return pd.DataFrame({
        "mean_a": first["mean"].to_numpy(dtype=float) / MEAN_SCALE,
        "n_a": first["count"].to_numpy(dtype=float),
        "log_activity_a": np.log1p(first["count"].to_numpy(dtype=float)),
        "any_purchase": (b["n"].to_numpy(dtype=float) > 0).astype(float),
        "eligibility_rule": ((b["n"].to_numpy(dtype=float) >= nb) & (b["q"].to_numpy(dtype=float) >= qb)).astype(float),
    }, index=ok_a)


def _design(table: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(table))] + [table[r].to_numpy(dtype=float) for r in REGRESSORS])


def ols_hc1(table: pd.DataFrame, outcome: str) -> dict:
    """Linear probability model with HC1 standard errors."""
    X = _design(table)
    yv = table[outcome].to_numpy(dtype=float)
    n, k = X.shape
    if n <= k:
        raise ValueError(f"need more than {k} households for the continuation regression")
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ yv
    u = yv - X @ beta
    V = bread @ ((X * u[:, None]).T @ (X * u[:, None])) @ bread * n / (n - k)
    se = np.sqrt(np.diag(V))
    ss = float(np.sum((yv - yv.mean()) ** 2))
    out = {"n": int(n), "share": float(yv.mean()), "r2": 1.0 - float(u @ u) / ss if ss > 0 else np.nan, "converged": True}
    for i, name in enumerate(("intercept", *REGRESSORS)):
        out[f"coef_{name}"], out[f"se_{name}"] = float(beta[i]), float(se[i])
    return out


def logit(table: pd.DataFrame, outcome: str) -> dict:
    """Logit with HC1 standard errors (statsmodels); `converged` travels with the coefficients."""
    import statsmodels.api as sm

    X = _design(table)
    yv = table[outcome].to_numpy(dtype=float)
    if len(np.unique(yv)) < 2:
        raise ValueError(f"the {outcome!r} indicator does not vary")
    fit = sm.Logit(yv, X).fit(disp=0, method="newton", maxiter=200, cov_type="HC1")
    beta, se = np.asarray(fit.params, dtype=float), np.asarray(fit.bse, dtype=float)
    out = {"n": int(len(yv)), "share": float(yv.mean()), "r2": float(fit.prsquared), "converged": bool(fit.mle_retvals.get("converged", False))}
    for i, name in enumerate(("intercept", *REGRESSORS)):
        out[f"coef_{name}"], out[f"se_{name}"] = float(beta[i]), float(se[i])
    return out
