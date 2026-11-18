"""TOST equivalence testing with SESOI provenance (EQV-001..004).

A practical-equivalence CLAIM requires an approved bound; with a bound still
``PENDING_HUMAN_SIGNOFF`` (HS-03) every decision is emitted ``advisory=True``
and may not support manuscript language.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scipy import stats

DECISION_EQUIVALENT = "evidence_of_practical_equivalence"
DECISION_OVERLAPS = "economically_meaningful_effects_remain_plausible"
DECISION_SMALL = "statistically_nonzero_but_economically_small"
DECISION_MEANINGFUL = "potentially_economically_meaningful"


@dataclass(frozen=True)
class SESOIBound:
    value: float
    unit: str
    rationale: str | None = None
    approval_status: str = "PENDING_HUMAN_SIGNOFF"
    signatory: str | None = None
    freeze_hash: str | None = None

    @property
    def approved(self) -> bool:
        return self.approval_status == "APPROVED"


@dataclass
class EquivalenceResult:
    estimate: float
    se: float
    bound: SESOIBound
    p_lower: float
    p_upper: float
    tost_reject: bool
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    significant_vs_zero: bool
    decision: str
    advisory: bool
    notes: dict = field(default_factory=dict)


def tost_equivalence(
    estimate: float,
    se: float,
    bound: SESOIBound,
    df: int | None = None,
    alpha: float = 0.05,
) -> EquivalenceResult:
    """Two one-sided tests against the symmetric interval [-Delta, +Delta].

    Equivalence needs BOTH one-sided rejections; an ordinary 95% CI
    containing zero is never itself evidence of equivalence (EQV-004).
    """
    if se <= 0:
        raise ValueError("se must be positive")
    delta = abs(bound.value)
    dist = stats.t(df) if df is not None else stats.norm()
    t_lower = (estimate + delta) / se
    t_upper = (estimate - delta) / se
    p_lower = float(1.0 - dist.cdf(t_lower))
    p_upper = float(dist.cdf(t_upper))
    tost_reject = p_lower < alpha and p_upper < alpha

    q90 = float(dist.ppf(1.0 - alpha))
    q95 = float(dist.ppf(1.0 - alpha / 2.0))
    ci90 = (estimate - q90 * se, estimate + q90 * se)
    ci95 = (estimate - q95 * se, estimate + q95 * se)
    significant = ci95[0] > 0 or ci95[1] < 0
    inside = -delta < ci90[0] and ci90[1] < delta

    if tost_reject and inside and not significant:
        decision = DECISION_EQUIVALENT
    elif significant and inside and tost_reject:
        decision = DECISION_SMALL
    elif significant and not inside:
        decision = DECISION_MEANINGFUL
    elif significant:
        decision = DECISION_SMALL if tost_reject else DECISION_MEANINGFUL
    else:
        decision = DECISION_EQUIVALENT if tost_reject else DECISION_OVERLAPS

    return EquivalenceResult(
        estimate=estimate,
        se=se,
        bound=bound,
        p_lower=p_lower,
        p_upper=p_upper,
        tost_reject=tost_reject,
        ci90=ci90,
        ci95=ci95,
        significant_vs_zero=significant,
        decision=decision,
        advisory=not bound.approved,
        notes={
            "ci90_within_bound": inside,
            "alpha": alpha,
            "distribution": "t" if df is not None else "normal",
        },
    )
