"""Claim ledger and language gates (R08, RPT-003..008, JC-002).

Every empirical prose claim maps to registered results; the language class a
claim may use is a function of evidence tier and equivalence status, never of
convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_on_disclosure_v2.inference.event_bands import GaussianShortcutError


class ClaimError(RuntimeError):
    pass


#: Terms Tier H (or D) evidence may never sit next to (language linter,
#: 03_NO_SNOOPING §12).
FORBIDDEN_NEAR_TIER_H = (
    "confirms",
    "validates out of sample",
    "prospective",
    "untouched holdout",
    "preregistered confirmation",
    "definitive null",
)

ALLOWED_CLAIM_CLASSES = (
    "predictability",
    "practical_equivalence",
    "inconclusive",
    "descriptive",
    "diagnostic",
)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text_fragment: str
    supporting_test_ids: tuple[str, ...]
    evidence_tier: str  # tier the PROSE asserts
    claim_class: str
    path_wide: bool = False
    band_type: str | None = None  # sup_t | pointwise | None
    null_claim: bool = False
    equivalence_test_id: str | None = None
    copy_trading_claim: bool = False
    supporting_positions: tuple[str, ...] = ()
    required_caveat: str | None = None


@dataclass
class ClaimLedger:
    claims: dict[str, Claim] = field(default_factory=dict)

    def add(self, claim: Claim, registry: pd.DataFrame) -> None:
        validate_claim(claim, registry)
        if claim.claim_id in self.claims:
            raise ClaimError(f"duplicate claim id {claim.claim_id!r}")
        self.claims[claim.claim_id] = claim

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.claims.values()])


def validate_claim(claim: Claim, registry: pd.DataFrame) -> None:
    if claim.claim_class not in ALLOWED_CLAIM_CLASSES:
        raise ClaimError(f"{claim.claim_id}: unknown claim class {claim.claim_class!r}")

    if not claim.supporting_test_ids and claim.claim_class != "descriptive":
        raise ClaimError(f"{claim.claim_id}: no supporting registered results")

    known = set(registry["test_id"].astype(str))
    unknown = [t for t in claim.supporting_test_ids if t not in known]
    if unknown:
        raise ClaimError(
            f"{claim.claim_id}: cites unregistered results {unknown} (RPT-002)"
        )

    # Tier consistency: the prose may not assert a stronger tier than every
    # supporting result actually carries (JC-002; no tier laundering).
    if claim.supporting_test_ids:
        tiers = set(
            registry.set_index("test_id")
            .loc[list(claim.supporting_test_ids), "evidence_tier"]
            .astype(str)
        )
        if claim.evidence_tier not in tiers:
            raise ClaimError(
                f"{claim.claim_id}: asserts tier {claim.evidence_tier!r} but "
                f"supporting results carry {sorted(tiers)} (RPT-007)"
            )

    if claim.evidence_tier in {"H", "D"}:
        lowered = claim.text_fragment.lower()
        hits = [t for t in FORBIDDEN_NEAR_TIER_H if t in lowered]
        if hits:
            raise ClaimError(
                f"{claim.claim_id}: Tier {claim.evidence_tier} prose uses "
                f"forbidden terms {hits} (RPT-007)"
            )

    if claim.path_wide and claim.band_type != "sup_t":
        raise GaussianShortcutError(
            f"{claim.claim_id}: path-wide claim from band type "
            f"{claim.band_type!r}; simultaneous sup-t bands are required "
            "(INF-013/INF-014)"
        )

    if claim.null_claim:
        if not claim.equivalence_test_id:
            raise ClaimError(
                f"{claim.claim_id}: null/bounded claim without a registered "
                "equivalence test (EQV-004/RPT-008)"
            )
        if claim.equivalence_test_id not in known:
            raise ClaimError(
                f"{claim.claim_id}: equivalence test "
                f"{claim.equivalence_test_id!r} is not in the registry"
            )

    if claim.copy_trading_claim:
        positions = set(claim.supporting_positions)
        if positions and positions <= {"short_after_sale"}:
            raise ClaimError(
                f"{claim.claim_id}: copy-trading claim supported only by "
                "short-after-sale positions (RPT-006 — not literal copying)"
            )
        if "long_after_purchase" not in positions:
            raise ClaimError(
                f"{claim.claim_id}: copy-trading claim requires "
                "long-after-purchase support (RPT-006)"
            )
