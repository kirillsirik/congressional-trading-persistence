"""Central deterministic seed registry (§M, CFG-005/INF-008).

Every stochastic component derives its streams from a named root through
sha256; nothing seeds from wall-clock or process state. The registry below is
asserted against the actual module sources by tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedRule:
    name: str
    root_or_pattern: str
    derivation: str
    consumer: str  # module path that implements it


SEED_REGISTRY: tuple[SeedRule, ...] = (
    SeedRule(
        "block_bootstrap_streams",
        "sha256(freeze_hash|purpose)",
        "uint64 from first 8 bytes of sha256(f'{freeze_hash}|{purpose}')",
        "trading_on_disclosure_v2/inference/block_bootstrap.py",
    ),
    SeedRule(
        "event_band_evaluation",
        "hs08-event-band-evaluation-v1",
        "per-DGP/per-sim independent streams derived from the root by "
        "sha256(root|dgp|variant|sim_index)",
        "trading_on_disclosure_v2.inference.event_bands",
    ),
    SeedRule(
        "calibration_sample",
        "two-clock-calibration-v1",
        "stratified selection order from sha256(root|stratum|filing_id)",
        "manual-transcription calibration sample (not used by the locked design)",
    ),
    SeedRule(
        "audit_sample",
        "sha256(freeze_hash|audit|stratum)",
        "per-stratum inclusion draws; freeze hash supplied at the HG-02 "
        "freeze event, refused before it",
        "manual-transcription audit sample (not used by the locked design)",
    ),
    SeedRule(
        "placebo_battery",
        "derived_from_freeze_hash",
        "contract robustness.placebo_seed_rule; permutation/pseudo-date/shift "
        "seeds from sha256(freeze_hash|placebo_id)",
        "docs/two_clock_package/config/top_journal_analysis_contract.yaml",
    ),
)


def derive_seed(root: str, *parts: str) -> int:
    payload = "|".join([root, *parts])
    return int.from_bytes(
        hashlib.sha256(payload.encode()).digest()[:8], "big", signed=False
    )


def registry_as_dict() -> dict[str, dict]:
    return {
        r.name: {
            "root_or_pattern": r.root_or_pattern,
            "derivation": r.derivation,
            "consumer": r.consumer,
        }
        for r in SEED_REGISTRY
    }
