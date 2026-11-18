"""Strict loader for the locked canonical configuration (final directive §23).

configs/top_journal_two_clock_locked.yaml is the single binding record of the
streamlined two-clock design. Loading fails closed on duplicate keys, missing
keys, unknown or dead keys, and violated binding invariants. The schema below
IS the set of live keys: a key present in the file but absent here is dead
configuration and aborts the load.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

LOCKED_CONFIG_RELPATH = Path("configs") / "top_journal_two_clock_locked.yaml"

#: nested mapping: section -> exact allowed keys (None = scalar/list leaf)
SCHEMA: dict = {
    "meta": {
        "config_id": None,
        "design_frozen_date": None,
        "supersedes": None,
        "amendment_policy": None,
    },
    "paper_scope": None,
    "evidence_status": None,
    "sample_start": None,
    "sample_end": None,
    "data_review": {
        "assumption": None,
        "attestation_file": None,
        "description_language": None,
        "forbidden_descriptions": None,
    },
    "public_date": {
        "official_priority": None,
        "vendor_fallback_adjustment_calendar_days": None,
        "vendor_fallback_label": None,
        "unusable_date_action": None,
        "proxy_flag_field": None,
        "source_field": None,
        "source_native_fields_never_overwritten": None,
    },
    "entry": {
        "rule": None,
        "execution": None,
        "first_return": None,
        "entry_day_return_included": None,
    },
    "trade_date": {
        "official_priority": None,
        "vendor_adjustment_calendar_days": None,
        "raw_vendor_sensitivity_required": None,
    },
    "primary_unit": {
        "type": None,
        "mixed_direction": None,
        "preserved_aggregates": None,
    },
    "identity": {
        "hierarchy": None,
        "surname_only_merge": None,
        "outcome_guided_resolution": None,
        "ambiguous_merge_policy": None,
        "source_native_household_sensitivity_required": None,
    },
    "amendments": {
        "hierarchy": None,
        "exact_duplicate_fields": None,
        "backdating": None,
        "unresolved_flag": None,
        "manual_review_of_all_amendments_required": None,
    },
    "primary_outcome": {
        "horizon_trading_days": None,
        "benchmark": None,
        "estimand": None,
        "missing_return_zero_fill": None,
        "complete_horizon_required": None,
        "delisting_rule": None,
    },
    "primary_inference": {
        "estimator": None,
        "cluster_1": None,
        "cluster_2": None,
        "iid_allowed": None,
        "month_clustering_allowed": None,
        "finite_sample_correction": None,
        "dof_rule": None,
        "calendar_overlap_companion": None,
        "security_cluster_sensitivity": None,
    },
    "horizon_robustness": {"horizons": None, "adjustment": None},
    "source_timing_robustness": {"variants": None, "selective_reporting": None},
    "event_band": {
        "method_id": None,
        "multiplier": None,
        "kernel": None,
        "bandwidth": None,
        "null_imposed_studentization": None,
        "draws": None,
        "window": None,
        "runtime_manifest_required": None,
    },
    "placebos": {
        "direction_permutation_within_disclosure_month": None,
        "pseudo_entry_dates_within_security_year": None,
        "pseudo_entry_exclusion_window_trading_days": None,
        "additional_variants": None,
    },
    "portfolio": {
        "strategy": None,
        "holding_days": None,
        "security_cap_nav": None,
        "max_open_lots": None,
        "leverage": None,
        "shorting": None,
        "round_trip_cost_bps": None,
        "cost_split": None,
        "idle_cash": None,
        "terminal_rule": None,
        "train_eval_split": None,
    },
    "multiplicity": {
        "family_P1": None,
        "family_P2_adjustment": None,
        "family_P3": None,
        "family_P4": None,
        "senate_interaction_label": None,
    },
    "language_contract": {
        "historical_evidence_only": None,
        "prospective_language": None,
        "proxy_labeled_official": None,
        "signed_payoff_called_copy_trading": None,
        "transaction_clock_called_follower_strategy": None,
        "equivalence_claims_without_approved_bound": None,
    },
}


class LockedConfigError(RuntimeError):
    """Any structural or invariant violation of the locked configuration."""


class _NoDuplicateLoader(yaml.SafeLoader):
    pass


def _mapping_no_duplicates(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise LockedConfigError(
                f"duplicate key {key!r} at {key_node.start_mark}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDuplicateLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping_no_duplicates
)


@dataclass(frozen=True)
class LockedConfig:
    data: dict
    sha256: str
    source_path: Path

    def __getitem__(self, key: str):
        return self.data[key]


def _check_keys(data: dict, schema: dict, path: str = "") -> None:
    unknown = sorted(set(data) - set(schema))
    missing = sorted(set(schema) - set(data))
    if unknown:
        raise LockedConfigError(
            f"unknown/dead config keys at {path or 'root'}: {unknown}"
        )
    if missing:
        raise LockedConfigError(
            f"missing config keys at {path or 'root'}: {missing}"
        )
    for key, sub in schema.items():
        if isinstance(sub, dict):
            value = data[key]
            if not isinstance(value, dict):
                raise LockedConfigError(
                    f"{path + '.' if path else ''}{key} must be a mapping"
                )
            _check_keys(value, sub, f"{path + '.' if path else ''}{key}")


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise LockedConfigError(message)


def _validate_invariants(d: dict) -> None:
    _require(d["paper_scope"] == "two_clock", "paper_scope must be two_clock")
    _require(
        d["evidence_status"] == "historical",
        "evidence_status must be historical (no-data-snooping contract §19)",
    )
    pub = d["public_date"]
    _require(pub["official_priority"] is True, "official date priority required")
    _require(
        pub["vendor_fallback_adjustment_calendar_days"] == -1,
        "vendor public-date fallback must be minus one CALENDAR day",
    )
    _require(
        pub["vendor_fallback_label"]
        == "CAPITOL_TRADES_MINUS_ONE_CALENDAR_DAY_PROXY",
        "vendor fallback label must name the proxy, never an observed date",
    )
    _require(
        d["trade_date"]["vendor_adjustment_calendar_days"] == 1,
        "vendor transaction-date correction must be plus one calendar day",
    )
    _require(
        d["entry"]["rule"] == "first_trading_day_strictly_after_public_date"
        and d["entry"]["entry_day_return_included"] is False,
        "entry must be strict next trading day with entry-day return excluded",
    )
    inf = d["primary_inference"]
    _require(
        inf["estimator"] == "two_way_cgm"
        and inf["cluster_1"] == "filer_uid"
        and inf["cluster_2"] == "exact_public_entry_date",
        "primary inference must be two-way CGM by filer_uid and exact "
        "public-entry trading date",
    )
    _require(inf["iid_allowed"] is False, "IID headline inference forbidden")
    _require(
        inf["month_clustering_allowed"] is False,
        "month clustering forbidden in primary inference",
    )
    _require(
        d["primary_outcome"]["horizon_trading_days"] == 20
        and d["primary_outcome"]["estimand"] == "purchase_minus_sale",
        "primary estimand is the 20-day purchase-minus-sale spread",
    )
    _require(
        d["primary_outcome"]["missing_return_zero_fill"] == "forbidden",
        "missing-return zero fill is forbidden",
    )
    _require(
        sorted(d["horizon_robustness"]["horizons"]) == [5, 10, 40, 60, 127]
        and d["horizon_robustness"]["adjustment"] == "holm",
        "horizon family must be {5,10,40,60,127} with Holm adjustment",
    )
    band = d["event_band"]
    _require(
        band["method_id"] == "day_mult_nullimp_bwH"
        and band["multiplier"] == "gaussian"
        and band["kernel"] == "bartlett"
        and band["bandwidth"] == "horizon"
        and band["null_imposed_studentization"] is True
        and band["draws"] == 4999,
        "event band must be the validated day_mult_nullimp_bwH method "
        "(gaussian multipliers, Bartlett kernel, fixed bandwidth=H, "
        "null-imposed studentization, 4999 draws)",
    )
    port = d["portfolio"]
    _require(
        port["strategy"] == "all_disclosed_purchases"
        and port["holding_days"] == 20
        and port["security_cap_nav"] == 0.05
        and port["max_open_lots"] == 50
        and port["leverage"] is False
        and port["shorting"] is False
        and port["round_trip_cost_bps"] == 10
        and port["train_eval_split"] == "none",
        "portfolio must be the prespecified literal all-purchases follower "
        "strategy with the locked accounting rules",
    )
    _require(
        d["primary_unit"]["mixed_direction"] == "exclude_primary",
        "mixed-direction signals are excluded from the primary contrast",
    )
    _require(
        d["identity"]["surname_only_merge"] == "forbidden",
        "surname-only identity merging is forbidden",
    )
    _require(
        d["amendments"]["backdating"] == "forbidden",
        "backdating amendment information is forbidden",
    )
    _require(
        d["multiplicity"]["senate_interaction_label"]
        == "HISTORICAL_EXPLORATORY_RESULT_REQUIRING_NEW_DATA",
        "the historical Senate interaction stays exploratory",
    )


def load_locked_config(repo_root: Path | str | None = None) -> LockedConfig:
    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    path = repo / LOCKED_CONFIG_RELPATH
    raw = path.read_bytes()
    data = yaml.load(raw, Loader=_NoDuplicateLoader)
    if not isinstance(data, dict):
        raise LockedConfigError("locked config root must be a mapping")
    _check_keys(data, SCHEMA)
    _validate_invariants(data)
    return LockedConfig(
        data=data,
        sha256=hashlib.sha256(raw).hexdigest(),
        source_path=path,
    )
