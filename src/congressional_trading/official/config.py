"""Configuration for the official-source pipeline.

Precedence: explicit arguments > environment variables > ``configs/official.yaml``
defaults. The Senate adapter refuses to run unless the statutory-use
acknowledgment flag is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..paths import REPO_ROOT, get_paths

STOCK_ACT_PTR_EFFECTIVE = "2012-07-03"  # STOCK Act §6 PTR requirement effective date

# Requests identify the project and the operator; set OFFICIAL_CONTACT before collecting.
DEFAULT_USER_AGENT = (
    "TradingOnDisclosureResearch/0.1 "
    f"(academic research; contact: {os.environ.get('OFFICIAL_CONTACT', 'unset')})"
)


def current_utc_year() -> int:
    return datetime.now(timezone.utc).year


@dataclass
class OfficialConfig:
    data_root: Path
    start_year: int = 2012
    end_year: int | None = None  # None = current UTC year, resolved at run time
    chambers: tuple[str, ...] = ("house", "senate")
    user_agent: str = DEFAULT_USER_AGENT
    request_interval_seconds: float = 2.0
    connect_timeout: float = 20.0
    read_timeout: float = 120.0
    max_retries: int = 4
    backoff_base_seconds: float = 5.0
    senate_usage_acknowledged: bool = False
    ocr_enabled: bool = False
    confidence_review_threshold: float = 0.8
    force_refresh: bool = False

    house_base: str = "https://disclosures-clerk.house.gov"
    senate_base: str = "https://efdsearch.senate.gov"

    extra: dict = field(default_factory=dict)

    def resolved_end_year(self) -> int:
        return self.end_year if self.end_year is not None else current_utc_year()

    @property
    def raw_store(self) -> Path:
        return self.data_root / "raw_store"

    @property
    def manifest_dir(self) -> Path:
        return self.data_root / "manifests"

    @property
    def db_path(self) -> Path:
        return self.data_root / "state.sqlite"

    @property
    def normalized_dir(self) -> Path:
        return self.data_root / "normalized"

    @property
    def release_dir(self) -> Path:
        return self.data_root / "release"

    @property
    def reports_dir(self) -> Path:
        return self.data_root / "reports"

    def ensure_dirs(self) -> None:
        for p in (
            self.raw_store,
            self.manifest_dir,
            self.normalized_dir,
            self.release_dir,
            self.reports_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def load_official_config(
    data_root: Path | str | None = None,
    config_file: Path | None = None,
    **overrides,
) -> OfficialConfig:
    """Load config from configs/official.yaml, env, and overrides."""
    cfg_path = config_file or (REPO_ROOT / "configs" / "official.yaml")
    file_cfg: dict = {}
    if cfg_path.exists():
        file_cfg = yaml.safe_load(cfg_path.read_text()) or {}

    if data_root is None:
        env_root = os.environ.get("OFFICIAL_DATA_ROOT") or file_cfg.get("data_root")
        if env_root:
            data_root = Path(os.path.expanduser(os.path.expandvars(str(env_root))))
        else:
            data_root = get_paths().data_root / "official"
    data_root = Path(data_root)

    kwargs: dict = {
        "data_root": data_root,
        "start_year": int(file_cfg.get("start_year", 2012)),
        "end_year": file_cfg.get("end_year"),  # None or int; never hardcode a year
        "request_interval_seconds": float(file_cfg.get("request_interval_seconds", 2.0)),
        "max_retries": int(file_cfg.get("max_retries", 4)),
        "ocr_enabled": bool(file_cfg.get("ocr_enabled", False)),
        "confidence_review_threshold": float(file_cfg.get("confidence_review_threshold", 0.8)),
        "senate_usage_acknowledged": _bool_env(
            "SENATE_EFD_USAGE_ACKNOWLEDGED", bool(file_cfg.get("senate_usage_acknowledged", False))
        ),
    }
    if file_cfg.get("user_agent"):
        kwargs["user_agent"] = str(file_cfg["user_agent"])
    kwargs.update(overrides)
    cfg = OfficialConfig(**kwargs)
    if cfg.end_year is not None and isinstance(cfg.end_year, str):
        cfg.end_year = None if cfg.end_year == "auto" else int(cfg.end_year)
    return cfg
