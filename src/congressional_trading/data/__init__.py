"""Data layer: cleaning, market-data loading, matching, and dataset assembly.

Modules
-------
- :mod:`clean_capitol_trades` — raw scrape → canonical trade records.
- :mod:`crsp_loader` — CRSP v2 trimmed panel (primary), v1 fallback, trimmer.
- :mod:`factor_loader` — Fama-French factor data (pre-merged or raw files).
- :mod:`permno_matcher` — ticker→PERMNO matching with activity windows.
- :mod:`sample_filters` — main-sample inclusion criteria + attrition.
- :mod:`variable_construction` — regression controls (ln_size, ln_mktcap, …).
- :mod:`build_analysis_dataset` — orchestration of the above stages.
"""

from .clean_capitol_trades import CleanTradesResult, clean_capitol_trades
from .crsp_loader import CrspData, load_crsp_v1, load_crsp_v2_trimmed, trim_crsp_v2_file
from .factor_loader import build_factors_from_french, load_factors_daily, validate_factors
from .permno_matcher import MatchResult, build_ticker_lookup, match_trades_to_permnos
from .sample_filters import add_main_sample_flags, apply_main_sample_filters
from .variable_construction import residualize_ln_size, sic_to_sector

__all__ = [
    "CleanTradesResult",
    "clean_capitol_trades",
    "CrspData",
    "load_crsp_v1",
    "load_crsp_v2_trimmed",
    "trim_crsp_v2_file",
    "build_factors_from_french",
    "load_factors_daily",
    "validate_factors",
    "MatchResult",
    "build_ticker_lookup",
    "match_trades_to_permnos",
    "add_main_sample_flags",
    "apply_main_sample_filters",
    "residualize_ln_size",
    "sic_to_sector",
]
