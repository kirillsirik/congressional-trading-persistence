"""Date-valid security resolution with match tiers (SEC-001..SEC-008).

Candidate generation and tier assignment never select among nonunique
candidates automatically; unresolved rows are Tier D, excluded and counted.
"""

from trading_on_disclosure_v2.securities.matching import (  # noqa: F401
    apply_common_stock_screen,
    assign_tiers,
    generate_candidates,
    selection_diagnostics,
)
