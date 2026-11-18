"""Output and manuscript infrastructure (standing directive §K).

Every displayed value flows result artifact → numbers ledger → shell →
export. There is no manual number-entry path: shells hold ledger keys, and
exports resolve keys against the ledger at render time (RPT-002, JC-023).
"""

__all__ = ["shells", "figures", "numbers_ledger", "claims", "exports"]
