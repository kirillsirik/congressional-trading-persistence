"""Deterministic figure generators for the catalog's F01–F05 (RPT-001).

Each generator records its band/estimator provenance in the returned
metadata; a path-wide display from a pointwise band is refused at the source
(INF-014).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trading_on_disclosure_v2.inference.event_bands import GaussianShortcutError


@dataclass(frozen=True)
class FigureRecord:
    output_id: str
    path: Path
    band_type: str | None
    ledger_keys: tuple[str, ...]
    caption: str


def _finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata={"Software": ""})
    plt.close(fig)


def two_clock_schematic(out_path: Path) -> FigureRecord:
    """F01: transaction date → official public date → strict-next-close entry
    → 20-day horizon. Conceptual; no empirical values."""
    fig, ax = plt.subplots(figsize=(8, 2.2))
    points = [
        (0.0, "reported\ntransaction date"),
        (0.42, "official public\ndate proxy"),
        (0.55, "strict next-close\nfollower entry"),
        (1.0, "20-trading-day\nhorizon end"),
    ]
    ax.hlines(0, -0.02, 1.02, color="0.3", lw=1.5)
    for x, label in points:
        ax.plot([x], [0], marker="o", color="0.1")
        ax.annotate(label, (x, 0), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.annotate("transaction clock", (0.0, 0), xytext=(0, -22),
                textcoords="offset points", ha="left", fontsize=8, color="C0")
    ax.annotate("public-action clock", (0.55, 0), xytext=(0, -22),
                textcoords="offset points", ha="left", fontsize=8, color="C1")
    ax.set_axis_off()
    _finish(fig, out_path)
    return FigureRecord("F01", out_path, None, (),
                        "The two clocks: reported transaction vs public action.")


def event_study_paths(
    out_path: Path,
    event_time: np.ndarray,
    buy_path: np.ndarray,
    sell_path: np.ndarray,
    band_lower: np.ndarray,
    band_upper: np.ndarray,
    band_type: str,
    entry_marker: int,
    ledger_keys: tuple[str, ...],
) -> FigureRecord:
    """F03: purchase/sale cumulative paths with simultaneous bands."""
    if band_type != "sup_t":
        raise GaussianShortcutError(
            f"event-study figure with band type {band_type!r}; path display "
            "requires simultaneous sup-t bands (INF-013/INF-014)"
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(event_time, band_lower, band_upper, alpha=0.2,
                    label=f"simultaneous band ({band_type})")
    ax.plot(event_time, buy_path, label="purchases")
    ax.plot(event_time, sell_path, label="sales", ls="--")
    ax.axvline(entry_marker, color="0.4", lw=1,
               label="strict next-close public entry")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("trading days relative to official public date")
    ax.set_ylabel("cumulative abnormal return")
    ax.legend(fontsize=8)
    _finish(fig, out_path)
    return FigureRecord("F03", out_path, band_type, ledger_keys,
                        "Disclosure-centered event paths with public-entry marker.")


def portfolio_wealth_exposure(
    out_path: Path,
    dates: np.ndarray,
    strategy_wealth: np.ndarray,
    benchmark_wealth: np.ndarray,
    risky_exposure: np.ndarray,
    ledger_keys: tuple[str, ...],
) -> FigureRecord:
    """F05: net wealth vs exposure-matched benchmark plus exposure share."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax1.plot(dates, strategy_wealth, label="long-after-purchase (net)")
    ax1.plot(dates, benchmark_wealth, label="exposure-matched benchmark",
             ls="--")
    ax1.set_ylabel("wealth")
    ax1.legend(fontsize=8)
    ax2.fill_between(dates, risky_exposure, alpha=0.4)
    ax2.set_ylabel("risky share")
    ax2.set_ylim(0, 1.05)
    _finish(fig, out_path)
    return FigureRecord("F05", out_path, None, ledger_keys,
                        "Portfolio wealth and exposure; idle cash earns rf.")
