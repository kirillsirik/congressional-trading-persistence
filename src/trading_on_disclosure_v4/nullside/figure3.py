"""Panels of Figure 3: the best household against its placebo distributions, and the simultaneous rank sets.

Reusable on the committed result files: `data/v4/results.csv` (the actual largest household mean and its
placebo p-values), the cached per-draw placebo statistics under `data/v4/restricted/e/placebo_stats/`
(timing-only and centered flavours, best-household key), and `data/v4/results_rank_sets.csv` (anonymized
indices). `main()` writes the aggregate tables `data/v4/figures/figure3_*.csv` (histogram bins and the
plotted rank sets, indices only). The figures themselves are composed in `nullside/scatter.py`, which pairs
the histogram with the household scatter for the main text and puts the rank sets in their own appendix
figure. No household id is read anywhere.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from trading_on_disclosure_v4.estimators import placebo_stats as ps  # noqa: E402
from trading_on_disclosure_v4.estimators.specs import analysis_specs  # noqa: E402
from trading_on_disclosure_v4.outcomes import panel  # noqa: E402

RESULTS = panel.REPO_ROOT / "data" / "v4" / "results.csv"
RANK_SETS = panel.REPO_ROOT / "data" / "v4" / "results_rank_sets.csv"
PLACEBO_DIR = panel.REPO_ROOT / "data" / "v4" / "restricted" / "e" / "placebo_stats"
FIG_DIR = panel.REPO_ROOT / "paper" / "figures"
TABLE_DIR = panel.REPO_ROOT / "data" / "v4" / "figures"
CLOCK_LABEL = {"T": "trade date", "P": "disclosure date"}
FILE_SUFFIX = {"T": "", "P": "_public"}
# grayscale-readable pair: a light solid fill and a hatched outline, plus distinct line styles
STYLE = {"N1": dict(label="timing-only placebo", facecolor="#c9c9c4", edgecolor="#6b6a63", hatch=None, ls="--"),
         "N2": dict(label="centered placebo", facecolor="none", edgecolor="#1f1f1e", hatch="////", ls=":")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#dcdbd3"
N_BINS = 40
FS = 8  # minimum font size as printed: the figure is drawn at its printed width (.96\textwidth = 6.24 in)
FIG_SIZE = (6.24, 2.9)


# ---------------------------------------------------------------- inputs

def load_placebo_max_mean(clock: str, placebo_dir: Path = PLACEBO_DIR) -> dict[str, np.ndarray]:
    """Per-draw largest household mean (households with >= 20 purchases, winsorized) for both flavours."""
    spec = analysis_specs(clock)["best_household"]
    out = {}
    for flavour in ("N1", "N2"):
        table = pd.read_parquet(placebo_dir / f"placebo_{clock}_{flavour}_default.parquet")
        out[flavour] = ps.values(table, spec, "max_mean")
    return out


def load_actual(clock: str, results: Path = RESULTS) -> dict[str, float]:
    """Actual largest household mean and its two placebo p-values from results.csv."""
    df = pd.read_csv(results)
    b = df.loc[(df["clock"] == clock) & (df["analysis"] == "best_household") & (df["statistic"] == "max_mean")].set_index("variant")
    return {"max_mean": float(b.loc["vs_centered_placebo", "value"]), "p_centered": float(b.loc["vs_centered_placebo", "placebo_p"]),
            "p_timing": float(b.loc["vs_timing_placebo", "placebo_p"]), "n_households": int(b.loc["vs_centered_placebo", "n_households"])}


def load_rank_sets(clock: str, rank_sets: Path = RANK_SETS) -> pd.DataFrame:
    r = pd.read_csv(rank_sets)
    return r.loc[r["clock"] == clock].sort_values("point_rank").reset_index(drop=True)


# ---------------------------------------------------------------- tables

def histogram_table(draws: dict[str, np.ndarray], actual: float, n_bins: int = N_BINS) -> pd.DataFrame:
    """Common bin edges over both flavours (the actual value inside the range) and the count per bin and flavour."""
    allv = np.concatenate(list(draws.values()) + [np.array([actual])])
    edges = np.histogram_bin_edges(allv, bins=n_bins)
    rows = {"bin_lower_bp": edges[:-1], "bin_upper_bp": edges[1:]}
    for flavour, v in draws.items():
        rows[f"count_{STYLE[flavour]['label'].split()[0]}"] = np.histogram(v, bins=edges)[0]
    return pd.DataFrame(rows)


def rank_set_table(sets: pd.DataFrame) -> pd.DataFrame:
    """The plotted rank sets, indices only (no estimate, no purchase count)."""
    return sets[["household_index", "point_rank", "rank_lower", "rank_upper", "width"]].copy()


# ---------------------------------------------------------------- drawing

def _panel_a(ax, draws: dict[str, np.ndarray], actual: dict[str, float], edges: np.ndarray) -> None:
    for flavour in ("N1", "N2"):
        v, st = draws[flavour], STYLE[flavour]
        p = actual["p_centered"] if flavour == "N2" else actual["p_timing"]
        ax.hist(v, bins=edges, density=True, facecolor=st["facecolor"], edgecolor=st["edgecolor"], hatch=st["hatch"], linewidth=0.7,
                label=f"{st['label'].split()[0]}, p = {p:.3f}")
        ax.axvline(np.quantile(v, 0.95), color=st["edgecolor"], linewidth=1, linestyle=st["ls"])
    ax.axvline(actual["max_mean"], color=INK, linewidth=2)
    ymax = ax.get_ylim()[1]
    ax.annotate(f"actual: {actual['max_mean']:,.0f} bp", (actual["max_mean"], ymax * 0.98), xytext=(4, 0), textcoords="offset points",
                ha="left", va="top", fontsize=FS, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("Largest household mean abnormal return\n(bp per 20 trading days)", fontsize=FS, color=INK, labelpad=2)
    ax.set_ylabel("Density across placebo datasets", fontsize=FS, color=INK, labelpad=3)
    ax.set_title(f"(a) Largest household mean ({actual['n_households']} households)", fontsize=FS, loc="left", color=INK, pad=4)
    n_draws = max(len(v) for v in draws.values())
    ax.legend(fontsize=FS, frameon=True, framealpha=1, edgecolor="none", loc="upper right", bbox_to_anchor=(1.03, 0.86), borderpad=0.25,
              labelspacing=0.25, handlelength=1.2, handletextpad=0.5,
              title=f"{n_draws:,} placebo datasets", title_fontsize=FS, alignment="left")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(colors=MUTED, labelsize=FS, length=2.5, pad=2)


def _panel_b(ax, sets: pd.DataFrame) -> None:
    H = len(sets)
    x = sets["household_index"].to_numpy(dtype=int)
    ax.vlines(x, sets["rank_lower"], sets["rank_upper"], color="#9a9991", linewidth=1.0, label="simultaneous 95% rank set")
    ax.plot(x, sets["point_rank"], color=INK, linewidth=0, marker="o", markersize=2.0, label="point rank")
    median = (H + 1) / 2.0
    ax.axhline(median, color=INK, linewidth=1, linestyle="--", label=f"median rank ({median:g})")
    n_full = int(((sets["rank_lower"] == 1) & (sets["rank_upper"] == H)).sum())
    n_distinct = int(((sets["rank_upper"] < median) | (sets["rank_lower"] > median)).sum())
    ax.text(0.03, 0.04, f"{n_full} of {H} sets cover every rank;\n{n_distinct} exclude the median rank", transform=ax.transAxes,
            fontsize=FS, color=INK, va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlim(0, H + 1)
    ax.set_ylim(H + 1, 0)  # rank 1 (largest mean) at the top
    ax.set_yticks([1] + list(range(20, H + 1, 20)))
    ax.set_xlabel("Household (anonymized index,\nordered by point estimate)", fontsize=FS, color=INK, labelpad=2)
    ax.set_ylabel("Rank (1 = largest mean)", fontsize=FS, color=INK, labelpad=3)
    ax.set_title("(b) Simultaneous 95% rank sets", fontsize=FS, loc="left", color=INK, pad=4)
    ax.legend(fontsize=FS, frameon=True, framealpha=1, edgecolor="none", loc="upper right", bbox_to_anchor=(1.03, 1.03), borderpad=0.25,
              labelspacing=0.2, handlelength=1.2, handletextpad=0.5)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=FS, length=2.5, pad=2)


def min_font_size(fig) -> float:
    """Smallest font size (pt) over every Text artist of the figure with non-empty text."""
    from matplotlib.text import Text

    return min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip())


def printed_box(fig, pad_inches: float = 0.02, width_in: float | None = None):
    """The tight box with a hairline pad, widened to `width_in` (less a hair) when the drawing alone is narrower, so
    that the page neither scales the figure down nor magnifies its text."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    bb = Bbox.from_extents(bb.x0 - pad_inches, bb.y0 - pad_inches, bb.x1 + pad_inches, bb.y1 + pad_inches)
    target = None if width_in is None else width_in - 0.01
    if target is not None and bb.width < target:
        grow = (target - bb.width) / 2
        bb = Bbox.from_extents(bb.x0 - grow, bb.y0, bb.x1 + grow, bb.y1)
    return bb


def save(fig, stem: Path, pad_inches: float = 0.02, width_in: float | None = None) -> list[Path]:
    """Both formats at the printed width: a tight box with a hairline pad, so the page never has to scale the figure down."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    meta = {"Creator": "Trading on Disclosure", "CreationDate": None, "ModDate": None}
    box = printed_box(fig, pad_inches, width_in)
    for p in out:
        fig.savefig(p, dpi=200 if p.suffix == ".png" else None, bbox_inches=box,
                    metadata=meta if p.suffix == ".pdf" else None)
    plt.close(fig)
    return out


def main(clocks=("T", "P"), table_dir: Path = TABLE_DIR, placebo_dir: Path = PLACEBO_DIR,
         results: Path = RESULTS, rank_sets: Path = RANK_SETS) -> list[Path]:
    """The two aggregate tables per clock. The figures are written by `nullside/scatter.py`."""
    table_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for clock in clocks:
        draws, actual, sets = load_placebo_max_mean(clock, placebo_dir), load_actual(clock, results), load_rank_sets(clock, rank_sets)
        histogram_table(draws, actual["max_mean"]).assign(clock=clock).to_csv(
            table_dir / f"figure3_hist_max_mean_{clock}.csv", index=False, lineterminator="\n")
        rank_set_table(sets).assign(clock=clock).to_csv(table_dir / f"figure3_rank_sets_{clock}.csv", index=False, lineterminator="\n")
        written += [table_dir / f"figure3_hist_max_mean_{clock}.csv", table_dir / f"figure3_rank_sets_{clock}.csv"]
    return written


if __name__ == "__main__":
    print("\n".join(str(p) for p in main()))
