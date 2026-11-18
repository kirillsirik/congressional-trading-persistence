"""Figure 4 (plan §6.8): years to detect an annualized-scale effect theta at n purchases/year.

Two entry points. The pre-run draft, `plot_figure4(ttd, activity, ...)`, draws the grid of
`nullside.time_to_detect.ttd_table` with a d-sensitivity band. The paper's figure,
`plot_figure4_results(grid, activity, actual, ...)`, draws `data/v4/results_time_to_detect.csv` (measured
sigma and design effect, plain and best-of-H) and marks the observed span of the record and the median
purchases per household-year from `data/v4/results.csv`; `main()` writes
`paper/figures/v4_fig4_time_to_detect[_public].{pdf,png}`. Both use D's purchases-per-household-year
histogram (counts only) as a marginal strip. `save_figure4(fig, stem)` writes <stem>.pdf and <stem>.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from trading_on_disclosure_v4.nullside import figure3  # noqa: E402
from trading_on_disclosure_v4.nullside.time_to_detect import (  # noqa: E402
    OBSERVED_SPAN_YEARS,
    activity_histogram,
)
from trading_on_disclosure_v4.outcomes import panel  # noqa: E402

# fixed categorical order (colorblind-validated palette; one hue per theta, never cycled)
THETA_COLORS = {2.0: "#2a78d6", 5.0: "#eb6834", 10.0: "#1baf7a", 20.0: "#eda100"}
THETA_STYLES = {2.0: "-", 5.0: "--", 10.0: "-.", 20.0: ":"}  # grayscale-readable twin of the hues
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#dcdbd3"
RESULTS = panel.REPO_ROOT / "data" / "v4" / "results.csv"
TTD_GRID = panel.REPO_ROOT / "data" / "v4" / "results_time_to_detect.csv"
FIG_DIR = panel.REPO_ROOT / "paper" / "figures"
FILE_SUFFIX = {"T": "", "P": "_public"}
X_LIM, X_TICKS = (0.7, 1000), [1, 3, 10, 30, 100, 300]
Y_LIM = (0.2, 40000)
FS = 9  # keeps labels above 8 pt when the padded canvas is printed at .96\textwidth
FIG_SIZE = (6.24, 4.3)
EXPORT_PADDING_IN = 0.25
LABEL_CLEAR_DECADES = 0.3  # a curve's end label moves off the dashed span line when its point is within this (log10 years)


def _panel(ax, sub: pd.DataFrame, d_main: float, d_band: tuple[float, float], title: str) -> None:
    for th, g in sub.groupby("theta_annual_pct", sort=True):
        c = THETA_COLORS[float(th)]
        main = g[g["d"] == d_main].sort_values("n_per_year")
        lo = g[g["d"] == d_band[0]].sort_values("n_per_year")
        hi = g[g["d"] == d_band[1]].sort_values("n_per_year")
        if len(lo) and len(hi):
            ax.fill_between(lo["n_per_year"], lo["years"], hi["years"], color=c, alpha=0.12, linewidth=0)
        ax.plot(main["n_per_year"], main["years"], color=c, linewidth=2, marker="o", markersize=4,
                label=f"θ = {th:g}% (annualized scale)")
        ax.annotate(f"{th:g}%", (main["n_per_year"].iloc[-1], main["years"].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", color=INK, fontsize=FS, va="center")
    ax.axhline(OBSERVED_SPAN_YEARS, color=INK, linewidth=1, linestyle="--")
    ax.text(0.8, OBSERVED_SPAN_YEARS * 1.15, f"observed span: {OBSERVED_SPAN_YEARS} years", color=INK, fontsize=FS)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(X_TICKS)
    ax.set_xlim(*X_LIM)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title(title, fontsize=FS, loc="left", color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=FS)


def _activity_strip(ax, activity: pd.DataFrame, clock: str) -> None:
    a = activity[activity["clock"] == clock]
    for r in a.itertuples():
        lo, hi = r.lo - 0.5, min(r.hi + 0.5, 700)
        ax.bar(lo, r.share, width=hi - lo, color="#9ec5f4", edgecolor="white", linewidth=1, align="edge")
        ax.text(np.sqrt(lo * hi), r.share, f"{100 * r.share:.0f}", ha="center", va="bottom", fontsize=FS, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0, a["share"].max() * 1.9)
    ax.set_xticks(X_TICKS)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([])
    ax.set_ylabel("% of\nhousehold-\nyears", fontsize=FS, color=MUTED, rotation=0, ha="right", va="center", labelpad=4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=FS, length=2.5, pad=2)


def plot_figure4(ttd: pd.DataFrame, activity: pd.DataFrame, clock: str = "T", sigma_variant: str = "raw",
                 d_main: float = 1.0, d_band: tuple[float, float] = (1.0, 3.0), h_adjust: str | None = None):
    """Two panels (plain; Bonferroni best-of-H) + activity strip. Returns the matplotlib figure."""
    sub = ttd[(ttd["clock"] == clock) & (ttd["sigma_variant"] == sigma_variant)]
    if h_adjust is None:  # default: the smallest H (the R1-eligible count), not the 330 sensitivity
        bonf = sub[sub["adjustment"].str.startswith("bonferroni")]
        h_adjust = bonf.loc[bonf["H"].idxmin(), "adjustment"]
    sigma = float(sub["sigma_bp"].iloc[0])
    h = int(sub.loc[sub["adjustment"] == h_adjust, "H"].iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE, sharex="col", gridspec_kw={"height_ratios": [5, 1], "hspace": 0.08, "wspace": 0.25})
    _panel(axes[0, 0], sub[sub["adjustment"] == "plain"], d_main, d_band, "(a) one pre-named household")
    _panel(axes[0, 1], sub[sub["adjustment"] == h_adjust], d_main, d_band, f"(b) best of G = {h} households, Bonferroni")
    axes[0, 0].set_ylabel("years of trading needed", fontsize=FS, color=INK)
    for ax in axes[0]:
        ax.set_ylim(*Y_LIM)
    for ax in axes[1]:
        _activity_strip(ax, activity, clock)
        ax.set_xlabel("purchases per year (log scale)", fontsize=FS, color=INK)
    axes[0, 0].legend(fontsize=FS, frameon=False, loc="lower left")
    fig.suptitle(f"Years to detect persistent ability (one-sided 5%, 80% power) — {dict(T='trade date', P='disclosure date')[clock]}, σ = {sigma:,.0f} bp "
                 f"per 20 sessions ({sigma_variant}); line d = {d_main:g}, band d = {d_band[0]:g}–{d_band[1]:g}", fontsize=FS, color=INK, x=0.01, ha="left")
    return fig  # no footer inside the image: what a reader needs is in the caption


def save_figure4(fig, stem: Path, width_in: float | None = None) -> list[Path]:
    """Both formats; `width_in` widens the saved box to the printed width when the drawing alone is narrower."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    # Preserve the complete figure canvas, including the reserved legend band.
    # Union with the artist bounds protects any tick label outside that canvas;
    # quarter-inch padding then separates all artwork from every PDF edge.
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer(), bbox_extra_artists=tuple(fig.legends))
    canvas = fig.bbox_inches
    box = Bbox.union([canvas, tight]).padded(EXPORT_PADDING_IN)
    if width_in is not None and canvas.width < width_in:
        grow = (width_in - canvas.width) / 2
        box = Bbox.from_extents(box.x0 - grow, box.y0, box.x1 + grow, box.y1)
    for p in out:
        fig.savefig(p, dpi=200 if p.suffix == ".png" else None,
                    bbox_inches=box, pad_inches=EXPORT_PADDING_IN)
    plt.close(fig)
    return out


def mark_actual(ax, n_per_year: float, years: float, label: str) -> None:
    """Post-tag helper: place the estimated (n, years) point; unused before the tag."""
    ax.plot([n_per_year], [years], marker="D", color=INK, markersize=6)
    ax.annotate(label, (n_per_year, years), xytext=(6, 6), textcoords="offset points", fontsize=FS, color=INK)


# ---------------------------------------------------------------- the paper's figure (measured sigma and d)

def load_actual(clock: str, results: Path = RESULTS) -> dict[str, float]:
    """sigma, design effect, H, median purchases per household-year and span from results.csv (no identity)."""
    df = pd.read_csv(results)
    t = df.loc[(df["clock"] == clock) & (df["analysis"] == "time_to_detect") & (df["variant"] == "default")].set_index("statistic")["value"]
    return {"sigma": float(t["sigma"]), "design_effect": float(t["design_effect"]), "H": int(t["H"]),
            "median_purchases_per_household_year": float(t["median_purchases_per_household_year"]), "span_years": float(t["span_years"])}


def load_grid(clock: str, grid: Path = TTD_GRID) -> pd.DataFrame:
    g = pd.read_csv(grid)
    return g.loc[g["clock"] == clock].sort_values(["theta_annual_pct", "n_per_year"]).reset_index(drop=True)


def curve_label_offset(y_end: float, span: float, clear_decades: float = LABEL_CLEAR_DECADES) -> tuple[float, str]:
    """(vertical offset in points, va) of a curve's end label: centered on the last point unless the dashed span line
    passes within `clear_decades` of it, in which case the label moves to the side of the point away from the line."""
    gap = np.log10(y_end) - np.log10(span)
    if abs(gap) >= clear_decades:
        return 0.0, "center"
    return (2.0, "bottom") if gap > 0 else (-2.0, "top")


def _results_panel(ax, sub: pd.DataFrame, years_col: str, actual: dict[str, float], title: str) -> None:
    span = actual["span_years"]
    for th, g in sub.groupby("theta_annual_pct", sort=True):
        th = float(th)
        g = g.sort_values("n_per_year")
        ax.plot(g["n_per_year"], g[years_col], color=THETA_COLORS[th], linestyle=THETA_STYLES[th], linewidth=1.6, marker="o",
                markersize=3.2, label=f"edge {th:g}% per year")
        dy, va = curve_label_offset(float(g[years_col].iloc[-1]), span)
        ax.annotate(f"{th:g}%", (g["n_per_year"].iloc[-1], g[years_col].iloc[-1]), xytext=(4, dy), textcoords="offset points",
                    color=INK, fontsize=FS, va=va)
    ax.axhline(span, color=INK, linewidth=1, linestyle="--")
    ax.text(X_LIM[0] * 1.15, span * 1.15, f"span of the record:\n{span:g} years", color=INK, fontsize=FS, ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    med = actual["median_purchases_per_household_year"]
    ax.axvline(med, color=MUTED, linewidth=1, linestyle=":")
    ax.text(med * 1.12, Y_LIM[1], f"median household:\n{med:g} purchases a year", color=MUTED, fontsize=FS, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(X_TICKS)
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title, fontsize=FS, loc="left", color=INK, pad=4)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=FS, length=2.5, pad=2)


def plot_figure4_results(grid: pd.DataFrame, activity: pd.DataFrame, actual: dict[str, float], clock: str = "T"):
    """Two panels (plain; Bonferroni best-of-H) on the measured sigma and d, plus the activity strip; drawn at the printed
    width so that every text is >= FS pt on the page. The legend sits below the strips."""
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE, sharex="col", gridspec_kw={"height_ratios": [5, 1], "hspace": 0.10, "wspace": 0.22})
    _results_panel(axes[0, 0], grid, "years", actual, "(a) One pre-named household")
    _results_panel(axes[0, 1], grid, "years_best_of_h", actual, f"(b) Best of G = {actual['H']} households\n(Bonferroni correction)")
    axes[0, 0].set_ylabel("Years of disclosures to 80% power\n(log scale)", fontsize=FS, color=INK, labelpad=3)
    for ax in axes[1]:
        _activity_strip(ax, activity, clock)
        ax.set_xlabel("Purchases per year (log scale)", fontsize=FS, color=INK, labelpad=2)
    # Reserve room for the multiline labels under differing font-renderer
    # metrics, so the artwork stays within the fixed-width export canvas.
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.34, top=0.91)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=FS, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=2, columnspacing=2.0, handlelength=1.5, handletextpad=0.4, borderpad=0.3, borderaxespad=0,
               title=rf"$\sigma_Y = {actual['sigma']:,.1f}$ bp per 20 trading days, d = {actual['design_effect']:.3f} (measured)", title_fontsize=FS,
               alignment="center")
    return fig  # no footer inside the image: the clock, the test and the strip are named in the caption


def min_font_size(fig) -> float:
    """Smallest font size (pt) over every Text artist of the figure with non-empty text."""
    from matplotlib.text import Text

    return min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip())


def main(clocks=("T", "P"), fig_dir: Path = FIG_DIR, results: Path = RESULTS, grid_path: Path = TTD_GRID) -> list[Path]:
    activity = activity_histogram()
    out = []
    for clock in clocks:
        fig = plot_figure4_results(load_grid(clock, grid_path), activity, load_actual(clock, results), clock)
        out += save_figure4(fig, fig_dir / f"v4_fig4_time_to_detect{FILE_SUFFIX[clock]}", width_in=FIG_SIZE[0])
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clocks", choices=("T", "P"), nargs="+", default=("T", "P"))
    args = parser.parse_args()
    print("\n".join(str(p) for p in main(clocks=args.clocks)))
