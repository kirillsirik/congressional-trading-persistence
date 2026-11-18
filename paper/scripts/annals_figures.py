#!/usr/bin/env python3
"""Figure G1: the smallest gain at 80% power in the trade-date sample, from the stored detection frontiers.

Run with Python plus matplotlib, numpy and pandas. Every plotted curve and marker is checked against
its CSV coordinate, and the markers and annotations against the stored validation record of the figure
(paper/figures/detection_figures_validation.json). Writes annals_figure3_detection_trade.pdf (and .png).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from palatino_plot_font import configure_palatino_font

PAPER = Path(__file__).resolve().parents[1]
SOURCE = PAPER.parent
FS = 8.0
REACHED = {"interpolated", "at_grid_min"}
MARKERS = {"T1_blk20_2s": "o", "T2_I1cal_ES": "s", "T2_I2b_ES": "D", "T3": "^", "T4_z2": "v"}
NAMES = {"T1_blk20_2s": "Pooled-mean test", "T2_I1cal_ES": "Studentized covariance",
         "T2_I2b_ES": "Bootstrap covariance", "T3": "Maximum-household test", "T4_z2": "Upper-tail test"}
COLORS = {"T1_blk20_2s": "#0072b2", "T2_I1cal_ES": "#cc79a7", "T2_I2b_ES": "#009e73",
          "T3": "#d55e00", "T4_z2": "#6f4f98"}
FIGURES = (
    ("figure2_detection_region_v3", "v4_draft_fig2_detection_region", "annals_figure3_detection_trade", "Trade date", True),
)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_value(a, b, rtol=1e-14):
    """Equality of stored figure coordinates up to the last-ulp differences of platform CSV float parsing."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_value(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= rtol * max(abs(a), abs(b)))
    return a == b


def numeric_array(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def assert_line(line, x, y):
    np.testing.assert_array_equal(line.get_xdata(), np.asarray(x))
    np.testing.assert_array_equal(line.get_ydata(), np.asarray(y))


def export(fig, output_dir, target, title):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # A fixed canvas makes all four PDFs reproducible and keeps external keys
    # inside the media box. Check complete artist bounds before saving.
    bounds = fig.get_tightbbox(renderer)
    canvas = fig.bbox_inches
    assert bounds.x0 >= 0 and bounds.y0 >= 0
    assert bounds.x1 <= canvas.x1 and bounds.y1 <= canvas.y1
    paths = [output_dir / f"{target}.{suffix}" for suffix in ("pdf", "png")]
    fig.savefig(paths[0], metadata={"Title": title, "Subject": "Presentation-only redraw from frozen aggregate coordinates",
                                  "CreationDate": None, "ModDate": None})
    fig.savefig(paths[1], dpi=200)
    plt.close(fig)
    return {p.name: digest(p) for p in paths}


def missing_runs(x, missing):
    start = None
    for i, value in enumerate(missing):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(missing) - 1):
            yield float(x[start]), float(x[i if value else i - 1])
            start = None


def detection_panel(ax, tests, frontier, panel, households):
    x = frontier["x"].to_numpy(float)
    y = frontier["frontier_bp"].to_numpy(float)
    missing = ~np.isfinite(y)
    ax.set(xscale="log", yscale="log", xlim=(x.min(), 1), ylim=(5, 5000))
    # First crossings need not delimit a sustained-power region: no fill above the curve.
    line, = ax.plot(x, y, color="black", linewidth=1.4, zorder=3)
    assert_line(line, x, y)
    marker_values = []
    for test, group in tests.groupby("test"):
        for row in group[group["status"].isin(REACHED)].itertuples(index=False):
            j = int(np.argmin(np.abs(np.log(x) - np.log(row.fraction))))
            if frontier["best_test_id"].iloc[j] == test and np.isfinite(row.effect_bp_at_target):
                marker, = ax.plot(row.fraction, row.effect_bp_at_target, marker=MARKERS[test], markersize=4.5,
                                  markerfacecolor="white", markeredgecolor=COLORS[test], markeredgewidth=1.2,
                                  linestyle="none", zorder=4, clip_on=False)
                assert_line(marker, [row.fraction], [row.effect_bp_at_target])
                marker_values.append([test, float(row.fraction), float(row.effect_bp_at_target)])
    runs = list(missing_runs(x, missing))
    for first, last in runs:
        ax.axvspan(first, last, facecolor="white", hatch="////", edgecolor="#bcbcbc", linewidth=0, zorder=0)
    if panel == "a":
        ticks = [1 / households] + [v for v in (.03, .1, .3) if v > 1.6 / households] + [1.]
        labels = [f"1/{households:.0f}"] + [f"{v:.0%}" for v in ticks[1:-1]] + ["100%"]
        ax.set_title("(a)", loc="left", fontsize=FS, pad=5)
        ax.set_xlabel("Share of households receiving the gain", fontsize=FS, labelpad=4)
    else:
        ticks, labels = [.01, .03, .1, .3, 1.], ["1%", "3%", "10%", "30%", "100%"]
        ax.set_title("(b)", loc="left", fontsize=FS, pad=5)
        ax.set_xlabel("Share of purchases receiving the gain", fontsize=FS, labelpad=4)
    ax.set_xticks(ticks, labels)
    ax.set_yticks([10, 30, 100, 300, 1000, 3000], ["10", "30", "100", "300", "1,000", "3,000"])
    ax.minorticks_off()
    ax.tick_params(length=2.5, pad=2, labelsize=FS)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    finite_x = x[~missing]
    return {"frontier_rows": len(x), "missing_frontier_values": int(missing.sum()),
            "first_finite": [float(finite_x[0]), float(y[~missing][0])],
            "last_finite": [float(finite_x[-1]), float(y[~missing][-1])],
            "marker_values": marker_values, "x": numeric_array(x), "y": numeric_array(y),
            "best_test_id": frontier["best_test_id"].where(frontier["best_test_id"].notna(), None).tolist(),
            "nr_spans": runs, "exact_line_and_marker_equality": True}


def dense_thresholds(input_dir, stem, common):
    dense = pd.read_csv(input_dir / f"{stem}_dense.csv")
    values = dense.loc[dense["status"].isin(REACHED), "effect_bp_at_target"].dropna()
    assert len(values) == 2
    note = None
    if not common:
        all_rows = pd.read_csv(input_dir / "frontiers_80.csv")
        rows = all_rows[(all_rows["clock"] == "P") & (all_rows["test"] == "T3")
                        & (all_rows["alternative"] == "dense") & (all_rows["variant"] == "wins")
                        & ~all_rows["superseded"].astype(bool) & all_rows["status"].isin(REACHED)]
        assert len(rows) == 1
        note = float(rows["effect_bp_at_target"].iloc[0])
    return float(values.min()), float(values.max()), note


def detection_figure(output_dir, spec, old_validation):
    stem, old, target, clock, common = spec
    input_dir = SOURCE / "data/v4/calibration"
    tests = {p: pd.read_csv(input_dir / f"{stem}_{p}_tests.csv") for p in ("a", "b")}
    frontiers = {p: pd.read_csv(input_dir / f"{stem}_{p}_frontier.csv") for p in ("a", "b")}
    count_rows = tests["a"] if common else tests["a"][tests["a"]["analysis_set"] == "n20"]
    households = float(count_rows["H_mean"].dropna().iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(6.70, 4.28), sharey=True)
    fig.subplots_adjust(left=.12, right=.975, bottom=.24, top=.94, wspace=.16)
    result = {"source_stem": stem, "common_sample": common, "clock": clock,
              "households_for_first_tick": households}
    for panel, ax in zip(("a", "b"), axes, strict=True):
        result[panel] = detection_panel(ax, tests[panel], frontiers[panel], panel, households)
        for key, old_value in old_validation[old][panel].items():
            assert same_value(result[panel][key], old_value), (target, panel, key)
    axes[0].set_ylabel("Gain per purchase at 80% power\n(bp, log scale)", fontsize=FS, labelpad=3)
    lo, hi, note = dense_thresholds(input_dir, stem, common)
    present = {m[0] for p in ("a", "b") for m in result[p]["marker_values"]}
    handles = [Line2D([], [], marker=MARKERS[t], markerfacecolor="white", markeredgecolor=COLORS[t],
                      markeredgewidth=1.2, linestyle="none", label=NAMES[t], markersize=4.5)
               for t in MARKERS if t in present]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .10), ncol=3, frameon=False,
               fontsize=FS,
               handlelength=1, columnspacing=1.5)
    fig.legend(handles=[Line2D([], [], color="black", linewidth=1.4, label="Smallest gain across tests"),
                        Patch(facecolor="white", hatch="////", edgecolor="#bcbcbc", label="80% not reached")],
               loc="lower center", bbox_to_anchor=(.54, .035), ncol=2, frameon=False, fontsize=FS, columnspacing=1.5)
    result["dense_covariance_threshold_range_bp"] = [lo, hi]
    result["dense_maximum_threshold_bp"] = note
    for key in ("households_for_first_tick", "dense_covariance_threshold_range_bp", "dense_maximum_threshold_bp"):
        assert result[key] == old_validation[old][key]
    result["matches_original_validation"] = True
    result["outputs"] = export(fig, output_dir, target, f"First 80% power crossings: {clock.lower()}")
    return target, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PAPER / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    family, font_records = configure_palatino_font()
    plt.rcParams.update({"font.family": "serif", "font.serif": [family], "font.size": FS, "axes.linewidth": .6,
                         "hatch.linewidth": .4, "pdf.fonttype": 42, "mathtext.default": "it", "mathtext.fontset": "custom",
                         "mathtext.rm": family, "mathtext.it": family+":italic", "mathtext.bf": family+":bold"})
    old_validation = json.loads((PAPER / "figures/detection_figures_validation.json").read_text())
    for spec in FIGURES:
        detection_figure(args.output_dir, spec, old_validation)
    print("Saved Figure G1; every plotted coordinate verified.")


if __name__ == "__main__":
    main()
