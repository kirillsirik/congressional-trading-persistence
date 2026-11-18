"""Figure 2 v3 — the detection region with ONE frontier per panel.

Panel (a): share of households informed (x from 1/H to 100%, log), y = effect per informed household in bp per
20 trading days (log). Panel (b): share of purchases informed (0.5% to 100%), y = effect per informed purchase.
At each x the frontier is the lowest 80%-power effect over the correctly sized tests — block-clustered mean
test, placebo-test tau2 (event-share weights), household bootstrap tau2, best-household test, tail test (z > 2)
— each judged on its plan outcome variant and with informed households drawn from its own analysis set
(tau2 tests: the persistence set; best-household test: households with >= 20 purchases; mean and tail tests:
every household). Between measured shares a test's frontier is a straight segment in log-log space. The
region above the frontier is shaded ('detectable at 80% power'); published average effects sit at x = 100%, while a
published effect per informed trade sits inside panel (b) at the share of trades its paper reports informed.

CLI: python -m trading_on_disclosure_v4.calibration.figure2_v3 [--clock T|P] [--out <stem>]
Outputs: <stem>.{pdf,png}, <stem>_a_tests.csv, <stem>_a_frontier.csv, <stem>_b_tests.csv, <stem>_b_frontier.csv,
<stem>_published.csv, <stem>_dense.csv (the two tau2 tests the dense bar of panel (a) spans) and
data/v4/calibration/figure2_caption.txt.
Drawn at the printed width (\textwidth = 6.5 in) so that every text element is >= FS = 8 pt on the page.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_on_disclosure_v4.calibration.figure2 import (  # noqa: E402
    CALIB_DIR,
    LIT_CSV,
    load_inputs,
)

TESTS = ("T1_blk20_2s", "T2_I1cal_ES", "T2_I2b_ES", "T3", "T4_z2")
PLAIN = {"T1_blk20_2s": "block-clustered mean test", "T2_I1cal_ES": "placebo-test tau2", "T2_I2b_ES": "household bootstrap tau2",
         "T3": "best-household test", "T4_z2": "tail test"}
# analysis set the informed households are drawn from (panel a) and the outcome variant each test is judged on
BASE_OF = {"T1_blk20_2s": "all", "T2_I1cal_ES": "R1", "T2_I2b_ES": "R1", "T3": "n20", "T4_z2": "all"}
VARIANT_A = {"T1_blk20_2s": ("raw", "wins"), "T2_I1cal_ES": ("wins",), "T2_I2b_ES": ("wins",), "T3": ("wins",), "T4_z2": ("raw", "wins")}
MARKERS = {"T1_blk20_2s": "o", "T2_I1cal_ES": "s", "T2_I2b_ES": "D", "T3": "^", "T4_z2": "v"}
# The points shown, one published effect per paper. Every coordinate is read from
# data/v4/literature/detection_map_points.csv (y in bp per 20 trading days, x the share of purchases informed
# where the paper reports one and NA otherwise); this module supplies only the labels.
PUBLISHED = {"ZIO2004-1": "Ziobrowski 2004, Senate buys", "ZIO2011-1": "Ziobrowski 2011, House buys",
             "JMZ2003-2": "Jeng et al. 2003, insiders", "WZ2025-1": "Wei-Zhou 2025, leaders (250 d.)",
             "WZ2025-2": "Wei-Zhou 2025, leaders (10 d.)", "CS2026-5": "Chen-Sacerdote 2026, value-wt.",
             "BCC2025-5": "Blonien et al. 2025, per informed trade"}
# shorter labels for the figure itself (the CSV keeps the full PUBLISHED labels)
SHORT_LABEL = {"ZIO2004-1": "Ziobrowski 2004", "ZIO2011-1": "Ziobrowski 2011", "JMZ2003-2": "Jeng et al. 2003",
               "WZ2025-1": "Wei-Zhou 2025, 250 d.", "WZ2025-2": "Wei-Zhou 2025, 10 d.", "CS2026-5": "Chen-Sacerdote 2026",
               "BCC2025-5": "Blonien et al. 2025"}
OMITTED = "Eggers and Hainmueller (2013), Belmont et al. (2020) and the remaining Chen and Sacerdote (2026) estimates"
# leadership premia (leaders vs matched non-leaders), not average effects: flagged next to the points
SCALE_ONLY = {"WZ2025-1": "leaders only, shown for scale", "WZ2025-2": "leaders only, shown for scale"}
# --- the common-sample version: every test judged on the persistence households ---
COMMON_FRONTIERS = CALIB_DIR / "frontiers_80_common.csv"
COMMON_SIZE = CALIB_DIR / "common_sample_size.csv"
COMMON_SAMPLE = "persistence_households_purchases"
COMMON_BASE = "R1"          # one analysis set for every test
COMMON_VARIANT = ("wins",)  # one outcome variant for every test
# the dense runs are on the full panel, where only the tau2 tests read the persistence households; the dense
# bar of either version therefore spans those two tests, with the best-household test noted beside it
DENSE_BAR_TESTS = ("T2_I1cal_ES", "T2_I2b_ES")
DENSE_NOTE_TEST = "T3"
ENVELOPE_NAME = "best-available-test upper bound on detectability"  # the common-sample version
ENVELOPE_NAME_OWN = "best available test on the record"  # the own-sample version, which is Figure 2

N_PLACEBO = 500
Y_LIM = (5.0, 5000.0)
N_FINE = 400
FS = 8.0  # minimum font size as printed: the figure is drawn at its printed width (\textwidth = 6.5 in)
FIG_SIZE = (6.5, 3.6)
DENSE_X = 0.84  # share at which the dense-dispersion bracket is drawn (inside the published points at 100%)
LABEL_X_MAX = 0.8  # test names anchored no further right than this (clear of the published points at 100%)
COLUMN_X = 1.19  # the published-effect label column, in axes fractions of panel (b): 0.33 in outside the frame, left-aligned
COLUMN_TOP = 0.72  # labels stay below this axes fraction: the four-line column header sits above
LINE_GAP_PT = 1.5 * FS  # one label per line: minimum vertical distance between label baselines (points)
INTERIOR_WRAP = 15  # characters per line of a label placed inside a panel, next to its point


def _reached(g: pd.DataFrame) -> pd.DataFrame:
    return g[g["status"].isin(["interpolated", "at_grid_min"])]


def _rows(fr: pd.DataFrame, clock: str, test: str, alternative: str, base: str, selection: str, variants) -> pd.DataFrame:
    sub = fr[(fr["clock"] == clock) & (fr["test"] == test) & (fr["alternative"] == alternative) & (fr["base_set"] == base)
             & (fr["selection"] == selection) & (~fr["superseded"].astype(bool)) & (fr["variant"].isin(variants))]
    if test.startswith("T1") and "t1_on_raw_y" in sub.columns:
        sub = sub[sub["t1_on_raw_y"].astype("boolean").fillna(False).astype(bool)]
    # one row per share: prefer the plan variant listed first
    sub = sub.assign(_pref=sub["variant"].map({v: i for i, v in enumerate(variants)}))
    sub = sub.sort_values(["fraction", "_pref"]).drop_duplicates("fraction", keep="first").drop(columns="_pref")
    return sub


def panel_tests(fr: pd.DataFrame, clock: str, panel: str, common: bool = False) -> pd.DataFrame:
    """Frontier grid rows of every test of one panel. With `common`, every test is read from the common-sample
    table (one analysis set, one outcome variant); otherwise each test keeps its own set and plan variant."""
    parts = []
    for t in TESTS:
        if common:
            alt = "sparse_households_k" if panel == "a" else "sparse_trades"
            g = _rows(fr, clock, t, alt, COMMON_BASE, "uniform" if panel == "a" else "common", COMMON_VARIANT)
        elif panel == "a":
            base = BASE_OF[t]
            share = _rows(fr, clock, t, "sparse_households", base, "uniform", VARIANT_A[t]) if base == "all" else fr.iloc[0:0]
            kk = _rows(fr, clock, t, "sparse_households_k", base, "uniform", VARIANT_A[t])
            g = pd.concat([share, kk], ignore_index=True)
        else:
            g = _rows(fr, clock, t, "sparse_trades", "all", "common", ("raw",))
        g = g.sort_values("fraction").drop_duplicates("fraction", keep="last")
        aset = COMMON_BASE if common else (BASE_OF[t] if panel == "a" else "all")
        parts.append(g.assign(test_name=PLAIN[t], analysis_set=aset))
    out = pd.concat(parts, ignore_index=True)
    keep = ["test", "test_name", "analysis_set", "fraction", "k", "H_mean", "variant", "effect_bp_at_target", "effect_se_bp", "status",
            "max_power_on_grid", "n_panels", "grid", "panel_set", "sample"]
    return out[[c for c in keep if c in out.columns]].sort_values(["test", "fraction"]).reset_index(drop=True)


def envelope(tests: pd.DataFrame, x_min: float, x_max: float = 1.0, n: int = N_FINE) -> pd.DataFrame:
    """Lowest 80%-power effect over tests on a fine log grid; log-log segments between adjacent measured shares
    that both reach 80%; undefined (NaN) where no test reaches."""
    xs = np.exp(np.linspace(np.log(x_min), np.log(x_max), n))
    best = np.full(n, np.inf)
    who = np.array([""] * n, dtype=object)
    for t, g in tests.groupby("test"):
        g = g.sort_values("fraction")
        x, e, ok = g["fraction"].to_numpy(float), g["effect_bp_at_target"].to_numpy(float), _reached(g).index
        okm = g.index.isin(ok) & np.isfinite(e)
        for i in range(len(g) - 1):
            if okm[i] and okm[i + 1]:
                m = (xs >= x[i]) & (xs <= x[i + 1])
                val = np.exp(np.interp(np.log(xs[m]), [np.log(x[i]), np.log(x[i + 1])], [np.log(e[i]), np.log(e[i + 1])]))
                better = val < best[m]
                idx = np.flatnonzero(m)[better]
                best[idx], who[idx] = val[better], t
        # an isolated reached point (neighbours not reached) still counts at its own share
        for i in range(len(g)):
            if okm[i]:
                j = int(np.argmin(np.abs(np.log(xs) - np.log(x[i]))))
                if e[i] < best[j]:
                    best[j], who[j] = e[i], t
    best[~np.isfinite(best)] = np.nan
    return pd.DataFrame({"x": xs, "frontier_bp": best, "best_test": [PLAIN.get(w, "") for w in who], "best_test_id": who})


def published_points(path: Path = LIT_CSV) -> pd.DataFrame:
    """The published effects shown, read from the literature points file. `x` is the share of purchases the
    paper reports informed, NaN for a pooled average, which is an effect on every purchase and is drawn at the
    right edge of both panels; a point with an `x` is drawn inside panel (b) at that share."""
    lit = pd.read_csv(path)
    lit = lit[lit["claim_id"].isin(PUBLISHED)].copy()
    lit["label"] = lit["claim_id"].map(PUBLISHED)
    lit["y"] = lit["y"].astype(float)
    lit["x"] = pd.to_numeric(lit["x"], errors="coerce")
    lit["note"] = lit["claim_id"].map(SCALE_ONLY).fillna("")
    cols = ["claim_id", "label", "x", "y", "reported_horizon_sessions", "benchmark", "note"]
    return lit[cols].sort_values("y").reset_index(drop=True)


def edge_points(pub: pd.DataFrame) -> pd.DataFrame:
    """Pooled averages: an effect on every purchase, drawn at x = 100% of both panels and labelled in the column."""
    return pub[pub["x"].isna()].reset_index(drop=True)


def interior_points(pub: pd.DataFrame) -> pd.DataFrame:
    """Effects per informed unit at the share of purchases the paper reports informed: drawn inside panel (b)."""
    return pub[pub["x"].notna()].reset_index(drop=True)


def _spread(ys: np.ndarray, gap: float, lo: float, hi: float) -> np.ndarray:
    """Sorted positions at least `gap` apart inside [lo, hi], each as close as possible to its input (ys sorted ascending)."""
    out = np.array(ys, dtype=float)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    out = np.minimum(out, hi - gap * np.arange(len(out) - 1, -1, -1))  # push down from the top where the column overflows
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(out[i], out[i + 1] - gap)
    return np.maximum(out, lo + gap * np.arange(len(out)))


def _runs(env: pd.DataFrame) -> list[tuple[str, float, float]]:
    """(best test, x_start, x_end) of each stretch where one test is best (fine-grid resolution)."""
    d = env.dropna(subset=["frontier_bp"])
    runs, cur = [], None
    for x, t in zip(d["x"], d["best_test_id"], strict=True):
        if cur is None or t != cur[0]:
            if cur is not None:
                runs.append(tuple(cur))
            cur = [t, x, x]
        cur[2] = x
    if cur is not None:
        runs.append(tuple(cur))
    return runs


def _contiguous(xs: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """(x_start, x_end) of each contiguous run of True in mask."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            runs.append((float(xs[start]), float(xs[i if m else i - 1])))
            start = None
    return runs


def _label_runs(ax, env: pd.DataFrame, max_runs: int = 3) -> list:
    """Name the test that is best on each of the longest stretches (at most `max_runs`). The frontier falls to the right,
    so a name sits clear of it either below the line and right-aligned at the end of its stretch or above the line and
    left-aligned at its start; the first placement that stays inside the frame and clear of the text already drawn wins.
    Where neither is clear the name is pushed down past the names already placed and joined to its stretch by a leader."""
    runs = sorted(_runs(env), key=lambda r: -(np.log(r[2]) - np.log(r[1])))[:max_runs]
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    frame = ax.get_window_extent(rend)
    drawn = []
    boxes = [t.get_window_extent(rend) for t in ax.texts if t.get_text().strip()]

    def at(x: float) -> tuple[float, float]:
        j = int(np.argmin(np.abs(np.log(env["x"]) - np.log(x))))
        return x, float(env["frontier_bp"].iloc[j])

    def place(test: str, xy, offset, above: bool, leader: bool):
        kw = {"fontsize": FS, "ha": "left" if above else "right", "va": "bottom" if above else "top",
              "color": "#222222", "zorder": 5}
        if leader:
            kw["arrowprops"] = {"arrowstyle": "-", "color": "#888888", "lw": 0.5, "shrinkA": 1.5, "shrinkB": 1.5}
        return ax.annotate(PLAIN[test], xy=xy, xytext=offset, textcoords="offset points", **kw)

    def box(ann):  # one clear line of air above and below each name
        return ann.get_window_extent(rend).expanded(1.05, 2.2)

    def clear(bb) -> bool:
        inside = bb.x0 >= frame.x0 - 1 and bb.x1 <= frame.x1 + 1 and bb.y0 >= frame.y0 - 1 and bb.y1 <= frame.y1 + 1
        return inside and not any(bb.overlaps(b) for b in boxes)

    for t, x0, x1 in runs:
        if x1 / x0 < 1.6 and len(runs) > 2:
            continue
        end = at(min(x1, LABEL_X_MAX))
        chosen = None
        for xy, off, above in ((end, (0, -6), False), (at(x0), (2, 4), True)):
            ann = place(t, xy, off, above, False)
            if clear(box(ann)):
                chosen = ann
                break
            ann.remove()
        if chosen is None:
            chosen = place(t, end, (0, -6), False, True)
            bb = box(chosen)
            drop = max([bb.y1 - b.y0 for b in boxes if bb.overlaps(b)], default=0.0)
            chosen.set_position((0, -6 - drop))
        drawn.append(chosen)
        boxes.append(box(chosen))
    return drawn


def _label_boxes(fig, names: dict) -> pd.DataFrame:
    """Where the test names ended up, in display coordinates: one row per name (for the layout test)."""
    from matplotlib.text import Text

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    rows = [{"panel": panel, "text": a.get_text(),  # the words alone: an annotation's own extent would include its leader
             **dict(zip(("x0", "y0", "x1", "y1"), Text.get_window_extent(a, rend).extents, strict=True))}
            for panel, anns in names.items() for a in anns]
    return pd.DataFrame(rows, columns=["panel", "text", "x0", "y0", "x1", "y1"])


def _text_boxes(fig, axes: dict) -> pd.DataFrame:
    """Every non-empty text of each panel in display coordinates, with that panel's frame (for the layout test)."""
    from matplotlib.text import Text

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    rows = []
    for panel, ax in axes.items():
        f = ax.get_window_extent(rend)
        for a in ax.texts:
            if a.get_text().strip():
                rows.append({"panel": panel, "text": a.get_text(),
                             **dict(zip(("x0", "y0", "x1", "y1"), Text.get_window_extent(a, rend).extents, strict=True)),
                             "frame_x0": f.x0, "frame_y0": f.y0, "frame_x1": f.x1, "frame_y1": f.y1})
    return pd.DataFrame(rows, columns=["panel", "text", "x0", "y0", "x1", "y1", "frame_x0", "frame_y0", "frame_x1", "frame_y1"])


def _draw_panel(ax, tests: pd.DataFrame, env: pd.DataFrame, title: str, xlabel: str, xticks: list[float], xlabels: list[str]) -> list:
    d = env.dropna(subset=["frontier_bp"])
    ax.fill_between(d["x"], d["frontier_bp"], Y_LIM[1], color="#dcdcdc", lw=0, zorder=1)
    ax.plot(d["x"], d["frontier_bp"], color="black", lw=1.4, zorder=3)
    # measured grid points of the test that is best there
    for t, g in tests.groupby("test"):
        g = _reached(g)
        for x, e in zip(g["fraction"], g["effect_bp_at_target"], strict=True):
            j = int(np.argmin(np.abs(np.log(env["x"]) - np.log(x))))
            if env["best_test_id"].iloc[j] == t and np.isfinite(e):
                ax.plot(x, e, marker=MARKERS[t], ms=4, mfc="white", mec="black", mew=0.9, ls="none", zorder=4)
    # shares where no test reaches 80% within the effect grid (the caption explains the hatching; a note only where it fits)
    miss = env["frontier_bp"].isna().to_numpy()
    for x0, x1 in _contiguous(env["x"].to_numpy(), miss):
        ax.fill_between([x0, x1], Y_LIM[0], Y_LIM[1], facecolor="white", hatch="////", edgecolor="#c8c8c8", lw=0, zorder=0)
        if x1 / x0 > 4.0:
            ax.text(np.sqrt(x0 * x1), Y_LIM[0] * 1.3, "no test\nreaches\n80%\npower", fontsize=FS, ha="center", va="bottom", color="#555555")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(env["x"].min(), 1.0)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=FS)
    ax.set_yticks([10, 30, 100, 300, 1000, 3000])
    ax.set_yticklabels(["10", "30", "100", "300", "1,000", "3,000"], fontsize=FS)
    ax.minorticks_off()
    ax.tick_params(length=2.5, pad=2)
    ax.set_title(title, fontsize=FS, loc="left", pad=4)
    ax.set_xlabel(xlabel, fontsize=FS, labelpad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # region labels
    d_lo = d[(d["x"] > d["x"].quantile(0.35)) & (d["x"] < d["x"].quantile(0.65))]
    ax.text(0.98, Y_LIM[1] / 1.2, "detectable\nat 80% power", fontsize=FS, ha="right", va="top", color="#333333")
    ax.text(np.exp(np.log(d_lo["x"]).mean()), Y_LIM[0] * 1.3, "not detectable", fontsize=FS, ha="center", va="bottom", color="#333333")
    # which test is best, once per stretch: last, so that the scales, the limits and the region labels are already set
    return _label_runs(ax, env)


def label_column(pub: pd.DataFrame, ax, top: float = COLUMN_TOP, gap_pt: float = LINE_GAP_PT) -> pd.DataFrame:
    """Where each published effect is labelled: one line per point in a column outside the frame of `ax`, sorted by value,
    lines at least `gap_pt` apart, the scale-only note on its own line between the labels it belongs to. Columns: claim_id
    (empty for the note line), text, y (bp, NaN for the note), point_frac (the point's axes-fraction height), label_frac."""
    pub = pub.sort_values("y").reset_index(drop=True)
    ylo, yhi = np.log(Y_LIM)
    frac = (np.log(pub["y"].to_numpy(float)) - ylo) / (yhi - ylo)
    rows = []
    noted = pub.index[pub["note"] != ""].tolist()
    for i, r in pub.iterrows():
        rows.append({"claim_id": r["claim_id"], "text": f"{SHORT_LABEL[r['claim_id']]}: {r['y']:.0f} bp", "y": r["y"], "point_frac": frac[i]})
        if noted and i == noted[0]:  # the note sits between the first two noted labels (or under a single one)
            rows.append({"claim_id": "", "text": pub.loc[i, "note"], "y": np.nan, "point_frac": frac[i] + 1e-9})
    lab = pd.DataFrame(rows)
    ax_h_pt = ax.get_position().height * ax.figure.get_size_inches()[1] * 72
    lab["label_frac"] = _spread(lab["point_frac"].to_numpy(), gap_pt / ax_h_pt, 0.0, top)
    return lab


def _draw_interior(ax, pts: pd.DataFrame, env: pd.DataFrame | None = None) -> list:
    """Published effects that carry a share of informed purchases, drawn inside panel (b) at (share, effect) and
    labelled in place, the label wrapped to `INTERIOR_WRAP` characters.

    A placement must stay inside the frame and clear of the text already drawn (the frontier's test names, the
    region labels and the label column) and of the frontier itself (`env`). The label takes the first side of its
    dot that is clear; failing that a corner of the panel, joined to the dot by a leader line; failing that it is
    pushed down the panel from the dot until it is clear."""
    from matplotlib.text import Text

    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    frame = ax.get_window_extent(rend)
    boxes = [t.get_window_extent(rend) for t in ax.texts if t.get_text().strip()]
    curve = np.empty((0, 2))
    if env is not None:
        d = env.dropna(subset=["frontier_bp"])
        curve = ax.transData.transform(np.c_[d["x"].to_numpy(float), d["frontier_bp"].to_numpy(float)])
    leader = {"arrowstyle": "-", "color": "#888888", "lw": 0.5, "shrinkA": 1.5, "shrinkB": 2.5}
    drawn = []

    def box(ann):  # the words alone: an annotation's own extent would include its leader line
        ann.update_positions(rend)  # a moved annotation only knows where it is once its positions are updated
        return Text.get_window_extent(ann, rend).expanded(1.05, 1.4)

    def clear(bb) -> bool:
        inside = bb.x0 >= frame.x0 - 1 and bb.x1 <= frame.x1 + 1 and bb.y0 >= frame.y0 - 1 and bb.y1 <= frame.y1 + 1
        on_curve = len(curve) and np.any((curve[:, 0] >= bb.x0) & (curve[:, 0] <= bb.x1)
                                         & (curve[:, 1] >= bb.y0) & (curve[:, 1] <= bb.y1))
        return inside and not on_curve and not any(bb.overlaps(b) for b in boxes)

    for r in pts.itertuples(index=False):
        ax.plot([r.x], [r.y], marker="o", ms=3.6, mfc="black", mec="black", ls="none", zorder=6)
        text = textwrap.fill(f"{SHORT_LABEL[r.claim_id]}: {r.y:,.0f} bp", width=INTERIOR_WRAP)
        kw = {"fontsize": FS, "color": "#222222", "zorder": 7}
        chosen = None
        # beside the dot, then anchored in a corner of the panel with a leader back to it
        for off, ha, va in (((5, -6), "left", "top"), ((5, 5), "left", "bottom"), ((-5, -6), "right", "top"),
                            ((-5, 5), "right", "bottom")):
            ann = ax.annotate(text, xy=(r.x, r.y), xytext=off, textcoords="offset points", ha=ha, va=va, **kw)
            if clear(box(ann)):
                chosen = ann
                break
            ann.remove()
        for corner, ha, va in (((0.02, 0.98), "left", "top"), ((0.02, 0.02), "left", "bottom"),
                               ((0.98, 0.02), "right", "bottom")):
            if chosen is not None:
                break
            ann = ax.annotate(text, xy=(r.x, r.y), xytext=corner, textcoords="axes fraction", ha=ha, va=va,
                              arrowprops=leader, **kw)
            if clear(box(ann)):
                chosen = ann
                break
            ann.remove()
        if chosen is None:  # no side and no corner is clear: walk the label down the panel from its dot
            chosen = ax.annotate(text, xy=(r.x, r.y), xytext=(5, -6), textcoords="offset points",
                                 ha="left", va="top", arrowprops=leader, **kw)
            placed = False
            for dy in np.arange(-6.0, -(frame.y1 - frame.y0), -6.0):
                chosen.set_position((5, float(dy)))
                if clear(box(chosen)):
                    placed = True
                    break
            if not placed:  # nowhere in the panel is clear: keep the label beside its dot
                chosen.set_position((5, -6.0))
        boxes.append(box(chosen))
        drawn.append(chosen)
    return drawn


def _draw_published(axes, pub: pd.DataFrame, label_ax) -> pd.DataFrame:
    """Pooled published effects as dots at x = 100% of both panels, labelled in a column outside the right frame of
    `label_ax`, one per line, sorted by value, each joined to its dot by a leader line. A published effect that carries
    a share of informed purchases is not in the column: `_draw_interior` places it inside panel (b). Returns the column
    layout (see `label_column`)."""
    edge = edge_points(pub)
    for ax in axes:
        ax.plot(np.ones(len(edge)), edge["y"], marker="o", ms=3.6, mfc="black", mec="black", ls="none", clip_on=False, zorder=6)
    lab = label_column(edge, label_ax)
    for r in lab.itertuples(index=False):
        if r.claim_id:
            label_ax.annotate(r.text, xy=(1.0, r.point_frac), xytext=(COLUMN_X, r.label_frac), xycoords="axes fraction",
                              textcoords="axes fraction", fontsize=FS, color="#222222", va="center", ha="left", annotation_clip=False,
                              arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.5, "shrinkA": 0, "shrinkB": 2.5})
        else:
            label_ax.annotate(r.text, xy=(COLUMN_X, r.label_frac), xycoords="axes fraction", xytext=(6, 0), textcoords="offset points",
                              fontsize=FS, color="#555555", va="center", ha="left", annotation_clip=False)
    return lab


def dense_frontier(fr: pd.DataFrame, clock: str, tests=TESTS) -> pd.DataFrame:
    """80%-power cross-household SD tau of the dense alternative (theta_h ~ N(0, tau^2) for every household), one row per
    correctly sized test on its plan variant; NaN where the test does not reach 80% on the grid."""
    parts = []
    for t in tests:
        sub = fr[(fr["clock"] == clock) & (fr["test"] == t) & (fr["alternative"] == "dense") & (~fr["superseded"].astype(bool))
                 & (fr["variant"].isin(VARIANT_A[t]))]
        if t.startswith("T1") and "t1_on_raw_y" in sub.columns:
            sub = sub[sub["t1_on_raw_y"].astype("boolean").fillna(False).astype(bool)]
        sub = sub.assign(_pref=sub["variant"].map({v: i for i, v in enumerate(VARIANT_A[t])})).sort_values("_pref").head(1)
        parts.append(sub.assign(test_name=PLAIN[t]))
    out = pd.concat(parts, ignore_index=True)
    keep = ["test", "test_name", "variant", "effect_bp_at_target", "effect_se_bp", "status", "max_power_on_grid", "n_panels", "grid", "panel_set"]
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)


def dense_range(fr: pd.DataFrame, clock: str, tests=TESTS) -> tuple[float, float]:
    """(min, max) of the 80%-power tau over the correctly sized tests that reach it (dense alternative)."""
    d = dense_frontier(fr, clock, tests)
    v = _reached(d)["effect_bp_at_target"].dropna()
    return float(v.min()), float(v.max())


def _draw_dense_bracket(ax, lo: float, hi: float, note: str | None = None, x: float = DENSE_X) -> None:
    """Short vertical bar with end ticks at share x spanning [lo, hi] on the effect axis, labelled to its left.
    `note` adds short lines under the label, for a test whose dense tau is not inside the bar."""
    half = 0.04  # half-width of the end ticks in log10(x) units
    xl, xr = x * 10 ** (-half), x * 10 ** half
    ax.plot([x, x], [lo, hi], color="black", lw=1.4, solid_capstyle="butt", zorder=6)
    for y in (lo, hi):
        ax.plot([xl, xr], [y, y], color="black", lw=1.0, zorder=6)
    text = f"shared\ndispersion:\n{lo:,.0f}-{hi:,.0f} bp" + (f"\n{note}" if note else "")
    ax.annotate(text, xy=(x, hi), xytext=(xl, hi * 1.9), textcoords="data",
                fontsize=FS, ha="right", va="bottom", color="#222222", zorder=7,
                arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.5, "shrinkA": 1, "shrinkB": 1})


def make_figure_v3(fr: pd.DataFrame, clock: str = "T", out: Path | None = None, common: bool = False,
                   dense_fr: pd.DataFrame | None = None) -> dict:
    """Draw Figure 2. With `common`, `fr` is the common-sample frontier table (every test on the persistence
    households) and `dense_fr` supplies the dense-alternative rows, which are only run on the full panel."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.linewidth": 0.6, "hatch.linewidth": 0.4})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)
    # the axes shrink to leave 2.2 in to the right of panel (b) for the leaders and the label column; the layout is fixed
    # first because every text is placed against the printed panel (the line gap is measured in points against its height)
    fig.subplots_adjust(left=0.10, right=0.665, bottom=0.11, top=0.92, wspace=0.12)
    ta, tb = panel_tests(fr, clock, "a", common), panel_tests(fr, clock, "b", common)
    if common:
        H_n20 = H_r1 = H_all = float(ta["H_mean"].dropna().iloc[0])
    else:
        H_n20 = float(ta.loc[ta["analysis_set"] == "n20", "H_mean"].dropna().iloc[0])
        H_r1 = float(ta.loc[ta["analysis_set"] == "R1", "H_mean"].dropna().iloc[0])
        h_all = ta.loc[(ta["analysis_set"] == "all") & ta["H_mean"].notna(), "H_mean"]
        H_all = float(h_all.iloc[0]) if len(h_all) else np.nan
    ea = envelope(ta, x_min=1.0 / H_n20)
    eb = envelope(tb, x_min=float(tb["fraction"].min()))
    clock_word = {"T": "trade date", "P": "disclosure date"}[clock]
    ticks_a = [1 / H_n20] + [v for v in (0.03, 0.1, 0.3) if v > 1.6 / H_n20] + [1.0]
    labels_a = [f"1/{H_n20:.0f}"] + [f"{v:.0%}" for v in ticks_a[1:-1]] + ["100%"]
    names_a = _draw_panel(ax_a, ta, ea, "(a) Informed households", "share of households informed", ticks_a, labels_a)
    names_b = _draw_panel(ax_b, tb, eb, "(b) Informed purchases", "share of purchases informed",
                          [0.01, 0.03, 0.1, 0.3, 1.0], ["1%", "3%", "10%", "30%", "100%"])
    ax_a.set_ylabel("effect needed for 80% power\n(bp per 20 trading days, log scale)", fontsize=FS, labelpad=3)
    dfr = fr if dense_fr is None else dense_fr
    dense = dense_frontier(dfr, clock, DENSE_BAR_TESTS)
    lo, hi = dense_range(dfr, clock, DENSE_BAR_TESTS)
    dense_all = dense_frontier(dfr, clock, TESTS)
    # the best-household test reads its own sample, so on the own-sample figure its dense tau is noted beside the bar
    note_row, note = None, None
    if not common:
        reached = _reached(dense_all[dense_all["test"] == DENSE_NOTE_TEST]).dropna(subset=["effect_bp_at_target"])
        if len(reached):
            note_row = reached.iloc[0]
            note = f"({PLAIN[DENSE_NOTE_TEST].replace(' test', '')}\ntest: {note_row['effect_bp_at_target']:,.0f} bp)"
    _draw_dense_bracket(ax_a, lo, hi, note)
    pub = published_points()
    labels = _draw_published((ax_a, ax_b), pub, ax_b)
    ax_b.text(COLUMN_X, 0.98, "published average effects\n(dots at the right edge of\nboth panels; the dot inside\n(b) is per informed trade)",
              transform=ax_b.transAxes, fontsize=FS, color="#222222", ha="left", va="top")
    ax_b.annotate(clock_word, xy=(COLUMN_X, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", fontsize=FS,
                  color="#222222", ha="left", va="bottom", annotation_clip=False)
    # last, so that the in-place label can be kept clear of every other text of panel (b) and of the frontier
    interior = _draw_interior(ax_b, interior_points(pub), eb)
    # no footer note: what it said is in the caption (figure2_caption.txt); everything stays inside the 6.5 in canvas
    res = {"a_tests": ta, "a_frontier": ea, "b_tests": tb, "b_frontier": eb, "published": pub, "dense": dense, "labels": labels,
           "H": {"R1": H_r1, "n20": H_n20, "all": H_all}, "min_font_pt": min_font_size(fig), "common": common,
           "dense_all": dense_all, "dense_note": note_row,
           "dense_tests": list(DENSE_BAR_TESTS), "clock": clock, "test_labels": _label_boxes(fig, {"a": names_a, "b": names_b}),
           "interior_labels": _label_boxes(fig, {"b": interior}), "text_boxes": _text_boxes(fig, {"a": ax_a, "b": ax_b})}
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
        for k in ("a_tests", "a_frontier", "b_tests", "b_frontier", "published", "dense"):
            res[k].to_csv(out.parent / f"{out.stem}_{k}.csv", index=False)
    plt.close(fig)
    return res


def min_font_size(fig) -> float:
    """Smallest font size (pt) over every Text artist of the figure with non-empty text."""
    from matplotlib.text import Text

    return min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip())


def _r2(v: float) -> str:
    """Two significant figures with a thousands separator, e.g. 558 -> 560, 2398 -> 2,400, 29 -> 29."""
    m = 10 ** (int(np.floor(np.log10(v))) - 1)
    return f"{int(round(v / m) * m):,}"


def load_common(calib_dir: Path = CALIB_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The common-sample frontier table and the common-sample size table (one row per clock)."""
    return pd.read_csv(calib_dir / COMMON_FRONTIERS.name), pd.read_csv(calib_dir / COMMON_SIZE.name)


def _sample_clause(size: pd.DataFrame, clock: str) -> str:
    r = size.loc[size["clock"] == clock].iloc[0]
    return (f"{r['n_households_r1']:.0f} households that buy in both periods and their {r['n_events_mean']:,.0f} purchases on "
            f"average across the placebo datasets ({r['n_events_min']:,.0f}-{r['n_events_max']:,.0f})")


def caption(res_t: dict, res_p: dict | None = None, size: pd.DataFrame | None = None) -> str:
    """Caption of one Figure 2. `size` (the common-sample size table) is required for the common-sample version."""
    def edge(res, key):
        d = res[key].dropna(subset=["frontier_bp"])
        return d.iloc[-1]["frontier_bp"], d.iloc[0]["frontier_bp"], d.iloc[0]["x"]

    def dense_clause(res):
        """The two ends of the dense bar: the lowest and the highest tau over the tests the bar spans."""
        d = _reached(res["dense"]).dropna(subset=["effect_bp_at_target"]).sort_values("effect_bp_at_target")
        return d.iloc[0], d.iloc[-1]

    def dense_missed(res) -> str:
        """The tests of the figure that do not reach 80% power on the dense runs at the effects tried."""
        d = res.get("dense_all")
        if d is None:
            return ""
        miss = d.loc[~d.index.isin(_reached(d).index), "test_name"].tolist()
        return f"The {' and the '.join(miss)} do not reach 80% power on the dense runs at the effects tried. " if miss else ""

    def interior_clause(res) -> str:
        """One clause for each published effect that reports its own share of informed trades."""
        if res.get("published") is None:
            return ""
        ip = interior_points(res["published"])
        if not len(ip):
            return ""
        which = "; ".join(f"{SHORT_LABEL[r.claim_id]}, {_r2(r.y)} bp per informed trade at {r.x:.1%} of trades"
                          for r in ip.itertuples(index=False))
        return ("One published estimate is an effect per informed trade rather than an average and reports the share of trades "
                f"it applies to ({which}), so it is drawn inside panel (b) at that share instead of at the right edge. ")

    common = bool(res_t.get("common"))
    d_lo, d_hi = dense_clause(res_t)
    a_hi, a_lo, a_x0 = edge(res_t, "a_frontier")
    b_hi, b_lo, b_x0 = edge(res_t, "b_frontier")
    H = res_t["H"]
    if common:
        assert size is not None, "the common-sample caption needs common_sample_size.csv"
        sample = (f"All five tests are judged on one sample, the {_sample_clause(size, res_t['clock'])}, so the frontiers are "
                  f"comparable and the black envelope of the five test frontiers is a {ENVELOPE_NAME}: an effect below it was "
                  "missed by every test tried, not by one particular test. This version is the comparability check for Figure 2 "
                  "of the main text, which keeps each test on the sample that test reads. ")
        sets = ("Informed households are drawn from that same sample; open markers are measured shares, the segments between them "
                "are straight lines in log-log space. ")
        dense_lead = ("The short vertical bar inside the right edge of panel (a) marks the dense case, in which every household has "
                      "its own ability drawn from a normal distribution with standard deviation tau. The dense runs were made on the "
                      "full panel, where the two tau2 tests are the ones that read the persistence households: they detect tau from "
                      f"about {_r2(d_lo['effect_bp_at_target'])} bp ({d_lo['test_name']}) to about {_r2(d_hi['effect_bp_at_target'])} bp "
                      f"({d_hi['test_name']}) per 20 trading days. ")
    else:
        sample = ("Each frontier is one test on the sample that test reads: the block-clustered mean test and the tail test on all "
                  f"{H['all']:.0f} households and all their purchases, the best-household test on the {H['n20']:.0f} households with at "
                  f"least 20 purchases, and the placebo test and the household bootstrap of tau2 on the {H['R1']:.0f} persistence "
                  f"households. The black envelope is therefore the {ENVELOPE_NAME_OWN}: at each share it is the smallest effect any "
                  "test of the record detects. The samples differ, so the frontiers are not comparable with one another; the version "
                  "in which every test is judged on one common sample is the comparability check in the appendix. ")
        sets = ("Informed households are drawn from the test's own sample; open markers are measured shares, the segments between "
                "them are straight lines in log-log space. ")
        note = res_t.get("dense_note")
        dense_lead = ("The short vertical bar inside the right edge of panel (a) marks the dense case, in which every household has its "
                      "own ability drawn from a normal distribution with standard deviation tau. The bar spans the two tau2 tests, the "
                      f"tests that read the persistence households: from about {_r2(d_lo['effect_bp_at_target'])} bp "
                      f"({d_lo['test_name']}) to about {_r2(d_hi['effect_bp_at_target'])} bp ({d_hi['test_name']}) per 20 trading days. ")
        if note is not None:
            dense_lead += (f"The note beside the bar gives the {note['test_name']}, which reaches 80% power at about "
                           f"{_r2(note['effect_bp_at_target'])} bp on its own sample. ")
        dense_lead += dense_missed(res_t)
    where = "the public-clock version follows" if common else "the public-clock version is in the appendix"
    lines = [
        "Figure 2. What the disclosure record could have detected. Each panel shows the smallest effect that the best correctly "
        f"sized test detects with 80% power, that is, rejects in 80% of {N_PLACEBO} centered placebo datasets into which the effect "
        f"was injected (trade date; {where}). " + sample
        + "Panel (a): a share of households is informed and each informed household earns the effect on the vertical "
        "axis on every purchase; panel (b): a share of purchases is informed and each informed purchase earns the effect. The grey "
        "region above the frontier is detectable, the white region below is not; in the hatched columns no test reaches 80% power "
        "within the range of effects tried. Both axes are logarithmic and the horizontal axis "
        "runs to 100%, where every household (a) or every purchase (b) carries the same effect, which is an average effect; the "
        "published average effects for congressional and corporate-insider purchases are therefore placed at the right edge. The "
        "two Wei and Zhou (2025) points are premia earned by congressional leaders relative to matched non-leaders, not average "
        "effects; they are shown for scale only and labelled as such. " + interior_clause(res_t)
        + f"Reading: one informed household among {H['n20']:.0f} is visible only if it earns about {_r2(a_lo)} bp per purchase per 20 "
        f"trading days, while an effect shared by every household is visible from about {_r2(a_hi)} bp; one informed purchase in "
        f"{1 / b_x0:.0f}, the smallest share of purchases any test reaches, is visible only from about {_r2(b_lo)} bp, and an effect "
        f"on every purchase from about {_r2(b_hi)} bp. The tests are the "
        "block-clustered mean test, the placebo test and the household bootstrap for the persistence variance tau2 (event-share "
        "weights), the best-household test and the tail test; the name printed along the frontier is the test that is best there. "
        + sets + dense_lead
        + f"Published estimates that are zero or negative ({OMITTED}) are not shown.",
    ]
    if res_p is not None:
        pa_hi, pa_lo, _ = edge(res_p, "a_frontier")
        pb_hi, pb_lo, _ = edge(res_p, "b_frontier")
        p_lo, p_hi = dense_clause(res_p)
        extra = f", on the {_sample_clause(size, res_p['clock'])}" if common else ""
        _, _, pb_x0 = edge(res_p, "b_frontier")
        p_note = res_p.get("dense_note")
        p_dense = f"dense bar {_r2(p_lo['effect_bp_at_target'])}-{_r2(p_hi['effect_bp_at_target'])} bp"
        if p_note is not None:
            p_dense += f", {p_note['test_name']} {_r2(p_note['effect_bp_at_target'])} bp"
        lines.append(f"Disclosure date (appendix version{extra}): one household among {res_p['H']['n20']:.0f} from about {_r2(pa_lo)} bp, every "
                     f"household from about {_r2(pa_hi)} bp; one purchase in {1 / pb_x0:.0f} from about {_r2(pb_lo)} bp, every purchase from "
                     f"about {_r2(pb_hi)} bp; {p_dense}.")
    return "\n\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Figure 2 v3 (one frontier per panel)")
    ap.add_argument("--out", type=Path, default=CALIB_DIR / "figure2_detection_region_v3")
    ap.add_argument("--testspecific-suffix", default="_testspecific",
                    help="stem suffix of the main-text version in which each test keeps its own analysis set")
    args = ap.parse_args(argv)
    _, fr = load_inputs()
    common_fr, size = load_common()
    # appendix comparability check: every test on the common (persistence) sample. The file stems are unchanged
    # so that the paper's copies and macros keep their names.
    res_t = make_figure_v3(common_fr, "T", out=args.out, common=True, dense_fr=fr)
    res_p = make_figure_v3(common_fr, "P", out=args.out.parent / (args.out.stem + "_public"), common=True, dense_fr=fr)
    (args.out.parent / "figure2_caption.txt").write_text(caption(res_t, res_p, size))
    # main text: each test on the sample it reads
    ts = args.out.parent / (args.out.stem + args.testspecific_suffix)
    ts_t = make_figure_v3(fr, "T", out=ts)
    ts_p = make_figure_v3(fr, "P", out=ts.parent / (ts.stem + "_public"))
    (args.out.parent / "figure2_caption_testspecific.txt").write_text(caption(ts_t, ts_p))
    print(f"wrote {args.out}.pdf/.png and *_public.* (common sample, the comparability check), {ts}.pdf/.png and "
          "*_public.* (own sample, Figure 2); captions in figure2_caption.txt and figure2_caption_testspecific.txt")


if __name__ == "__main__":
    main()
