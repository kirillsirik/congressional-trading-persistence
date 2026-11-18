"""Figure 2 — the detection region (plan §9 Fig 2). Reusable, grayscale-readable.

Two panels: (a) sparse households, (b) sparse trades; x = fraction informed (log), y = effect
size in bp per 20 sessions (log); one 80%-power frontier per test (from `frontiers_80.csv`);
an inset with the dense-case power curves (from `power_surface.csv`). Published effect sizes
from `data/v4/literature/detection_map_points.csv` are drawn as horizontal reference lines
with right-margin labels (no paper reports the share of informed units, so they are lines,
not points — referee W MJ-07). Identity is carried by line style AND marker, never color alone.

CLI: python -m trading_on_disclosure_v4.calibration.figure2 [--clock T] [--variant raw]
     [--households uniform] [--trades common] [--out data/v4/calibration/figure2_detection_region]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
CALIB_DIR = REPO_ROOT / "data" / "v4" / "calibration"
LIT_CSV = REPO_ROOT / "data" / "v4" / "literature" / "detection_map_points.csv"

# test id -> (label, color, linestyle, marker); grays + one dark blue keep the figure readable in print
STYLES = {
    "T1": ("T1 mean (two-way CGM)", "#000000", "-", "o"),
    "T2_I1_ES": ("T2 var. component, ES (I1)", "#1f4e79", "--", "s"),
    "T2_I2b_ES": ("T2 var. component, ES (I2b)", "#1f4e79", (0, (1, 1)), "D"),
    "T1_blk20": ("T1 mean (CGM, 20-session date blocks)", "#000000", "-", "o"),
    "T2_I1cal_ES": ("T2 var. component, ES (I1, N2-calibrated)", "#1f4e79", "--", "s"),
    "T3": ("T3 max-t vs null critical value", "#555555", "-.", "^"),
    "T4_z2": ("T4 share z > 2 vs null", "#000000", (0, (3, 1, 1, 1)), "v"),
    "T4_z3": ("T4 share z > 3 vs null", "#888888", (0, (3, 1, 1, 1)), "x"),
    "T1_blk20_2s": ("T1 mean, household x 20-session-block CGM (two-sided)", "#000000", "-", "o"),
}
# Figure 2 v2: size-correct procedures only; per-panel test sets and the outcome variant each test is judged on
V2_TESTS_R1 = ("T1_blk20_2s", "T2_I1cal_ES", "T2_I2b_ES", "T4_z2", "T4_z3")
V2_TESTS_N20 = ("T1_blk20_2s", "T3", "T4_z2", "T4_z3")
V2_TESTS_TRADES = ("T1_blk20_2s", "T2_I1cal_ES", "T3", "T4_z2", "T4_z3")
V2_TESTS_DENSE = ("T1_blk20_2s", "T2_I1cal_ES", "T2_I2b_ES", "T3", "T4_z2")
V2_PLAN_VARIANT = {"T1_blk20_2s": "raw", "T2_I1cal_ES": "wins", "T2_I2b_ES": "wins", "T3": "wins", "T4_z2": "raw", "T4_z3": "raw"}
# one label per paper/finding at the right margin (every positive claim is still drawn as a line)
LIT_LABELLED = ("ZIO2004-1", "ZIO2011-1", "ZIO2011-4", "WZ2025-1", "WZ2025-2", "WZ2025-3", "CS2026-5", "JMZ2003-2")
DEFAULT_TESTS = ("T1", "T2_I1_ES", "T3", "T4_z2", "T4_z3")
DENSE_TESTS = ("T1", "T2_I1_ES", "T2_I2b_ES", "T3", "T4_z2")
CALIBRATED_TESTS = ("T1_blk20", "T2_I1cal_ES", "T3", "T4_z2", "T4_z3")
CALIBRATED_DENSE_TESTS = ("T1_blk20", "T2_I1cal_ES", "T2_I2b_ES", "T3", "T4_z2")
LIT_SHORT = {
    "ZIO2004-1": "Ziobrowski 04 Senate (tw)", "ZIO2004-3": "Ziobrowski 04 Senate (ew)",
    "ZIO2011-1": "Ziobrowski 11 House (tw)", "ZIO2011-2": "Ziobrowski 11 House (ew)",
    "ZIO2011-4": "Ziobrowski 11 House (alpha)", "EH2013-1": "Eggers-Hainmueller 13",
    "WZ2025-1": "Wei-Zhou 25 [0,250]", "WZ2025-2": "Wei-Zhou 25 [0,10]", "WZ2025-3": "Wei-Zhou 25 alpha",
    "CS2026-5": "Chen-Sacerdote 26 (vw)", "JMZ2003-1": "Jeng et al. 03 insiders", "JMZ2003-2": "Jeng et al. 03 insiders (m)",
}


def load_inputs(calib_dir: Path = CALIB_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(calib_dir / "power_surface.csv"), pd.read_csv(calib_dir / "frontiers_80.csv")


def literature_lines(path: Path = LIT_CSV, y_min: float = 1.0, y_max: float = np.inf) -> pd.DataFrame:
    """Rows with a positive effect inside [y_min, y_max] (log axes cannot show y <= 0)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["label", "y", "claim_id"])
    lit = pd.read_csv(path)
    lit = lit[pd.to_numeric(lit["y"], errors="coerce").notna()].copy()
    lit["y"] = lit["y"].astype(float)
    keep = lit[(lit["y"] >= y_min) & (lit["y"] <= y_max)].copy()
    keep["short"] = keep["claim_id"].map(LIT_SHORT).fillna(keep["label"])
    return keep.sort_values("y").reset_index(drop=True)


def frontier_lines(fr: pd.DataFrame, clock: str, variant: str, alternative: str, selection: str, tests) -> pd.DataFrame:
    """Frontier rows for one panel; 'not_reached' cells are kept with NaN effect for the CSV."""
    sub = fr[(fr["clock"] == clock) & (fr["variant"] == variant) & (fr["alternative"] == alternative)
             & (fr["selection"] == selection) & (fr["test"].isin(tests))].copy()
    return sub.sort_values(["test", "fraction"]).reset_index(drop=True)


def _plot_frontiers(ax, lines: pd.DataFrame, tests, y_top: float) -> None:
    for t in tests:
        g = lines[lines["test"] == t]
        if g.empty:
            continue
        label, color, ls, mk = STYLES[t]
        ok = g[g["status"].isin(["interpolated", "at_grid_min"])]
        ax.plot(ok["fraction"], ok["effect_bp_at_target"], color=color, linestyle=ls, marker=mk, ms=5.5,
                lw=1.4, mfc="white" if mk != "x" else color, mew=1.2, label=label)
        nr = g[g["status"] == "not_reached"]
        if len(nr):
            ax.plot(nr["fraction"], np.full(len(nr), y_top), color=color, linestyle="none", marker=mk, ms=5.5,
                    mfc=color, mew=1.2, clip_on=False)


def _label_positions(ys: np.ndarray, min_ratio: float = 1.28) -> np.ndarray:
    """Nudge right-margin labels apart on the log scale so they do not overprint."""
    out = np.array(sorted(ys), dtype=float)
    for i in range(1, len(out)):
        if out[i] < out[i - 1] * min_ratio:
            out[i] = out[i - 1] * min_ratio
    return out


def _draw_literature(axes, lit: pd.DataFrame, y_lim: tuple[float, float]) -> None:
    if lit.empty:
        return
    for ax in axes:
        for y in lit["y"]:
            ax.axhline(y, color="#bbbbbb", lw=0.6, linestyle=(0, (1, 2)), zorder=0)
    right = axes[-1]
    lab = lit[lit["claim_id"].isin(LIT_LABELLED)].sort_values("y") if "claim_id" in lit else lit.sort_values("y")
    pos = _label_positions(lab["y"].to_numpy())
    for (_y, short), yp in zip(lab[["y", "short"]].itertuples(index=False), pos, strict=True):
        if y_lim[0] <= yp <= y_lim[1]:
            right.annotate(short, xy=(1.0, yp), xycoords=("axes fraction", "data"), xytext=(6, 0),
                           textcoords="offset points", fontsize=5.5, color="#555555", va="center", ha="left",
                           annotation_clip=False)


def _dense_inset(ax, surf: pd.DataFrame, clock: str, variant: str, tests, box=(0.08, 0.08, 0.40, 0.36)) -> pd.DataFrame:
    ins = ax.inset_axes(list(box))
    d = surf[(surf["clock"] == clock) & (surf["variant"] == variant) & (surf["alternative"] == "dense")
             & (surf["test"].isin(tests)) & (surf["effect_bp"] > 0)]
    for t in tests:
        g = d[d["test"] == t].sort_values("effect_bp")
        if g.empty:
            continue
        label, color, ls, mk = STYLES[t]
        ins.plot(g["effect_bp"], g["power"], color=color, linestyle=ls, marker=mk, ms=3, lw=1.0, mfc="white" if mk != "x" else color)
    ins.axhline(0.8, color="#999999", lw=0.6)
    ins.set_xscale("log")
    ins.set_ylim(0, 1.02)
    ins.set_xticks([10, 25, 50, 100, 200])
    ins.set_xticklabels(["10", "25", "50", "100", "200"], fontsize=5.5)
    ins.tick_params(axis="y", labelsize=5.5)
    ins.set_xlabel("tau (bp / 20 sessions)", fontsize=6, labelpad=1)
    ins.set_ylabel("power", fontsize=6)
    ins.set_title("(c) dense: every household", fontsize=6.5, pad=2)
    ins.grid(True, color="#e6e6e6", lw=0.5)
    ins.set_facecolor("white")
    return d


def make_figure(
    surf: pd.DataFrame, fr: pd.DataFrame, clock: str = "T", variant: str = "raw", households: str = "uniform",
    trades: str = "common", tests=DEFAULT_TESTS, lit: pd.DataFrame | None = None, out: Path | None = None, dense_tests=DENSE_TESTS,
    y_lim: tuple[float, float] = (8.0, 6000.0), title: str | None = None, inset_panel: int = 1,
    inset_box=(0.13, 0.16, 0.40, 0.34), legend_loc: str = "lower left",
) -> tuple[plt.Figure, pd.DataFrame]:
    """Build Figure 2; write <out>.pdf, <out>.png and the CSV behind it when `out` is given."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.7), sharey=True)
    panels = (("sparse_households", households, "(a) Sparse households"), ("sparse_trades", trades, "(b) Sparse trades"))
    csv_rows = []
    for ax, (alt, sel, ttl) in zip(axes, panels, strict=True):
        lines = frontier_lines(fr, clock, variant, alt, sel, tests)
        _plot_frontiers(ax, lines, tests, y_lim[1])
        csv_rows.append(lines.assign(panel=alt))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(*y_lim)
        xs = sorted(lines["fraction"].dropna().unique())
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{100 * x:g}%" for x in xs])
        ax.minorticks_off()
        ax.grid(True, which="major", color="#e6e6e6", lw=0.5)
        ax.set_title(f"{ttl} [{sel} selection]", fontsize=7, loc="left")
        ax.set_xlabel("share of informed households (log scale)" if alt == "sparse_households" else "share of informed trades (log scale)")
        ax.tick_params(length=2.5, labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("effect per informed unit at 80% power (bp per 20 sessions, log scale)", fontsize=7)
    lit = literature_lines(y_min=y_lim[0], y_max=y_lim[1]) if lit is None else lit
    _draw_literature(axes, lit, y_lim)
    dense = _dense_inset(axes[inset_panel], surf, clock, variant, dense_tests, inset_box)
    axes[0].legend(loc=legend_loc, fontsize=6, frameon=False, handlelength=2.8)
    fig.suptitle(title or f"Detection region: smallest effect detected with 80% power (N2 placebo panels, {clock} clock, {variant} outcome)",
                 fontsize=8, y=0.995)
    fig.text(0.01, 0.01, "Dotted gray lines: published effect sizes (no paper reports an informed share). Filled markers at the top: "
             "80% power not reached on the grid.", fontsize=5.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 0.84, 0.97))
    behind = pd.concat(csv_rows, ignore_index=True)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=220)
        behind.to_csv(out.with_suffix(".csv"), index=False)
        dense.to_csv(out.parent / (out.stem + "_dense_inset.csv"), index=False)
        lit.assign(panel="reference_line").to_csv(out.parent / (out.stem + "_literature_lines.csv"), index=False)
    return fig, behind


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Figure 2 (detection region)")
    ap.add_argument("--clock", default="T")
    ap.add_argument("--variant", default="raw")
    ap.add_argument("--households", default="uniform", choices=("uniform", "activity"))
    ap.add_argument("--trades", default="common", choices=("common", "household"))
    ap.add_argument("--out", type=Path, default=CALIB_DIR / "figure2_detection_region")
    ap.add_argument("--calibrated", action="store_true", help="T1 with 20-session date blocks and T2 with the N2-calibrated critical value")
    args = ap.parse_args(argv)
    surf, fr = load_inputs()
    tests, dense = (CALIBRATED_TESTS, CALIBRATED_DENSE_TESTS) if args.calibrated else (DEFAULT_TESTS, DENSE_TESTS)
    make_figure(surf, fr, args.clock, args.variant, args.households, args.trades, tests=tests, dense_tests=dense, out=args.out)
    print(f"wrote {args.out}.pdf/.png/.csv")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- Figure 2 v2 

def _not_reached_strip(ax, lines: pd.DataFrame, tests, base_y: float = 1.015, step: float = 0.052) -> None:
    """One dedicated row per test ABOVE the axis for shares at which 80% power is not reached (no over-printing)."""
    import matplotlib.transforms as mtrans

    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    rows = [t for t in tests if not lines[(lines["test"] == t) & (lines["status"] == "not_reached")].empty]
    for i, t in enumerate(rows):
        label, color, _ls, mk = STYLES[t]
        nr = lines[(lines["test"] == t) & (lines["status"] == "not_reached")]
        y = base_y + i * step
        ax.plot(nr["fraction"], np.full(len(nr), y), transform=tr, color=color, linestyle="none", marker=mk, ms=4.5,
                mfc=color, mew=1.0, clip_on=False)
    if rows:
        ax.text(-0.01, base_y + (len(rows) - 1) * step / 2, "not reached\nwithin grid", transform=ax.transAxes, fontsize=5,
                color="#555555", ha="right", va="center")


def _draw_literature_v2(axes_h, axes_v, lit: pd.DataFrame, y_lim: tuple[float, float], label_ax) -> None:
    """Horizontal reference lines on the sparse panels, vertical on the dense panel; labels with leader ticks."""
    if lit.empty:
        return
    for ax in axes_h:
        for y in lit["y"]:
            ax.axhline(y, color="#bbbbbb", lw=0.6, linestyle=(0, (1, 2)), zorder=0)
    for ax in axes_v:
        for y in lit["y"]:
            ax.axvline(y, color="#bbbbbb", lw=0.6, linestyle=(0, (1, 2)), zorder=0)
    lab = lit[lit["claim_id"].isin(LIT_LABELLED)].sort_values("y") if "claim_id" in lit else lit.sort_values("y")
    pos = _label_positions(lab["y"].to_numpy(), min_ratio=1.22)
    for (y, short), yp in zip(lab[["y", "short"]].itertuples(index=False), pos, strict=True):
        if y_lim[0] <= yp <= y_lim[1]:
            label_ax.annotate(short, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(1.03, yp), textcoords=("axes fraction", "data"),
                              fontsize=5.2, color="#555555", va="center", ha="left", annotation_clip=False,
                              arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.5, "shrinkA": 0, "shrinkB": 0})


def _plan_variant_rows(sub: pd.DataFrame, t: str) -> pd.DataFrame:
    """Rows of test t on its plan outcome variant: T1 rows must be T1 on raw Y (`t1_on_raw_y`, also inside a
    winsorized unit); other tests take `V2_PLAN_VARIANT`, falling back to raw when that variant was not run."""
    if t.startswith("T1") and "t1_on_raw_y" in sub.columns:
        sub = sub[sub["t1_on_raw_y"].astype("boolean").fillna(False).astype(bool)]
    v = V2_PLAN_VARIANT.get(t, "raw")
    if (sub["variant"] == v).any():
        return sub[sub["variant"] == v]
    return sub[sub["variant"] == "raw"] if (sub["variant"] == "raw").any() else sub


def _frontiers_v2(fr: pd.DataFrame, clock: str, alternative: str, selection: str, tests, base_set: str, x_kind: str) -> pd.DataFrame:
    """Current (non-superseded) frontier rows of one panel, each test on its plan outcome variant."""
    parts = []
    for t in tests:
        sub = fr[(fr["clock"] == clock) & (fr["alternative"] == alternative) & (fr["selection"] == selection) & (fr["test"] == t)
                 & (fr["base_set"] == base_set) & (fr["x_kind"] == x_kind) & (~fr["superseded"].astype(bool))]
        parts.append(_plan_variant_rows(sub, t))
    out = pd.concat(parts, ignore_index=True) if parts else fr.iloc[0:0]
    return out.sort_values(["test", "fraction"]).reset_index(drop=True)


def _panel_v2(ax, lines: pd.DataFrame, tests, title: str, xlabel: str, y_lim, k_axis: bool, sizes: dict | None) -> None:
    for t in tests:
        g = lines[lines["test"] == t]
        if g.empty:
            continue
        label, color, ls, mk = STYLES[t]
        if sizes and t in sizes:
            label = f"{label} — null rejection {sizes[t]}"
        ok = g[g["status"].isin(["interpolated", "at_grid_min"])]
        ax.plot(ok["fraction"], ok["effect_bp_at_target"], color=color, linestyle=ls, marker=mk, ms=5.5, lw=1.4,
                mfc="white" if mk != "x" else color, mew=1.2, label=label)
    _not_reached_strip(ax, lines, tests)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*y_lim)
    xs = sorted(lines["fraction"].dropna().unique())
    ax.set_xticks(xs)
    if k_axis:
        H = float(lines["H_mean"].dropna().iloc[0]) if lines["H_mean"].notna().any() else np.nan
        ks = sorted(lines["k"].dropna().unique())
        ax.set_xticklabels([f"{int(k)}\n({100 * k / H:.1f}%)" for k in ks], fontsize=6.5)
    else:
        ax.set_xticklabels([f"{100 * x:g}%" for x in xs])
    ax.minorticks_off()
    ax.grid(True, which="major", color="#e6e6e6", lw=0.5)
    ax.set_title(title, fontsize=7, loc="left", y=1.29)
    ax.set_xlabel(xlabel, fontsize=6.5)
    ax.tick_params(length=2.5, labelsize=6.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _dense_panel_v2(ax, surf: pd.DataFrame, clock: str, tests, sizes: dict | None) -> pd.DataFrame:
    rows = []
    for t in tests:
        g = surf[(surf["clock"] == clock) & (surf["alternative"] == "dense") & (surf["test"] == t)
                 & (surf["effect_bp"] > 0) & (~surf["superseded"].astype(bool))]
        g = _plan_variant_rows(g, t).sort_values("effect_bp")
        if g["variant"].nunique() > 1:
            g = g[g["variant"] == g["variant"].min()]
        if g.empty:
            continue
        label, color, ls, mk = STYLES[t]
        ax.plot(g["effect_bp"], g["power"], color=color, linestyle=ls, marker=mk, ms=4.5, lw=1.2, mfc="white" if mk != "x" else color)
        rows.append(g)
    ax.axhline(0.8, color="#999999", lw=0.6)
    ax.text(10.5, 0.815, "80% power", fontsize=5.5, color="#777777")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xticks([10, 25, 50, 100, 200])
    ax.set_xticklabels(["10", "25", "50", "100", "200"], fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5, length=2.5)
    ax.set_xlabel("tau = cross-household SD of persistent ability (bp per 20 sessions, log scale)", fontsize=6.5)
    ax.set_ylabel("power at nominal 5%", fontsize=6.5)
    ax.set_title("(d) Dense ability: every household has theta_h ~ N(0, tau^2)\n(T2/T3 on winsorized Y; T1/T4 on raw Y)", fontsize=7, loc="left", y=1.29)
    ax.grid(True, color="#e6e6e6", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return pd.concat(rows, ignore_index=True) if rows else surf.iloc[0:0]


def make_figure_v2(
    surf: pd.DataFrame, fr: pd.DataFrame, clock: str = "T", households: str = "uniform", trades: str = "common",
    lit: pd.DataFrame | None = None, out: Path | None = None, y_lim: tuple[float, float] = (8.0, 6000.0), sizes: dict | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Figure 2 v2: (a) k informed of the R1 households, (b) k informed of the n >= 20 households, (c) sparse trades,
    (d) dense power curves; size-correct procedures only, each on its plan outcome variant; literature as reference lines."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 9.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    sel_words = {"uniform": "drawn at random", "activity": "drawn in proportion to activity"}
    trade_words = {"common": "every purchase informed with the same probability", "household": "household-specific informed probability (CV 1)"}
    la = _frontiers_v2(fr, clock, "sparse_households_k", households, V2_TESTS_R1, "R1", "k_over_H")
    lb = _frontiers_v2(fr, clock, "sparse_households_k", households, V2_TESTS_N20, "n20", "k_over_H")
    lc = _frontiers_v2(fr, clock, "sparse_trades", trades, V2_TESTS_TRADES, "all", "share_of_all")
    Ha = la["H_mean"].dropna().iloc[0] if la["H_mean"].notna().any() else np.nan
    Hb = lb["H_mean"].dropna().iloc[0] if lb["H_mean"].notna().any() else np.nan
    _panel_v2(ax_a, la, V2_TESTS_R1, f"(a) k informed households among the H = {Ha:.0f} eligible (R1) households\n(the T2 sample); informed households {sel_words[households]}",
              "k informed households (share k/H below)", y_lim, True, sizes)
    _panel_v2(ax_b, lb, V2_TESTS_N20, f"(b) k informed households among the H = {Hb:.0f} with >= 20 purchases\n(the T3 sample); informed households {sel_words[households]}",
              "k informed households (share k/H below)", y_lim, True, sizes)
    _panel_v2(ax_c, lc, V2_TESTS_TRADES, f"(c) Sparse informed purchases:\n{trade_words[trades]}",
              "share of all purchases that are informed (log scale)", y_lim, False, sizes)
    dense = _dense_panel_v2(ax_d, surf, clock, V2_TESTS_DENSE, sizes)
    for ax in (ax_a, ax_c):
        ax.set_ylabel("effect per informed unit at 80% power\n(bp per 20 sessions, log scale)", fontsize=6.5)
    lit = literature_lines(y_min=y_lim[0], y_max=y_lim[1]) if lit is None else lit
    _draw_literature_v2((ax_a, ax_b, ax_c), (ax_d,), lit, y_lim, ax_c)
    handles, labels = {}, []
    for ax in (ax_a, ax_b, ax_c):
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    order = [STYLES[t][0] for t in ("T1_blk20_2s", "T2_I1cal_ES", "T2_I2b_ES", "T3", "T4_z2", "T4_z3")]
    labels = sorted(labels, key=lambda lab: next((i for i, o in enumerate(order) if lab.startswith(o)), 99))
    fig.legend([handles[lab] for lab in labels], labels, loc="lower center", ncol=2, fontsize=6, frameon=False, handlelength=3.0,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title or f"Detection region ({clock} clock): smallest effect detected with 80% power against the block-shift placebo null;\n"
                 f"size-correct procedures only (measured null rejection in the legend)", fontsize=8, y=0.995)
    fig.text(0.01, 0.083, "Dotted gray lines: published effect sizes rescaled to bp per 20 sessions (pooled averages, not per-informed-unit effects; no paper\n"
             "reports an informed share; drawn vertically in (d)). Markers in the strip above a panel: 80% power not reached within the effect grid at that share.",
             fontsize=5.5, color="#555555", va="bottom")
    fig.tight_layout(rect=(0, 0.10, 0.86, 0.965), h_pad=3.0)
    behind = pd.concat([la.assign(panel="a_k_R1"), lb.assign(panel="b_k_n20"), lc.assign(panel="c_sparse_trades")], ignore_index=True)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=220)
        behind.to_csv(out.with_suffix(".csv"), index=False)
        dense.to_csv(out.parent / (out.stem + "_dense_panel.csv"), index=False)
        lit.assign(panel="reference_line").to_csv(out.parent / (out.stem + "_literature_lines.csv"), index=False)
    return fig, behind


def sizes_from_csv(path: Path, clock: str) -> dict:
    """Measured null rejection (size_null_v2.csv) per Figure-2 test id, formatted for legend labels."""
    if not Path(path).exists():
        return {}
    s = pd.read_csv(path)
    s = s[s["clock"] == clock]

    def pick(test, inference, sided, variant, weights="none"):
        r = s[(s["test"] == test) & (s["inference"] == inference) & (s["sidedness"] == sided) & (s["variant"] == variant) & (s["weights"] == weights)]
        return f"{100 * r['reject_rate'].iloc[0]:.1f}%" if len(r) else None

    out = {"T1_blk20_2s": pick("T1", "I1blk20", "two-sided", "raw"), "T2_I1cal_ES": pick("T2", "I1cal", "one-sided", "wins", "ES"),
           "T2_I2b_ES": pick("T2", "I2b", "one-sided", "wins", "ES"), "T3": pick("T3", "N2_critical_value", "one-sided", "wins"),
           "T4_z2": pick("T4_z2", "N2_placebo_p", "one-sided", "raw"), "T4_z3": pick("T4_z3", "N2_placebo_p", "one-sided", "raw")}
    return {k: v for k, v in out.items() if v}


def main_v2(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Figure 2 v2 (detection region, size-correct procedures)")
    ap.add_argument("--clock", default="T")
    ap.add_argument("--households", default="uniform", choices=("uniform", "activity"))
    ap.add_argument("--trades", default="common", choices=("common", "household"))
    ap.add_argument("--out", type=Path, default=CALIB_DIR / "figure2_detection_region_v2")
    args = ap.parse_args(argv)
    surf, fr = load_inputs()
    sizes = sizes_from_csv(CALIB_DIR / "size_null_v2.csv", args.clock)
    make_figure_v2(surf, fr, args.clock, args.households, args.trades, out=args.out, sizes=sizes)
    print(f"wrote {args.out}.pdf/.png/.csv")
