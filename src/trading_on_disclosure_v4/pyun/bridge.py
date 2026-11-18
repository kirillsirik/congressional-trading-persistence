"""The ordered bridge from Pyun's Table 3 specification to our registered specifications.

Row 0 is Pyun's print (Table 3, p. 57). Every later row is computed on our data and changes ONE
element relative to the row above; the two terminal rows
must reproduce the registered numbers (``data/v4/existing/rerun_tables.csv`` de-duplicated block-
clustered rows; ``data/v4/results.csv`` purchase means), which ``check_terminal`` asserts. Elements
that cannot be implemented are listed with the reason and no number (``NOT_IMPLEMENTABLE``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.outcomes import panel
from trading_on_disclosure_v4.pyun.estimate import QUANTITIES, Inputs, State, state_quantities

OUT_DIR = panel.REPO_ROOT / "data" / "v4" / "pyun"
BRIDGE_PATH = OUT_DIR / "bridge_table.csv"
RERUN_TABLES = panel.REPO_ROOT / "data" / "v4" / "existing" / "rerun_tables.csv"
RESULTS = panel.REPO_ROOT / "data" / "v4" / "results.csv"
PYUN_ROW0 = {  # Table 3, market-adjusted, h = 30, p. 57; percent -> bp over 30 sessions
    "T_mean": (-4.4, "from t + 1: -0.044%"), "P_mean": (4.6, "from d + 1: 0.046%"),
    "TP_diff": (-9.1, "difference -0.091% (t -0.60)")}
PYUN_ROW0_N = 36681
QUANTITY_LABELS = {
    "T_mean": "purchase mean from the transaction date", "P_mean": "purchase mean from the public date",
    "TP_diff": "paired two-date difference, purchases", "TP_diff_all": "paired two-date difference, purchases and sales pooled",
    "P_spread": "purchase-minus-sale spread, public clock", "T_spread": "purchase-minus-sale spread, transaction clock"}
COLUMNS = ["step", "chain", "element", "from_to", "quantity", "estimate_bp", "estimate_bp_per20", "se_headline",
           "clusters", "se_pyun", "se_block", "se_day", "n_events", "n_households", "n_clusters_block", "h",
           "record", "unit", "pairing", "entry_T", "entry_P", "bench", "period", "note"]


@dataclass(frozen=True)
class Step:
    step: str
    element: str
    from_to: str
    state: State
    clusters: str  # which SE is the row's headline: pyun | block
    note: str = ""
    chain: str = "both"  # both | spread_paired | purchase_mean


BASE = State(record="all", unit="line", pairing=True, entry_T="pyun", entry_P="pyun", h=30, bench="vw", period="2015-2025")
S = [BASE]
S.append(S[-1].with_(record="official"))
S.append(S[-1].with_(record="dedup"))
S_ENTRY_T = S[-1].with_(entry_T="ours")
S_ENTRY_P = S_ENTRY_T.with_(entry_P="ours")
S_UNIT = S_ENTRY_P.with_(unit="event")
S_UNPAIRED = S_UNIT.with_(pairing=False)
S_H20 = S_UNPAIRED.with_(h=20)
S_SP = S_H20.with_(bench="sp")
S_FF = S_SP.with_(bench="ff5mom")
S_TERM_A = S_SP.with_(period="all")
S_TERM_B = S_FF.with_(period="plan_AB")

STEPS: tuple[Step, ...] = (
    Step("1", "period", "filings January 2015-December 2025 -> transaction and public dates inside 2015-2025 (our whole record: official lines and vendor lines as they stand, Pyun's Table 3 design)",
         S[0], "pyun", "the move from row 0 is the record itself (his population, his parse -> our lines); the period window is the same"),
    Step("2", "record", "population (electronic + transcribed paper filings) -> official text-parsed lines only (our analogue of Pyun's electronic segment)",
         S[1], "pyun", "the paper (scanned) segment does not exist on our side; this row is the replication row of replication.csv"),
    Step("3", "record", "official only -> official + vendor lines that do not reproduce an official transaction (plan §2 duplicate rule at the line level)",
         S[2], "pyun", "vendor lines carry proxy public dates (Capitol Trades published date minus one calendar day)"),
    Step("4", "asset filter", "CRSP common stock by name match -> CRSP EQTY/NS by PERMNO (no-op)", S[2], "pyun",
         "no-op: every line already carries a PERMNO; Pyun's name-match loss cannot be reproduced"),
    Step("5", "direction", "purchases -> purchases (no-op; the spread and pooled paired rows are shown from here)", S[2], "pyun",
         "no-op; sales enter only the spread and the pooled paired difference"),
    Step("6", "entry, transaction clock", "close of t (returns from t + 1) -> first session on or after t (returns from the next session)",
         S_ENTRY_T, "pyun", "differs only when the transaction date is not a session"),
    Step("7", "entry, public clock", "close of the filing day d (returns from d + 1) -> close of the first session strictly after d (one session later)",
         S_ENTRY_P, "pyun", "the public-clock return starts one session later than Pyun's"),
    Step("8a", "unit", "one row per transaction line -> household x PERMNO x transaction date x direction episode (transaction clock) / non-mixed signal (public clock and paired rows); events containing at least one paired line",
         S_UNIT, "pyun", "the paired difference becomes the signal's own transaction anchor minus its public anchor (the released form)"),
    Step("8b", "pairing rule", "both dates inside the period and 1-365 days apart -> every event whose anchor lies in 2015-2025 (no pairing requirement)",
         S_UNPAIRED, "pyun", "vendor rows without an observed lag and lags outside 1-365 days enter; the two clocks no longer see the same trades except in the paired rows"),
    Step("9", "horizon", "30 -> 20 sessions", S_H20, "pyun", ""),
    Step("10", "weighting", "equal per purchase -> equal per event (no-op)", S_H20, "pyun",
         "no-op; value weights would need market caps, which the cache does not carry"),
    Step("11a", "benchmark", "value-weighted CRSP market -> S&P 500 total return (the existing paper)", S_SP, "pyun", ""),
    Step("12a", "inference", "household x filing month -> household x 20-session block (SE column only)", S_SP, "block", "",
         "spread_paired"),
    Step("13a", "period", "anchors 2015-2025 -> the existing paper's window (every anchor)", S_TERM_A, "block",
         "= registered TW_P1_E20 / TW_C1_T20 / TW_C1_PAIRED, de-duplicated, block clusters", "spread_paired"),
    Step("11b", "benchmark", "S&P 500 -> Fama-French five factors plus momentum (the plan's outcome; purchase means)", S_FF, "pyun",
         "FF5+MOM outcomes from the forward cache at 20 sessions", "purchase_mean"),
    Step("12b", "inference", "household x filing month -> household x 20-session block (SE column only)", S_FF, "block", "",
         "purchase_mean"),
    Step("13b", "period", "anchors 2015-2025 -> the plan's two periods (2015-2020 windows ending by 2020-12-31; 2021-2025)", S_TERM_B, "block",
         "= registered purchase means of results.csv (Section 4)", "purchase_mean"),
)

NOT_IMPLEMENTABLE: tuple[tuple[str, str, str], ...] = (
    ("X1", "record: paper (scanned) filings", "Pyun's 2,300 transcribed PTRs supply 55,350 of 85,461 two-date transactions and 28,696 of 43,115 purchases (p. 16); no scanned filing is parsed on our side, so his population, his electronic-vs-paper split (Table 5) and the bulk-filer exclusion (Table IA2) have no row"),
    ("X2", "benchmark: DGTW characteristic portfolios", "125 size x book-to-market x momentum portfolios (p. 19) need Compustat book equity, which is not on disk; Table 3's characteristic-adjusted columns cannot be run"),
    ("X3", "benchmark: value-weighted 2-digit-SIC industry", "buildable from the raw CRSP file (SICCD, DlyCap, DlyRet) but not a stored series; not run"),
    ("X4", "regression design (Table 2): seven controls, date x industry fixed effects", "log book-to-market, log Amihud illiquidity and log turnover need Compustat or share volume; not run"),
    ("X5", "volume response (Table 4, 6)", "no share volume in any market file on disk"),
    ("X6", "weighting: value-weighted by lagged market capitalisation", "Pyun's Table 3 is equal-weighted per purchase, so no step is needed; the cache carries no market caps"),
    ("X7", "unit: member", "Pyun attributes spouse and dependent-child trades to the member; our unit is the harmonised household id, and the crosswalk is pending"),
)


def _row(step: Step, quantity: str, q: dict, chain: str | None = None) -> dict:
    st = step.state
    est = q.get("estimate", np.nan)
    return {"step": step.step, "chain": chain or step.chain, "element": step.element, "from_to": step.from_to,
            "quantity": quantity, "estimate_bp": est, "estimate_bp_per20": est * 20.0 / st.h if np.isfinite(est) else np.nan,
            "se_headline": q.get(f"se_{step.clusters}", np.nan), "clusters": step.clusters,
            "se_pyun": q.get("se_pyun", np.nan), "se_block": q.get("se_block", np.nan), "se_day": q.get("se_day", np.nan),
            "n_events": q.get("n", 0), "n_households": q.get("n_households", 0),
            "n_clusters_block": q.get("n_clusters_block", 0), "h": st.h, "record": st.record, "unit": st.unit,
            "pairing": st.pairing, "entry_T": st.entry_T, "entry_P": st.entry_P, "bench": st.bench, "period": st.period,
            "note": step.note}


def row0() -> list[dict]:
    rows = []
    for quantity, (bp, text) in PYUN_ROW0.items():
        rows.append({"step": "0", "chain": "both", "element": "Pyun as printed",
                     "from_to": "Table 3, market-adjusted (VW CRSP), h = 30, per purchase, population, from t + 1 and d + 1, clustered member x filing month (p. 57)",
                     "quantity": quantity, "estimate_bp": bp, "estimate_bp_per20": bp * 20.0 / 30.0, "se_headline": np.nan,
                     "clusters": "pyun", "se_pyun": np.nan, "se_block": np.nan, "se_day": np.nan, "n_events": PYUN_ROW0_N,
                     "n_households": np.nan, "n_clusters_block": np.nan, "h": 30, "record": "population", "unit": "line",
                     "pairing": True, "entry_T": "pyun", "entry_P": "pyun", "bench": "vw", "period": "2015-2025",
                     "note": f"{text}; N {PYUN_ROW0_N:,} purchases (p. 57); no SE printed"})
    return rows


def run_bridge(inp: Inputs, steps: tuple[Step, ...] = STEPS, log=None) -> pd.DataFrame:
    rows = row0()
    cache: dict[State, dict] = {}
    for step in steps:
        if step.state not in cache:
            cache[step.state] = state_quantities(inp, step.state)
            if log:
                log(f"step {step.step} computed ({step.element})")
        qs = cache[step.state]
        wanted = ("T_mean", "P_mean") if step.state.bench == "ff5mom" else QUANTITIES
        for quantity in wanted:
            rows.append(_row(step, quantity, qs[quantity]))
    for step_id, element, reason in NOT_IMPLEMENTABLE:
        rows.append({"step": step_id, "chain": "none", "element": element, "from_to": "cannot be implemented", "quantity": "",
                     "note": reason})
    return pd.DataFrame(rows)[COLUMNS]


def check_terminal(table: pd.DataFrame, rerun_path=RERUN_TABLES, results_path=RESULTS, tol: float = 5e-3) -> pd.DataFrame:
    """The terminal rows against the registered numbers (bp), to well inside the printed precision."""
    rt = pd.read_csv(rerun_path)
    rt = rt[(rt["sample"] == "dedup") & (rt["clusters"] == "block20")].set_index("test_id")
    res = pd.read_csv(results_path)
    pm = res[(res["analysis"] == "purchase_mean") & (res["variant"] == "vs_zero") & (res["statistic"] == "mean")].set_index("clock")
    targets = {("13a", "P_spread"): (rt.loc["TW_P1_E20", "estimate"], rt.loc["TW_P1_E20", "se"], rt.loc["TW_P1_E20", "n"]),
               ("13a", "T_spread"): (rt.loc["TW_C1_T20", "estimate"], rt.loc["TW_C1_T20", "se"], rt.loc["TW_C1_T20", "n"]),
               ("13a", "TP_diff_all"): (rt.loc["TW_C1_PAIRED", "estimate"], rt.loc["TW_C1_PAIRED", "se"], rt.loc["TW_C1_PAIRED", "n"]),
               ("13b", "T_mean"): (pm.loc["T", "value"], pm.loc["T", "se_block"], pm.loc["T", "n_events"]),
               ("13b", "P_mean"): (pm.loc["P", "value"], pm.loc["P", "se_block"], pm.loc["P", "n_events"])}
    out = []
    for (step, quantity), (est, se, n) in targets.items():
        r = table[(table["step"] == step) & (table["quantity"] == quantity)].iloc[0]
        out.append({"step": step, "quantity": quantity, "ours": r["estimate_bp"], "registered": est, "d_estimate": r["estimate_bp"] - est,
                    "d_se": r["se_block"] - se, "d_n": int(r["n_events"] - n)})
    diffs = pd.DataFrame(out)
    bad = diffs[(diffs[["d_estimate", "d_se"]].abs() > tol).any(axis=1) | (diffs["d_n"] != 0)]
    if not bad.empty:
        raise AssertionError(f"terminal rows do not reproduce the registered numbers: {bad.to_dict('records')}")
    return diffs
