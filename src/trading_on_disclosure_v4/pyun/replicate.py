"""Closest feasible replication of Pyun's Table 3 on our data, and the CLI that runs the whole A7 set.

Replication = Pyun's design as printed (p. 57): one row per purchase transaction with a transaction
date and a public date inside January 2015-December 2025 and 1-365 calendar days apart; official
sources only (his source, pp. 15-16); CRSP-matched securities (PERMNO); buy-and-hold return from the
close of the transaction day and from the close of the filing day; equal-weighted; market-adjusted
against the value-weighted CRSP index (``vwretd``), raw and S&P-500-adjusted beside it; horizons
5, 10, 20, 30, 60 and 120 sessions (Pyun's 5/30/60/120; our 10/20; 300 runs past the calendar);
standard errors clustered by household x filing month (Pyun), by household x 20-session block (ours)
and by household x anchor day. The paired difference is the mean of (transaction-date minus
filing-date return) on the purchases with both windows complete. Value weights are not used: Pyun's
Table 3 is equal-weighted, and the cache carries no market capitalisation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.pyun import bridge as br
from trading_on_disclosure_v4.pyun.estimate import Inputs, State, line_frame, quantities

OUT_DIR = br.OUT_DIR
REPLICATION_PATH = OUT_DIR / "replication.csv"
REPORT_PATH = OUT_DIR / "pyun_report.json"
LOG_PATH = br.panel.REPO_ROOT / "data" / "v4" / "restricted" / "pyun" / "run.log"
HORIZONS = (5, 10, 20, 30, 60, 120)
BENCHES = ("raw", "vw", "sp")
QUANT = ("T_mean", "P_mean", "TP_diff")
#: Pyun Table 3 (p. 57), percent over h sessions -> bp: (from t + 1, from d + 1, difference, t of the difference, purchases)
PYUN_TABLE3 = {
    (5, "raw"): (30.4, 17.8, 12.6, 0.98, 37183), (5, "vw"): (3.5, 3.6, 0.0, -0.01, 37183),
    (30, "raw"): (155.1, 142.3, 12.8, 0.23, 36681), (30, "vw"): (-4.4, 4.6, -9.1, -0.60, 36681),
    (60, "raw"): (317.7, 299.1, 18.6, 0.30, 36394), (60, "vw"): (-1.0, -19.3, 18.3, 0.81, 36396),
    (120, "vw"): (-64.6, -67.6, 3.0, 0.19, 35112),
}
COLUMNS = ["record", "entry", "h", "bench", "winsor", "quantity", "estimate_bp", "estimate_bp_per20", "se_pyun", "t_pyun",
           "se_block", "t_block", "se_day", "n_events", "n_households", "n_filing_months", "n_blocks", "n_days",
           "pyun_bp", "pyun_t", "pyun_n", "pyun_page"]


def replication_rows(inp: Inputs, records=("official", "dedup"), entries=("pyun", "ours"), horizons=HORIZONS,
                     benches=BENCHES, winsor_h30_vw: bool = True, log=None) -> pd.DataFrame:
    rows = []
    for record in records:
        for entry in entries:
            for h in horizons:
                for bench in benches:
                    variants = [False] + ([True] if (winsor_h30_vw and h == 30 and bench == "vw") else [])
                    for winsor in variants:
                        st = State(record=record, entry_T=entry, entry_P=entry, h=h, bench=bench, winsor=winsor)
                        q = quantities(inp, st, line_frame(inp, st), which=QUANT)
                        for quantity in QUANT:
                            r = q[quantity]
                            est = r["estimate"]
                            py = PYUN_TABLE3.get((h, bench)) if (record == "official" and entry == "pyun" and not winsor) else None
                            k = {"T_mean": 0, "P_mean": 1, "TP_diff": 2}[quantity]
                            rows.append({"record": record, "entry": entry, "h": h, "bench": bench, "winsor": winsor, "quantity": quantity,
                                         "estimate_bp": est, "estimate_bp_per20": est * 20.0 / h,
                                         "se_pyun": r.get("se_pyun"), "t_pyun": est / r["se_pyun"] if r.get("se_pyun") else np.nan,
                                         "se_block": r.get("se_block"), "t_block": est / r["se_block"] if r.get("se_block") else np.nan,
                                         "se_day": r.get("se_day"), "n_events": r["n"], "n_households": r["n_households"],
                                         "n_filing_months": r.get("n_clusters_pyun"), "n_blocks": r.get("n_clusters_block"),
                                         "n_days": r.get("n_clusters_day"),
                                         "pyun_bp": py[k] if py else np.nan, "pyun_t": (py[3] if k == 2 else np.nan) if py else np.nan,
                                         "pyun_n": py[4] if py else np.nan, "pyun_page": 57 if py else np.nan})
                if log:
                    log(f"replication {record}/{entry}/h{h} done")
    return pd.DataFrame(rows)[COLUMNS]


def sample_report(inp: Inputs) -> dict:
    """Count-only description of the line frames (no identifiers)."""
    out = {}
    for record in ("all", "official", "dedup"):
        f = line_frame(inp, State(record=record))
        buy = f["direction"] == "purchase"
        out[record] = {"paired_lines": int(len(f)), "purchases": int(buy.sum()), "sales": int((~buy).sum()),
                       "households": int(f["household_id"].nunique()), "permnos": int(f["permno"].nunique())}
    lines = inp.lines
    both = (lines["trade_date_analysis"] >= "2015-01-01") & (lines["trade_date_analysis"] <= "2025-12-31") & \
           (lines["analysis_public_date"] >= "2015-01-01") & (lines["analysis_public_date"] <= "2025-12-31")
    out["lines"] = {"directional_with_permno": int(len(lines)), "both_dates_2015_2025": int(both.sum()),
                    "lag_1_365_within_period": int((both & lines["lag_days"].between(1, 365)).sum()),
                    "lag_below_1_within_period": int((both & (lines["lag_days"] < 1)).sum()),
                    "lag_above_365_within_period": int((both & (lines["lag_days"] > 365)).sum()),
                    "vendor_duplicate_lines": int(lines["dup_vendor"].sum()),
                    "official_lines": int((lines["record_source"] == "official").sum()),
                    "vendor_lines": int((lines["record_source"] == "capitol_trades").sum()),
                    "proxy_public_date_lines": int(lines["analysis_public_date_is_proxy"].astype(bool).sum()),
                    "transaction_date_not_a_session": int((lines["a_T_pyun"] != lines["a_T_ours"]).sum()),
                    "public_anchor_ours_one_session_after_pyun": int(((lines["a_P_ours"] - lines["a_P_pyun"]) == 1).sum()),
                    "public_anchor_ours_missing": int((lines["a_P_ours"] < 0).sum())}
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Pyun (2026) replication, bridge and macros")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with open(LOG_PATH, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    t0 = time.time()
    inp = Inputs()
    log(f"inputs loaded in {time.time() - t0:.1f}s")
    rep = replication_rows(inp, log=log)
    rep.to_csv(args.out / REPLICATION_PATH.name, index=False)
    table = br.run_bridge(inp, log=log)
    diffs = br.check_terminal(table)
    table.to_csv(args.out / br.BRIDGE_PATH.name, index=False)
    report = {"sample": sample_report(inp), "terminal_check": diffs.to_dict("records"),
              "seconds": round(time.time() - t0, 1), "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (args.out / REPORT_PATH.name).write_text(json.dumps(report, indent=2, default=float))
    from trading_on_disclosure_v4.pyun import macros

    macros.write(rep, table, report["sample"])
    log(f"done in {time.time() - t0:.1f}s")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
