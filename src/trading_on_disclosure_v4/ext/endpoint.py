"""Endpoint censoring resolved with the July 2026 CRSP slice, and the two treatments of delisting proceeds
(extension plan addendum 2026-10-19, section B). Public clock, de-duplicated universe, market-adjusted
outcome. Writes data/v4/ext/endpoint_completion.csv and missing_return_sensitivity_endpoint.csv.
"""

from __future__ import annotations

import argparse
import gzip
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.ext import common as cm
from trading_on_disclosure_v4.ext import windows as w
from trading_on_disclosure_v4.outcomes import panel as pm

H = 20
BP = 1e4
SLICE = "crsp_v2_daily_202607_202607.csv.gz"
COMMON_EXCH = ("N", "A", "Q")
T_COMPLETED = "(vi) endpoint_windows_completed_july_2026"
T_CASH = "(vii) completed_plus_delisted_proceeds_in_cash"
T_BENCH = "(viii) completed_plus_delisted_proceeds_at_benchmark"
OUT_COMPLETION = cm.OUT_DIR / "endpoint_completion.csv"
OUT_SENS = cm.OUT_DIR / "missing_return_sensitivity_endpoint.csv"
RESTRICTED = cm.RESTRICTED_EXT / "endpoint"
LOG = cm.LOG_DIR / "endpoint.log"


def july_slice(permnos: set[int]) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.Series]:
    """July 2026 sessions, the securities' daily returns (common-stock rows plus the delisting-return rows, as the panel
    builder attaches them) and the S&P 500 return by session, from the raw monthly slice."""
    path = pm.project_root() / "data" / "raw" / "crsp" / SLICE
    with gzip.open(path, "rt") as fh:
        d = pd.read_csv(fh, usecols=["PERMNO", "DlyCalDt", "PrimaryExch", "DlyRet", "sprtrn"], dtype={"PERMNO": "int64", "PrimaryExch": str})
    d["DlyCalDt"] = pd.to_datetime(d["DlyCalDt"])
    sessions = pd.DatetimeIndex(sorted(d["DlyCalDt"].unique()))
    bench = d.groupby("DlyCalDt")["sprtrn"].first().reindex(sessions)
    keep = d["PERMNO"].isin(list(permnos)) & (d["PrimaryExch"].isin(COMMON_EXCH) | (d["PrimaryExch"] == "X"))
    rows = d.loc[keep, ["PERMNO", "DlyCalDt", "PrimaryExch", "DlyRet"]].rename(columns={"PERMNO": "permno", "DlyCalDt": "date", "DlyRet": "ret"})
    rows = rows.drop_duplicates(["permno", "date"], keep="first")
    return sessions, rows, bench


def extended_grid(market: w.Market, july_sessions: pd.DatetimeIndex, july_rows: pd.DataFrame, july_bench: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray, dict[int, np.ndarray]]:
    """Calendar through July, benchmark grid, and a per-PERMNO return grid on it (panel through June, slice after)."""
    cal = market.calendar.append(july_sessions[july_sessions > market.calendar[-1]])
    bench = np.concatenate([market.bench, july_bench.reindex(cal[len(market.calendar):]).to_numpy(dtype=float)])
    grids: dict[int, np.ndarray] = {}
    by_p = {int(p): g.set_index("date")["ret"] for p, g in july_rows.groupby("permno")}
    for p in sorted(set(july_rows["permno"].astype(int))):
        base = market.grid(p)
        extra = by_p[p].reindex(cal[len(market.calendar):]).to_numpy(dtype=float)
        grids[p] = np.concatenate([base, extra])
    return cal, bench, grids


def complete_windows(u: pd.DataFrame, market: w.Market, cal: pd.DatetimeIndex, bench: np.ndarray, grids: dict[int, np.ndarray]) -> pd.DataFrame:
    """For every endpoint-censored window: the 20-session market-adjusted outcome on the extended grid, or why not."""
    rows = []
    for r in u.itertuples(index=False):
        p = int(r.permno)
        lo, hi = int(r.anchor_idx) + 1, int(r.anchor_idx) + H
        g = grids.get(p)
        rec = {"event_id": r.event_id, "signal_uid": r.signal_uid, "permno": p, "direction": r.direction, "anchor_idx": int(r.anchor_idx),
               "completed": False, "reason": "", "n_observed": 0, "y_completed_bp": np.nan}
        if g is None:
            base = market.grid(p)
            rec["reason"] = "no July rows for the security"
            rec["n_observed"] = int(np.isfinite(base[lo:min(hi, len(base) - 1) + 1]).sum())
        elif hi > len(cal) - 1:
            rec["reason"] = "window runs past 31 July 2026"
        else:
            wr, wb = g[lo:hi + 1], bench[lo:hi + 1]
            obs = np.isfinite(wr) & np.isfinite(wb)
            rec["n_observed"] = int(obs.sum())
            if obs.all():
                rec["completed"] = True
                rec["y_completed_bp"] = float((np.prod(1.0 + wr) - 1.0) - (np.prod(1.0 + wb) - 1.0)) * BP
            else:
                rec["reason"] = "a return is missing in July (delisted or not traded)"
        rows.append(rec)
    return pd.DataFrame(rows)


def delisting_treatments(u: pd.DataFrame, market: w.Market) -> pd.DataFrame:
    """For windows with a delisting-return row inside: proceeds reinvested at the benchmark (partial against the
    benchmark over the same sessions) and proceeds held in cash (partial against the benchmark over all 20 sessions)."""
    rows = []
    for r in u.itertuples(index=False):
        g = market.grid(int(r.permno))
        lo, hi = w.window_bounds(int(r.anchor_idx), market.n)
        pw = w.partial_window(g, market.bench, lo, hi)
        wb = market.bench[lo:hi + 1]
        bh_full = float(np.prod(1.0 + wb[np.isfinite(wb)]) - 1.0) if np.isfinite(wb).any() else np.nan
        rows.append({"event_id": r.event_id, "signal_uid": r.signal_uid, "permno": int(r.permno), "direction": r.direction,
                     "n_observed": pw["n_observed"], "bh_security_to_delisting": pw["bh_security"],
                     "y_benchmark_reinvested_bp": pw["partial_outcome"] * BP if np.isfinite(pw["partial_outcome"]) else np.nan,
                     "y_cash_bp": (pw["bh_security"] - bh_full) * BP if np.isfinite(pw["bh_security"]) and np.isfinite(bh_full) else np.nan})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="endpoint censoring completed with the July 2026 slice; delisting proceeds")
    ap.parse_args(argv)
    log = lambda m: cm.log_line(LOG, m)  # noqa: E731
    t0 = time.time()
    calendar = pm.v2_trading_calendar()
    market = w.Market(calendar, pm.load_benchmark(), pm.load_factors(), pm.daily_parquet_path())
    u, report = w.public_universe(calendar)
    u["blk"] = u["anchor_idx"] // 20
    y = u["y_market_adjusted"].to_numpy(dtype=float)
    status = u["status_market_adjusted"].to_numpy()
    trunc = u.loc[status == "end_of_sample_truncation"]
    market.load(u["permno"].unique())
    delist = market.delist
    dl_idx = delist.reindex(u["permno"].to_numpy()).to_numpy(dtype=float)
    has_dl = w.delisting_in_window(u["anchor_idx"].to_numpy(), dl_idx)
    delisted_missing = u.loc[has_dl & ~np.isfinite(y)]
    log(f"universe {len(u)}; endpoint-censored {len(trunc)}; delisted windows without an outcome {len(delisted_missing)}")
    sessions, rows, bench_j = july_slice(set(trunc["permno"].astype(int)))
    cal, bench, grids = extended_grid(market, sessions, rows, bench_j)
    comp = complete_windows(trunc, market, cal, bench, grids)
    dlt = delisting_treatments(delisted_missing, market)
    RESTRICTED.mkdir(parents=True, exist_ok=True)
    comp.to_parquet(RESTRICTED / "completed_windows.parquet", index=False)
    dlt.to_parquet(RESTRICTED / "delisting_treatments.parquet", index=False)
    # treatments
    sens = []
    base = w.spread_estimate(u, y)
    sens.append(w.sens_row("public_spread_dedup", "P", w.T_OBSERVED, base["estimate"], "bp", base["se"], base["n"], 0, np.nan, np.nan,
                           "the record (treatment (i) of the window audit)"))
    y6 = y.copy()
    done = comp.loc[comp["completed"]].set_index("event_id")["y_completed_bp"]
    fill = u["event_id"].map(done)
    m6 = ~np.isfinite(y6) & fill.notna().to_numpy()
    y6[m6] = fill.to_numpy(dtype=float)[m6]
    r6 = w.spread_estimate(u, y6)
    sens.append(w.sens_row("public_spread_dedup", "P", T_COMPLETED, r6["estimate"], "bp", r6["se"], r6["n"], int(m6.sum()), np.nan, np.nan,
                           f"the {int(m6.sum())} endpoint-censored windows completed on the July 2026 slice; other incomplete windows still missing"))
    for label, col in ((T_CASH, "y_cash_bp"), (T_BENCH, "y_benchmark_reinvested_bp")):
        yv = y6.copy()
        d = dlt.set_index("event_id")[col]
        f2 = u["event_id"].map(d)
        m = ~np.isfinite(yv) & f2.notna().to_numpy()
        yv[m] = f2.to_numpy(dtype=float)[m]
        r = w.spread_estimate(u, yv)
        sens.append(w.sens_row("public_spread_dedup", "P", label, r["estimate"], "bp", r["se"], r["n"], int(m6.sum() + m.sum()), np.nan, np.nan,
                               f"as (vi) plus the {int(m.sum())} delisted windows with proceeds {'held in cash (partial security return against the benchmark over all 20 sessions)' if col == 'y_cash_bp' else 'reinvested at the benchmark (partial window against the benchmark over the same sessions, as treatment (ii))'}"))
    pd.DataFrame(sens)[w.SENS_COLUMNS].to_csv(OUT_SENS, index=False, lineterminator="\n")
    # completion report
    rep = []
    for dirn in ("purchase", "sale", "all"):
        c = comp if dirn == "all" else comp[comp["direction"] == dirn]
        d = dlt if dirn == "all" else dlt[dlt["direction"] == dirn]
        rep.append({"direction": dirn, "endpoint_censored": int(len(c)), "completed": int(c["completed"].sum()),
                    "not_completed_no_july_rows": int((c["reason"] == "no July rows for the security").sum()),
                    "not_completed_missing_july_return": int((c["reason"] == "a return is missing in July (delisted or not traded)").sum()),
                    "completed_mean_bp": float(c.loc[c["completed"], "y_completed_bp"].mean()) if c["completed"].any() else np.nan,
                    "completed_median_bp": float(c.loc[c["completed"], "y_completed_bp"].median()) if c["completed"].any() else np.nan,
                    "delisted_windows": int(len(d)), "delisted_with_partial": int(d["y_cash_bp"].notna().sum()),
                    "delisted_mean_bp_benchmark_reinvested": float(d["y_benchmark_reinvested_bp"].mean()) if len(d) else np.nan,
                    "delisted_mean_bp_cash": float(d["y_cash_bp"].mean()) if len(d) else np.nan,
                    "delisted_median_bp_cash": float(d["y_cash_bp"].median()) if len(d) else np.nan,
                    "july_sessions": int(len(sessions))})
    pd.DataFrame(rep).to_csv(OUT_COMPLETION, index=False, lineterminator="\n")
    cm.write_manifest("endpoint", [cm.file_entry(pm.project_root() / "data" / "raw" / "crsp" / SLICE), cm.file_entry(pm.daily_parquet_path()),
                                   cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "signals.csv")], {"seconds": round(time.time() - t0, 1), **{k: int(v) for k, v in report.items()}})
    log(f"completed {int(comp['completed'].sum())} of {len(comp)}; delisted treated {len(dlt)}; written in {time.time() - t0:.0f} s")
    print(pd.DataFrame(rep).to_string(index=False))
    print(pd.DataFrame(sens)[["treatment", "estimate", "se_block", "n_used", "n_imputed"]].to_string(index=False))


if __name__ == "__main__":
    main()
