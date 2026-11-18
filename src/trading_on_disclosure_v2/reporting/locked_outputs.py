"""Canonical output surface for the locked design (final directive §20-§22).

Everything paper-facing is generated from the machine-readable numerical
ledger — table writers resolve ledger keys, never literals — and all captions
and claims pass the language guard before touching disk.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.dates import (
    ANALYSIS_PUBLIC_PROXY_COL,
    ANALYSIS_PUBLIC_SOURCE_COL,
    SOURCE_OFFICIAL_OBSERVED,
)
from trading_on_disclosure_v2.reporting.language_guard import assert_language_allowed

HORIZON_TESTS = (
    ("TW_P2_E5", 5),
    ("TW_P2_E10", 10),
    ("TW_P1_E20", 20),
    ("TW_P2_E40", 40),
    ("TW_P2_E60", 60),
    ("TW_P2_E127", 127),
)

TABLE4_TESTS = (
    ("TW_P1_E20", "Canonical hierarchy (official, else CT minus one calendar day proxy)"),
    ("TW_P3_RAW_CT", "Raw Capitol Trades publication date (vendor-only rows)"),
    ("TW_P3_OFFICIAL_ONLY", "Observed-official-date sample"),
    ("TW_P3_OFFICIAL_SEGMENT", "Official-source record segment"),
    ("TW_P3_VENDOR_SEGMENT", "Vendor-proxy record segment"),
    ("TW_C1_TRAW20", "Raw vendor transaction date (transaction-clock benchmark)"),
)


class LedgerError(RuntimeError):
    pass


def _row_lookup(results: pd.DataFrame) -> dict[str, dict]:
    return {r["test_id"]: r for r in results.to_dict("records")}


def annualized_total_return(
    final_equity: float, initial_equity: float, n_intervals: int
) -> float:
    """Geometric annualization over every realized daily interval, with the
    true initial account value as the base."""
    if n_intervals < 1:
        raise ValueError("annualization requires at least one return interval")
    return float((final_equity / initial_equity) ** (252.0 / n_intervals) - 1.0)


def build_numerical_ledger(ctx) -> dict:
    """Flat key -> value ledger; the ONLY source of paper-facing numbers."""
    results = ctx.frames.get("adjusted_results")
    rows = _row_lookup(results) if results is not None else {}
    signals = ctx.frames["signals"]
    from trading_on_disclosure_v2.pipeline.locked_estimands import primary_sample

    prim = primary_sample(signals)
    ledger: dict[str, float | int | str] = {}

    def put(key: str, value) -> None:
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        ledger[key] = value

    # ---- sample construction (Table 1)
    txns0 = ctx.frames.get("dated_transactions")
    if txns0 is not None:
        src = txns0.get("record_source", pd.Series("official", index=txns0.index))
        put("t1.official_records", int((src == "official").sum()))
        put("t1.vendor_records", int((src != "official").sum()))
        put("t1.parsed_records_total", int(len(txns0)))
        observed = txns0[ANALYSIS_PUBLIC_SOURCE_COL] == SOURCE_OFFICIAL_OBSERVED
        put("t1.observed_official_dates", int(observed.sum()))
        put(
            "t1.vendor_one_day_proxies",
            int(txns0[ANALYSIS_PUBLIC_PROXY_COL].fillna(False).sum()),
        )
    put("t1.matched_transactions", int(len(ctx.frames.get("matched_transactions", ()))))
    put("t1.unique_signals", int(len(signals)))
    put("t1.mixed_direction_excluded", int(ctx.artifacts.get("mixed_direction_signals", 0)))
    put("t1.complete_h20_outcomes", int(prim["ar_E_h20"].notna().sum()))
    put("t1.households", int(prim["filer_uid"].nunique()))
    put("t1.securities", int(prim["permno"].nunique()))
    conc = ctx.artifacts.get("concentration", {})
    for k, v in conc.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            put(f"t1.concentration.{k}", v)
    att = ctx.artifacts.get("attestation", {})
    put("t1.parse_review_status", str(att.get("status", "PENDING")))

    # ---- estimand rows (Tables 2-5)
    stat_keys = (
        "estimate",
        "se_multiway",
        "t_stat",
        "raw_p",
        "ci_low",
        "ci_high",
        "block_ci_low",
        "block_ci_high",
        "n",
        "n_households",
        "n_entry_dates",
        "purchase_mean",
        "sale_mean",
        "p_holm",
    )
    for test_id, row in rows.items():
        for k in stat_keys:
            if k in row and row[k] == row[k]:  # skip NaN
                put(f"res.{test_id}.{k}", row[k])
        put(f"res.{test_id}.status", str(row.get("status", "")))

    # ---- date diagnostics
    diag = ctx.artifacts.get("date_proxy_diagnostics", {})
    for k in (
        "n_matched",
        "mean_diff_days",
        "median_diff_days",
        "share_ct_exactly_one_day_later",
        "share_ct_same_day",
        "share_ct_more_than_one_day_later",
        "share_ct_earlier_than_official",
    ):
        if k in diag and diag[k] == diag[k]:
            put(f"dates.{k}", diag[k])

    # ---- portfolio (Table 6)
    daily = ctx.frames.get("portfolio_daily")
    if daily is not None and len(daily) > 1:
        # every daily row is one realized return interval (the first row is
        # initial account value -> first close), so intervals = len(daily) and
        # the geometric base is the true initial equity, not the first close
        n_intervals = len(daily)
        eq = daily["equity"]
        be = daily["benchmark_equity"]
        params = ctx.artifacts.get("portfolio_params") or {}
        initial = daily.attrs.get("initial_equity", params.get("initial_equity"))
        if initial is None:
            raise KeyError(
                "initial account equity unavailable: portfolio_daily has no "
                "attrs['initial_equity'] and ctx.artifacts['portfolio_params'] "
                "has no 'initial_equity'"
            )
        initial = float(initial)
        years = n_intervals / 252
        put("t6.annualized_return", annualized_total_return(eq.iloc[-1], initial, n_intervals))
        put(
            "t6.benchmark_annualized_return",
            annualized_total_return(be.iloc[-1], initial, n_intervals),
        )
        put("t6.avg_invested_fraction", float(daily["risky_fraction"].mean()))
        put("t6.max_security_exposure", float(daily.get("max_security_exposure_frac", pd.Series(dtype=float)).max()))
        lots = ctx.artifacts.get("portfolio_lots")
        routing = ctx.artifacts.get("portfolio_routing")
        if lots is not None and len(lots):
            traded = float(lots["entry_notional"].sum() + lots["final_value"].sum())
            put("t6.annual_turnover", float(traded / 2 / eq.mean() / years))
            put("t6.n_lots", int(len(lots)))
        if routing is not None and len(routing):
            counts = routing["state"].value_counts()
            put("t6.signals_accepted", int(counts.get("accepted", 0)))
            put("t6.signals_rejected", int(len(routing) - counts.get("accepted", 0)))
        rf = ctx.frames.get("risk_free")
        strat = daily["strategy_return"].dropna()
        if rf is not None:
            excess = (strat - rf.reindex(strat.index)).dropna()
            sd = float(excess.std(ddof=1))
            put("t6.sharpe", float(excess.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan)
    return ledger


def _macro_name(key: str) -> str:
    parts = re.split(r"[^0-9a-zA-Z]+", key)
    words = []
    for p in parts:
        if not p:
            continue
        for token in re.split(r"(\d+)", p):
            if not token:
                continue
            words.append(
                _NUM_WORDS(token) if token.isdigit() else token.capitalize()
            )
    return "tw" + "".join(words)


_DIGITS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
}


def _NUM_WORDS(token: str) -> str:
    return "".join(_DIGITS[c] for c in token)


def _fmt(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    v = float(value)
    if v != v:
        return "--"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.4f}"


def build_paper_macros(ledger: dict) -> str:
    lines = [
        "% AUTO-GENERATED from numerical_ledger.json — do not edit by hand.",
        "% Every empirical number in the manuscript resolves through these macros.",
    ]
    for key in sorted(ledger):
        text = _fmt(ledger[key])
        assert_language_allowed(text, f"macro {key}")
        lines.append(f"\\newcommand{{\\{_macro_name(key)}}}{{{text}}}")
    return "\n".join(lines) + "\n"


def _table_frame(rows: list[dict], caption: str) -> tuple[pd.DataFrame, str]:
    assert_language_allowed(caption, "table caption")
    for row in rows:
        for v in row.values():
            if isinstance(v, str):
                assert_language_allowed(v, "table cell")
    return pd.DataFrame(rows), caption


def build_tables(ctx, ledger: dict) -> dict[str, tuple[pd.DataFrame, str]]:
    def L(key: str):
        if key not in ledger:
            return np.nan
        return ledger[key]

    tables: dict[str, tuple[pd.DataFrame, str]] = {}

    t1_rows = [
        {"quantity": "Official-source parsed records", "ledger_key": "t1.official_records", "value": L("t1.official_records")},
        {"quantity": "Vendor (Capitol Trades) records", "ledger_key": "t1.vendor_records", "value": L("t1.vendor_records")},
        {"quantity": "Parser outputs entering hierarchy (author-reviewed)", "ledger_key": "t1.parsed_records_total", "value": L("t1.parsed_records_total")},
        {"quantity": "Observed official public dates", "ledger_key": "t1.observed_official_dates", "value": L("t1.observed_official_dates")},
        {"quantity": "Vendor one-calendar-day proxies", "ledger_key": "t1.vendor_one_day_proxies", "value": L("t1.vendor_one_day_proxies")},
        {"quantity": "CRSP-matched common-stock transactions", "ledger_key": "t1.matched_transactions", "value": L("t1.matched_transactions")},
        {"quantity": "Unique public directional signals", "ledger_key": "t1.unique_signals", "value": L("t1.unique_signals")},
        {"quantity": "Mixed-direction signals (excluded from primary)", "ledger_key": "t1.mixed_direction_excluded", "value": L("t1.mixed_direction_excluded")},
        {"quantity": "Signals with complete 20-day outcomes", "ledger_key": "t1.complete_h20_outcomes", "value": L("t1.complete_h20_outcomes")},
        {"quantity": "Households", "ledger_key": "t1.households", "value": L("t1.households")},
        {"quantity": "Securities", "ledger_key": "t1.securities", "value": L("t1.securities")},
    ]
    tables["table1_sample_construction"] = _table_frame(
        t1_rows,
        "Data construction and analytical sample. Parser outputs are "
        "author-reviewed (attestation status: "
        f"{L('t1.parse_review_status')}); vendor-only public dates use the "
        "Capitol Trades minus-one-calendar-day proxy and are never labeled "
        "as observed official dates.",
    )

    def res_row(test_id: str, label: str) -> dict:
        return {
            "row": label,
            "estimate": L(f"res.{test_id}.estimate"),
            "se_two_way_cgm": L(f"res.{test_id}.se_multiway"),
            "ci_low": L(f"res.{test_id}.ci_low"),
            "ci_high": L(f"res.{test_id}.ci_high"),
            "p_value": L(f"res.{test_id}.raw_p"),
            "n_signals": L(f"res.{test_id}.n"),
            "household_clusters": L(f"res.{test_id}.n_households"),
            "entry_date_clusters": L(f"res.{test_id}.n_entry_dates"),
            "ledger_key": f"res.{test_id}.estimate",
        }

    t2 = [res_row("TW_P1_E20", "Purchase-minus-sale spread, 20 trading days")]
    t2[0]["purchase_mean"] = L("res.TW_P1_E20.purchase_mean")
    t2[0]["sale_mean"] = L("res.TW_P1_E20.sale_mean")
    t2.append(res_row("TW_D_SAR20", "Signed mimicking payoff (descriptive)"))
    tables["table2_primary_result"] = _table_frame(
        t2,
        "Primary public-action result: 20-trading-day purchase-minus-sale "
        "abnormal-return spread after public entry, unique-signal unit, "
        "two-way CGM clustering by household and exact public-entry trading "
        "date. The signed payoff is a descriptive summary; shorting after a "
        "disclosed sale is not the same as selling an existing position.",
    )

    t3 = []
    for test_id, h in HORIZON_TESTS:
        row = res_row(test_id, f"{h} trading days")
        row["p_holm_horizon_family"] = (
            L(f"res.{test_id}.p_holm") if test_id != "TW_P1_E20" else "primary (single-test family)"
        )
        t3.append(row)
    tables["table3_horizons"] = _table_frame(
        t3,
        "Horizon robustness. The 20-day horizon is the primary test; the "
        "5/10/40/60/127-day horizons form the P2 family with Holm-adjusted "
        "p-values.",
    )

    t4 = [res_row(tid, label) for tid, label in TABLE4_TESTS]
    tables["table4_source_date_robustness"] = _table_frame(
        t4,
        "Public-date and source robustness, reported together without "
        "selection. For vendor-only observations the canonical hierarchy "
        "proxies official public availability as one calendar day before "
        "the Capitol Trades publication date.",
    )

    t5 = [
        res_row("TW_C1_T20", "Transaction clock, 20 days from reported transaction date"),
        res_row("TW_P1_E20", "Public-action clock, 20 days after public entry"),
        res_row("TW_C1_PAIRED", "Paired transaction-minus-public difference"),
    ]
    tables["table5_two_clocks"] = _table_frame(
        t5,
        "Transaction clock versus public-action clock (descriptive, "
        "secondary, noncausal). The transaction date was not known to any "
        "outside investor at execution, so the transaction-clock row is not "
        "publicly implementable.",
    )

    t6 = [
        {"quantity": "Annualized total-account return", "ledger_key": "t6.annualized_return", "value": L("t6.annualized_return")},
        {"quantity": "Exposure-matched benchmark return (annualized)", "ledger_key": "t6.benchmark_annualized_return", "value": L("t6.benchmark_annualized_return")},
        {"quantity": "Average invested fraction", "ledger_key": "t6.avg_invested_fraction", "value": L("t6.avg_invested_fraction")},
        {"quantity": "Annual turnover", "ledger_key": "t6.annual_turnover", "value": L("t6.annual_turnover")},
        {"quantity": "Signals accepted", "ledger_key": "t6.signals_accepted", "value": L("t6.signals_accepted")},
        {"quantity": "Signals rejected (caps/cash/lots)", "ledger_key": "t6.signals_rejected", "value": L("t6.signals_rejected")},
        {"quantity": "Maximum single-security exposure", "ledger_key": "t6.max_security_exposure", "value": L("t6.max_security_exposure")},
        {"quantity": "Sharpe ratio", "ledger_key": "t6.sharpe", "value": L("t6.sharpe")},
        {"quantity": "Mean daily excess return (annualized, Newey-West)", "ledger_key": "res.TW_IMPL_PORT_NW.estimate", "value": L("res.TW_IMPL_PORT_NW.estimate")},
        {"quantity": "CAPM alpha (annualized)", "ledger_key": "res.TW_IMPL_PORT_CAPM.estimate", "value": L("res.TW_IMPL_PORT_CAPM.estimate")},
        {"quantity": "FF5+momentum alpha (annualized)", "ledger_key": "res.TW_IMPL_PORT_FF5MOM.estimate", "value": L("res.TW_IMPL_PORT_FF5MOM.estimate")},
    ]
    tables["table6_portfolio"] = _table_frame(
        t6,
        "Long-after-purchase implementation portfolio: long every eligible "
        "disclosed purchase at the public-entry close, 20-trading-day holds, "
        "5% NAV per-security cap, 50-lot cap, no leverage or shorting, "
        "10 bp round-trip costs split at entry and exit, idle cash at the "
        "risk-free rate, terminal liquidation at the sample endpoint. An "
        "implementation complement, not a second discovery exercise.",
    )
    return tables


def build_claim_ledger(ctx, ledger: dict) -> pd.DataFrame:
    res = _row_lookup(ctx.frames["adjusted_results"])

    def claim(cid, text, tests, keys):
        assert_language_allowed(text, f"claim {cid}")
        return {
            "claim_id": cid,
            "claim_text": text,
            "supporting_tests": ";".join(tests),
            "ledger_keys": ";".join(keys),
            "evidence_status": "historical",
        }

    rows = [
        claim(
            "CL-01",
            "The 20-day public-action purchase-minus-sale spread is the "
            "primary estimand; its point estimate, two-way-clustered "
            "confidence interval, and p-value are reported in Table 2.",
            ["TW_P1_E20"],
            ["res.TW_P1_E20.estimate", "res.TW_P1_E20.raw_p"],
        ),
        claim(
            "CL-02",
            "Horizon results are robustness only, Holm-adjusted within the "
            "P2 family.",
            [t for t, _ in HORIZON_TESTS if t != "TW_P1_E20"],
            [f"res.{t}.p_holm" for t, _ in HORIZON_TESTS if t != "TW_P1_E20"],
        ),
        claim(
            "CL-03",
            "Public-date robustness is reported jointly across the "
            "canonical hierarchy, raw vendor dates, and the "
            "observed-official-date sample; no variant is selectively "
            "highlighted.",
            [t for t, _ in TABLE4_TESTS],
            [f"res.{t}.estimate" for t, _ in TABLE4_TESTS],
        ),
        claim(
            "CL-04",
            "The transaction-clock comparison is descriptive, secondary, "
            "noncausal, and not publicly implementable.",
            ["TW_C1_T20", "TW_C1_PAIRED"],
            ["res.TW_C1_T20.estimate", "res.TW_C1_PAIRED.estimate"],
        ),
        claim(
            "CL-05",
            "The 40-day Buy-by-Senate interaction remains labeled "
            "HISTORICAL_EXPLORATORY_RESULT_REQUIRING_NEW_DATA and is not a "
            "primary result.",
            [],
            [],
        ),
        claim(
            "CL-06",
            "All evidence is historical; the design was locked before the "
            "canonical reconstruction run, and related historical outcomes "
            "had been examined previously.",
            [],
            [],
        ),
    ]
    if not res.get("TW_P1_E20", {}).get("raw_p", 1) == res.get("TW_P1_E20", {}).get("raw_p", 1) or (
        res.get("TW_P1_E20", {}).get("raw_p", 0) > 0.05
    ):
        rows.append(
            claim(
                "CL-07",
                "The primary spread is not statistically distinguishable "
                "from zero at conventional levels; the confidence interval "
                "in Table 2 states the economic magnitudes admitted by the "
                "data.",
                ["TW_P1_E20"],
                ["res.TW_P1_E20.ci_low", "res.TW_P1_E20.ci_high"],
            )
        )
    return pd.DataFrame(rows)


def _figures(ctx, results: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    # Figure 1: two-clock timeline schematic
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    ax.annotate("transaction\n(reported date)", (0.08, 0.55), ha="center")
    ax.annotate("official disclosure /\nvendor publication", (0.45, 0.55), ha="center")
    ax.annotate("public entry\n(next close, strictly after)", (0.62, 0.15), ha="center")
    ax.annotate("20 trading days\n(primary horizon)", (0.85, 0.55), ha="center")
    ax.annotate("", xy=(0.98, 0.4), xytext=(0.02, 0.4), arrowprops={"arrowstyle": "->"})
    for x in (0.08, 0.45, 0.62, 0.85):
        ax.plot([x], [0.4], marker="o", color="black")
    ax.set_title("Two clocks: transaction time and public-action time")
    p = out_dir / "figure1_two_clock_timeline.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # Figure 2: spread across horizons
    rows = _row_lookup(results)
    hs, est, lo, hi = [], [], [], []
    for tid, h in HORIZON_TESTS:
        r = rows.get(tid, {})
        if r.get("status") != "executed":
            continue
        hs.append(h)
        est.append(r["estimate"])
        lo.append(r.get("ci_low", np.nan))
        hi.append(r.get("ci_high", np.nan))
    fig, ax = plt.subplots(figsize=(7, 4))
    if hs:
        est_a, lo_a, hi_a = map(np.asarray, (est, lo, hi))
        ax.errorbar(
            hs, est_a, yerr=[est_a - lo_a, hi_a - est_a], fmt="o", capsize=4
        )
    ax.axhline(0.0, color="grey", lw=1)
    ax.set_xlabel("Horizon (trading days after public entry)")
    ax.set_ylabel("Purchase minus sale abnormal return")
    ax.set_title("Purchase-minus-sale estimates across horizons (95% CI, two-way CGM)")
    p = out_dir / "figure2_horizon_estimates.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # Figure 3: event-study difference path with simultaneous band
    bands = ctx.artifacts.get("event_bands", {})
    band = bands.get("difference")
    fig, ax = plt.subplots(figsize=(8, 4))
    if band is not None:
        ax.plot(band.ks, band.estimate, color="black", label="purchase-minus-sale path")
        ax.fill_between(
            band.ks, band.lower, band.upper, alpha=0.25,
            label="simultaneous 95% band (day_mult_nullimp_bwH)",
        )
    ax.axhline(0.0, color="grey", lw=1)
    ax.axvline(0.0, color="grey", lw=1, ls="--")  # public entry at relative day 0
    ax.set_xlabel("Trading days relative to public entry (day 0 = entry)")
    ax.set_ylabel("Cumulative abnormal return difference")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Disclosure-centered event study with simultaneous band")
    p = out_dir / "figure3_event_study.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    # machine-readable band path for Figure 3 (all series)
    if bands:
        pd.concat(
            [
                pd.DataFrame({
                    "series": name,
                    "relative_trading_day": np.asarray(b.ks, dtype=int),
                    "estimate": b.estimate,
                    "se": b.se,
                    "lower": b.lower,
                    "upper": b.upper,
                })
                for name, b in bands.items()
            ],
            ignore_index=True,
        ).to_csv(out_dir / "figure3_event_study_path.csv", index=False)
    return paths


def write_canonical_outputs(ctx) -> dict[str, str]:
    """Write §22's canonical output set under ``ctx.out_root / 'canonical'``."""
    out = ctx.out_root / "canonical"
    (out / "runtime_method_manifests").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    results = ctx.frames["adjusted_results"]

    ledger = build_numerical_ledger(ctx)
    (out / "numerical_ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True, default=str) + "\n"
    )
    (out / "paper_macros.tex").write_text(build_paper_macros(ledger))
    results.to_csv(out / "result_registry.csv", index=False)
    build_claim_ledger(ctx, ledger).to_csv(out / "claim_ledger.csv", index=False)
    ctx.artifacts["attrition_ledger"].to_csv(out / "attrition_ledger.csv", index=False)
    (out / "date_proxy_diagnostics.json").write_text(
        json.dumps(ctx.artifacts.get("date_proxy_diagnostics", {}), indent=2, default=str)
        + "\n"
    )
    manifest = ctx.artifacts.get("event_band_runtime_manifest")
    if manifest is not None:
        (out / "runtime_method_manifests" / "event_band.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )
    (out / "runtime_method_manifests" / "primary_inference.json").write_text(
        json.dumps(
            {
                "estimator": "two_way_cgm",
                "clusters": ["filer_uid", "exact_public_entry_trading_date"],
                "finite_sample_correction": "cgm_per_term",
                "dof_rule": "min_cluster_count_minus_one",
                "iid_allowed": False,
                "companion": "calendar_block_bootstrap",
                "mode": ctx.mode,
            },
            indent=2,
        )
        + "\n"
    )

    tables = build_tables(ctx, ledger)
    for name, (frame, caption) in tables.items():
        frame.to_csv(out / "tables" / f"{name}.csv", index=False)
        (out / "tables" / f"{name}.caption.txt").write_text(caption + "\n")
    fig_paths = _figures(ctx, results, out / "figures")

    checks = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checks[str(path.relative_to(out))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    (out / "SHA256SUMS").write_text(
        "".join(f"{v}  {k}\n" for k, v in sorted(checks.items()))
    )
    return {
        "canonical_dir": str(out),
        "n_tables": len(tables),
        "n_figures": len(fig_paths),
        "n_ledger_keys": len(ledger),
    }
