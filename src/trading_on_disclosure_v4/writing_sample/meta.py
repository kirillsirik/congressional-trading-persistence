"""Sample counts, exclusions, cutoffs and cache-verification facts of the 11 November 2026 exports, as result rows.

python -m trading_on_disclosure_v4.writing_sample.meta
Reads the export and fit-cache manifests (no restricted rows) and writes component `continuation_meta`.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from trading_on_disclosure_v4.writing_sample import common as cm


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    ex = json.loads((cm.OUT / "exports_manifest.json").read_text())
    fit = json.loads((cm.OUT / "fit_cache_extended_manifest.json").read_text())
    rows = cm.Rows(f"data/v4/writing_sample/{cm.RUN_ID}/exports_manifest.json")
    for key, rec in ex["exclusions"].items():
        clock, sample = key.split("/")
        for k, v in rec.items():
            rows.add("continuation_meta", k, v, "count", clock=clock, sample=sample, term="sample_flow",
                     inference="purchases of periods A/B in the event frame -> finite event-study outcome -> duplicate rule -> household eligibility")
    for clock, rec in ex["reason_counts_h20_purchases"].items():
        for k, v in rec.items():
            rows.add("continuation_meta", f"purchases_reason_{k}", v, "count", clock=clock, term="cache_reason_h20", sample="purchases_periods_AB")
    for key, rec in ex["continuation_cutoffs_T"].items():
        sample, period = key.split("/")
        for k, v in rec.items():
            rows.add("continuation_meta", k, v, "count" if k == "purchases" else "bp", clock="T", sample=sample, scale="winsorized", term=f"cutoffs/period_{period}",
                     inference="x: p1/p99 of 20*alpha by period on the declared sample (new); Y0, Y1: committed pooled purchase cutoffs by period (original)")
    for key, v in ex["large_stock_missing"].items():
        clock, sample = key.split("/")
        rows.add("continuation_meta", "large_stock_missing", v, "count", clock=clock, sample=sample, term="size")
    for k, v in ex["grid"].items():
        if isinstance(v, (int, float)):
            rows.add("continuation_meta", k, v, "count", clock="T", sample="traded_securities", term="grid",
                     inference="formation grid: every 20th session from the first 2015 session through 2025; traded-security universe (2,481 PERMNOs ever in signals or matched transactions)")
    for k, v in ex["spread_counts"].items():
        if isinstance(v, (int, float)):
            rows.add("continuation_meta", k, v, "count", clock="P", sample="spread_sample", term="spread_counts")
    ver = fit["verification_against_forward_cache"]
    for col in ("bhar_ff5mom_h10", "bhar_ff5mom_h20", "bhar_ff5mom_h60", "bhar_ff5mom_noalpha_h20", "resid_sd"):
        for k, v in ver[col].items():
            rows.add("continuation_meta", k, v, "count" if k != "max_abs_diff_decimal" else "decimal", term=f"fit_cache_reproduction/{col}", clock="",
                     inference="extended fit cache vs stored forward cache, cell by cell", source=f"data/v4/writing_sample/{cm.RUN_ID}/fit_cache_extended_manifest.json")
    for k in ("cells_cache", "cells_extended", "cells_only_one_side", "n_est_mismatches", "cells_h20_ok", "alpha_finite_where_h20_ok"):
        rows.add("continuation_meta", k, ver[k], "count", term="fit_cache_reproduction", clock="", source=f"data/v4/writing_sample/{cm.RUN_ID}/fit_cache_extended_manifest.json")
    new = pd.read_csv(cm.OUT / "pooled_cutoffs_new_horizons.csv")
    for _, r in new[new.reproduces_committed.isna()].iterrows():
        for k in ("winsor_lo", "winsor_hi"):
            rows.add("continuation_meta", k, r[k] * cm.BP, "bp", clock=r.clock, scale="winsorized", term=f"new_pooled_cutoffs/{r.horizon}/period_{r.period}",
                     horizon=r.horizon[-2:], benchmark="factor_only", inference="pooled purchase p1/p99 (committed-table routine and %.8g storage), new for the factor-only 10/60-session outcomes",
                     source=f"data/v4/writing_sample/{cm.RUN_ID}/pooled_cutoffs_new_horizons.csv")
    rows.write_component("continuation_meta")


if __name__ == "__main__":
    main()
