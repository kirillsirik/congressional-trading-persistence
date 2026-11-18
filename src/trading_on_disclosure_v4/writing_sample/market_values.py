"""Daily CRSP market values (DlyCap) of the traded-security universe, from the same trimmed extract the market panel
was built from (no new data). Used only to define largest-market-value-third subsets at formation.

python -m trading_on_disclosure_v4.writing_sample.market_values
Writes data/v4/restricted/writing_sample/market_values.parquet (permno, date, dlycap in $000s; restricted).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.outcomes import panel
from trading_on_disclosure_v4.writing_sample import common as cm

PATH = cm.RESTRICTED / "market_values.parquet"
MANIFEST = cm.OUT / "market_values_manifest.json"
SOURCE_NAME = "crsp_trimmed_2012plus.csv"


def source_path():
    return panel.project_root() / "data" / "raw" / "crsp" / SOURCE_NAME


def build(log) -> pd.DataFrame:
    universe = set(int(p) for p in panel.universe_permnos())
    parts = []
    n_rows = 0
    t0 = time.time()
    for chunk in pd.read_csv(source_path(), usecols=["PERMNO", "DlyCalDt", "DlyCap"], chunksize=2_000_000,
                             dtype={"PERMNO": np.int64, "DlyCalDt": str, "DlyCap": float}):
        n_rows += len(chunk)
        c = chunk[chunk["PERMNO"].isin(universe)]
        parts.append(c)
    mv = pd.concat(parts, ignore_index=True).rename(columns={"PERMNO": "permno", "DlyCalDt": "date", "DlyCap": "dlycap"})
    mv["date"] = pd.to_datetime(mv["date"])
    n_dup = int(mv.duplicated(["permno", "date"]).sum())
    mv = mv.drop_duplicates(["permno", "date"], keep="first").sort_values(["permno", "date"], ignore_index=True)
    log(f"read {n_rows:,} extract rows in {time.time() - t0:.0f} s; kept {len(mv):,} universe rows; {n_dup} duplicate (permno, date) rows dropped (keep first, as the panel builder)")
    return mv, {"rows_read": n_rows, "rows_kept": int(len(mv)), "duplicates_dropped": n_dup,
                "universe_permnos": len(universe), "permnos_with_values": int(mv["permno"].nunique()),
                "date_min": str(mv["date"].min().date()), "date_max": str(mv["date"].max().date()),
                "nonpositive_or_missing_dlycap": int((~(mv["dlycap"] > 0)).sum())}


def load() -> pd.DataFrame:
    return pd.read_parquet(PATH)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    log = cm.logger("market_values")
    mv, rep = build(log)
    cm.RESTRICTED.mkdir(parents=True, exist_ok=True)
    mv.to_parquet(PATH, index=False)
    cm.write_json({"output": cm.file_entry(PATH, rows=len(mv)), "source": {"name": SOURCE_NAME, "sha256": cm.sha256_file(source_path()),
                   "bytes": source_path().stat().st_size}, "report": rep,
                   "definition": "CRSP v2 DlyCap (price x shares outstanding, $000s) on each session, for the PERMNOs of the "
                                 "traded-security universe (signals.csv + matched_transactions.csv)"}, MANIFEST)
    log(f"wrote {cm.rel(PATH)}")


if __name__ == "__main__":
    main()
