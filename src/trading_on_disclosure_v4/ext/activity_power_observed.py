"""The maximum-household statistic at recorded trade dates beside the critical value that the activity-power
figure uses for the persistence sample (the 95th percentile of the same maximum over size_common draws 1000-1999).
Writes data/v4/ext/activity_power_observed.csv.

CLI: python -m trading_on_disclosure_v4.ext.activity_power_observed [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import power_common as pc
from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.calibration import run_power_v2 as rp2
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import winners
from trading_on_disclosure_v4.ext import common as ec
from trading_on_disclosure_v4.outcomes import panel as panel_mod

OUT_CSV = ec.OUT_DIR / "activity_power_observed.csv"
CLOCK = "T"
ALPHA = 0.05
MIN_N = winners.MIN_EVENTS_FULL_SAMPLE
PERSISTENCE = "persistence_households_purchases"


def critical_value(clock: str = CLOCK) -> dict:
    ref = rp2.reference_from_size_v2(rc.load_grid(pc.SIZE_GRID), clock)
    return {"clock": clock, "sample": PERSISTENCE, "statistic": "max_t_critical_value", "value": float(np.nanquantile(ref["max_t_wins"].to_numpy(dtype=float), 1.0 - ALPHA)),
            "n_households": np.nan, "n_events": np.nan, "source": f"size_common winsorized draws {int(ref['draw'].min())}-{int(ref['draw'].max())}"}


def observed(clock: str = CLOCK) -> list[dict]:
    calendar = panel_mod.v2_trading_calendar()
    sp.register_household_exclusion(sp.EXCL_TWO_VENDOR, evf.blind_vendor_households(2))
    ctx = rs.Context(frames={clock: evf.load_event_frame(clock, calendar=calendar)}, calendar=calendar, cutoffs=evf.pooled_cutoffs,
                     placebo_source=lambda *a: iter(()))
    full = ctx.panel(sp.analysis_specs(clock)["best_household"])
    pers = ctx.panel(sp.headline_spec(clock))
    hf = winners.household_stats(full, min_n=MIN_N).set_index("household_id")
    hp = winners.household_stats(pers, min_n=MIN_N).set_index("household_id")
    shared = hp.index.intersection(hf.index)
    same = int(((hp.loc[shared, ["n", "mean", "t"]] - hf.loc[shared, ["n", "mean", "t"]]).abs().max(axis=1) == 0).sum())
    return [{"clock": clock, "sample": "households_with_20_purchases", "statistic": "max_t", "value": float(hf["t"].max()),
             "n_households": int(len(hf)), "n_events": int(len(full)), "source": "recorded trade dates"},
            {"clock": clock, "sample": PERSISTENCE, "statistic": "max_t", "value": float(hp["t"].max()),
             "n_households": int(len(hp)), "n_events": int(len(pers)), "source": "recorded trade dates"},
            {"clock": clock, "sample": PERSISTENCE, "statistic": "households_with_identical_t_in_full_set", "value": float(same),
             "n_households": int(len(hp)), "n_events": np.nan, "source": "recorded trade dates"}]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args(argv)
    out = pd.DataFrame(observed() + [critical_value()]).astype({"n_households": "Int64", "n_events": "Int64"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, lineterminator="\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
