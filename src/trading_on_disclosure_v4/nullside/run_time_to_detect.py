"""A6 runner: sigma table, tidy time-to-detect grid, Figure 4 draft (pdf + png). Pre-tag."""

from __future__ import annotations

import json
from pathlib import Path

from trading_on_disclosure_v4.nullside import figure4
from trading_on_disclosure_v4.nullside import time_to_detect as td
from trading_on_disclosure_v4.outcomes import panel

OUT_DIR = panel.REPO_ROOT / "data" / "v4" / "nullside"


def main(out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sigmas = td.sigma_table()
    sigmas.to_csv(out_dir / "sigma_by_clock.csv", index=False, float_format="%.8g")
    h = td.r1_household_counts()
    table = td.ttd_table(sigmas, h)
    table.to_csv(out_dir / "time_to_detect.csv", index=False, float_format="%.8g")
    activity = td.activity_histogram()
    files = []
    for clock in ("T", "P"):
        for variant in ("raw", "winsorized"):
            fig = figure4.plot_figure4(table, activity, clock=clock, sigma_variant=variant)
            files += figure4.save_figure4(fig, out_dir / f"figure4_{clock}_{variant}")
    info = {"unit_test_years": td.unit_test_value(), "H_R1": h, "rows": int(len(table)),
            "sigma_AB_bp": {f"{r.clock}_{r.sigma_variant}": round(r.sigma_bp, 3) for r in sigmas[sigmas["period"] == "AB"].itertuples()},
            "figures": [str(p) for p in files]}
    (out_dir / "time_to_detect_meta.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    main()
