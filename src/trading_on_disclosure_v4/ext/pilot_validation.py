"""Pilot validation against the assistant's transcription of the validation pages (extension plan round 3,
task 3): row detection and field accuracy for the date, the asset text, the transaction type and the amount
bracket, for the mark-token and the ink-based assignments, with the plan's decision rule.
Writes data/v4/ext/pilot_validation.csv (counts and rates only).
"""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.ext import common as cm

VAL = cm.RESTRICTED_EXT / "two_households" / "pilot" / "validation"
OUT = cm.OUT_DIR / "pilot_validation.csv"
RULE = {"row_detection_min": 0.90, "ink_type_min": 0.85, "ink_bracket_min": 0.85}
BRACKET_LETTERS = "ABCDEFGHIJK"
BRACKET_LABELS = {"1,001": "A", "1,000": "A", "15,001": "B", "50,001": "C", "100,001": "D", "250,001": "E", "500,001": "F", "1,000,001": "G",
                  "5,000,001": "H", "25,000,001": "I", "50,000,000": "J", "over": "J"}


def norm_text(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper()) if pd.notna(s) else ""


def norm_date(s) -> str:
    """mm/dd/yyyy from mm/dd/yy, mm-dd-yyyy, m/d/yyyy; empty when unparseable."""
    if pd.isna(s):
        return ""
    iso = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(s))
    if iso:
        return f"{int(iso.group(2)):02d}/{int(iso.group(3)):02d}/{int(iso.group(1))}"
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", str(s))
    if not m:
        return ""
    mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
    y = int(y) + 2000 if len(y) == 2 else int(y)
    return f"{mo:02d}/{d:02d}/{y}"


def norm_type(s) -> str:
    if pd.isna(s):
        return ""
    t = str(s).strip().upper()
    if t.startswith("P"):
        return "P"
    if t.startswith("S"):
        return "S"
    if t.startswith("E"):
        return "E"
    return ""


def norm_bracket(s) -> str:
    if pd.isna(s):
        return ""
    t = str(s).strip().upper()
    if len(t) == 1 and t in BRACKET_LETTERS:
        return t
    for key, letter in BRACKET_LABELS.items():
        if key.upper() in t:
            return letter
    return ""


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match_page(gold: pd.DataFrame, ocr: pd.DataFrame) -> list[tuple[int, int, float]]:
    """One-to-one matches (gold index, ocr index, similarity): same date first by descending asset similarity, then any
    remaining pair with similarity >= 0.75."""
    pairs = []
    for gi, g in gold.iterrows():
        for oi, o in ocr.iterrows():
            sim = SequenceMatcher(None, g["asset_n"], o["asset_n"]).ratio()
            pairs.append((gi, oi, sim, g["date_n"] == o["date_n"] and g["date_n"] != ""))
    used_g, used_o, out = set(), set(), []
    for stage in (True, False):
        cand = sorted([p for p in pairs if p[3] == stage or (not stage and p[2] >= 0.75)], key=lambda p: -p[2])
        for gi, oi, sim, _ in cand:
            if gi in used_g or oi in used_o:
                continue
            if stage or sim >= 0.75:
                used_g.add(gi)
                used_o.add(oi)
                out.append((gi, oi, sim))
    return out


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="pilot validation against the transcription")
    ap.parse_args(argv)
    gold = pd.concat([pd.read_csv(p) for p in sorted((VAL / "gold").glob("V*.csv"))], ignore_index=True)
    ocr = pd.read_csv(VAL / "ocr_rows_SEALED.csv")
    gold["asset_n"], gold["date_n"] = gold["asset"].map(norm_text), gold["date"].map(norm_date)
    ocr["asset_n"], ocr["date_n"] = ocr["asset_text"].map(norm_text), ocr["date"].map(norm_date)
    for c in ("assigned_type", "ink_type"):
        ocr[c + "_n"] = ocr[c].map(norm_type)
    for c in ("assigned_bracket", "ink_bracket"):
        ocr[c + "_n"] = ocr[c].map(norm_bracket)
    gold["type_n"], gold["bracket_n"] = gold["type"].map(norm_type), gold["bracket"].map(norm_bracket)
    rows = []
    matched_all = []
    for pid, g in gold.groupby("validation_page_id"):
        o = ocr.loc[ocr["validation_page_id"] == pid]
        m = match_page(g, o)
        rec = {"validation_page_id": pid, "household": o["household"].iloc[0] if len(o) else "", "gold_rows": int(len(g)), "ocr_rows": int(len(o)),
               "matched": len(m), "gold_unmatched": int(len(g) - len(m)), "ocr_unmatched": int(len(o) - len(m))}
        for gi, oi, sim in m:
            gr, orow = g.loc[gi], o.loc[oi]
            matched_all.append({"validation_page_id": pid, "sim": sim, "date_ok": gr["date_n"] == orow["date_n"], "asset_exact": gr["asset_n"] == orow["asset_n"],
                                "asset_cer": levenshtein(gr["asset_n"], orow["asset_n"]) / max(len(gr["asset_n"]), 1),
                                "type_mark_ok": gr["type_n"] == orow["assigned_type_n"], "type_mark_assigned": orow["assigned_type_n"] != "",
                                "type_ink_ok": gr["type_n"] == orow["ink_type_n"], "type_ink_assigned": orow["ink_type_n"] != "",
                                "bracket_mark_ok": gr["bracket_n"] == orow["assigned_bracket_n"], "bracket_mark_assigned": orow["assigned_bracket_n"] != "",
                                "bracket_ink_ok": gr["bracket_n"] == orow["ink_bracket_n"], "bracket_ink_assigned": orow["ink_bracket_n"] != ""})
        rows.append(rec)
    mt = pd.DataFrame(matched_all)
    per = pd.DataFrame(rows)
    for pid, g in mt.groupby("validation_page_id"):
        i = per.index[per["validation_page_id"] == pid][0]
        for c in ("date_ok", "asset_exact", "type_mark_ok", "type_ink_ok", "bracket_mark_ok", "bracket_ink_ok"):
            per.loc[i, c.replace("_ok", "_acc").replace("asset_exact", "asset_exact_acc")] = float(g[c].mean())
        per.loc[i, "asset_cer_mean"] = float(g["asset_cer"].mean())
        per.loc[i, "type_mark_assigned_share"] = float(g["type_mark_assigned"].mean())
        per.loc[i, "bracket_mark_assigned_share"] = float(g["bracket_mark_assigned"].mean())
    tot = {"validation_page_id": "total", "household": "both", "gold_rows": int(per["gold_rows"].sum()), "ocr_rows": int(per["ocr_rows"].sum()), "matched": int(per["matched"].sum()),
           "gold_unmatched": int(per["gold_unmatched"].sum()), "ocr_unmatched": int(per["ocr_unmatched"].sum())}
    for c in ("date_ok", "asset_exact", "type_mark_ok", "type_ink_ok", "bracket_mark_ok", "bracket_ink_ok"):
        tot[c.replace("_ok", "_acc").replace("asset_exact", "asset_exact_acc")] = float(mt[c].mean())
    tot["asset_cer_mean"] = float(mt["asset_cer"].mean())
    tot["type_mark_assigned_share"] = float(mt["type_mark_assigned"].mean())
    tot["bracket_mark_assigned_share"] = float(mt["bracket_mark_assigned"].mean())
    per = pd.concat([per, pd.DataFrame([tot])], ignore_index=True)
    per["row_detection"] = per["matched"] / per["gold_rows"]
    per["ocr_spurious_share"] = per["ocr_unmatched"] / per["ocr_rows"].replace(0, np.nan)
    t = per.loc[per["validation_page_id"] == "total"].iloc[0]
    per["rule_row_detection_min"], per["rule_ink_type_min"], per["rule_ink_bracket_min"] = RULE["row_detection_min"], RULE["ink_type_min"], RULE["ink_bracket_min"]
    per["decision_recovery_proceeds"] = bool(t["row_detection"] >= RULE["row_detection_min"] and t["type_ink_acc"] >= RULE["ink_type_min"] and t["bracket_ink_acc"] >= RULE["ink_bracket_min"])
    per["transcriber"] = "the assistant, from the page images, before opening the OCR rows"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT, index=False, lineterminator="\n")
    cm.write_manifest("pilot_validation", [cm.file_entry(VAL / "ocr_rows_SEALED.csv"), *[cm.file_entry(p) for p in sorted((VAL / "gold").glob("V*.csv"))]])
    pd.set_option("display.width", 250)
    print(per[["validation_page_id", "gold_rows", "ocr_rows", "matched", "row_detection", "ocr_spurious_share", "date_acc", "asset_exact_acc", "asset_cer_mean",
               "type_mark_acc", "type_ink_acc", "bracket_mark_acc", "bracket_ink_acc", "decision_recovery_proceeds"]].round(3).to_string(index=False))
    return per


if __name__ == "__main__":
    main()
