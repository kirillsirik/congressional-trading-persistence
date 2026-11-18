"""Reader-facing labels, independent of the stable analysis and macro keys."""

from __future__ import annotations

import re

DATE_LABELS = {"T": "Trade date", "P": "Disclosure date"}


def display_text(text: str) -> str:
    """Format table prose without changing CSV keys, LaTeX commands or numbers.

    Historical exports retain their original keys.  Only text sent to a
    manuscript table passes through this function; data are never rewritten.
    """
    for pattern, replacement in (
        (r"\b[Pp]ublic[- ]action clock\b", "Disclosure date"),
        (r"\b[Pp]ublic[- ]clock\b", "Disclosure date"),
        (r"\b[Tt]ransaction[- ]clock\b", "Trade date"),
        (r"\bTrade date (?=purchases\b)", "Trade-date "),
        (r"\bDisclosure date (?=purchases\b)", "Disclosure-date "),
        (r"\bDe-duplicated\b", "Deduplicated"),
        (r"\bde-duplicated\b", "deduplicated"),
        (r"\bPublic-date\b", "Disclosure-date"),
        (r"\bPublic date\b", "Disclosure date"),
        (r"\bPublic-entry\b", "Disclosure-date"),
        (r"\bAudited, baseline cohort\b", "Extended sample (baseline cohort)"),
        (r"\bAudited, expanded cohort\b", "Extended sample"),
        (r"\bAudited\b", "Extended sample"),
        (r"\bInitial cohort\b", "Baseline cohort"),
        (r"\bOriginal cohort\b", "Baseline cohort"),
        (r"\bInitial match\b", "Baseline match"),
        (r"\bBaseline\b(?! (?:cohort|match))", "Baseline match"),
        (r"\binitial cohort\b", "baseline cohort"),
        (r"\binitial match\b", "baseline match"),
        (r"\bbaseline cutoffs\b", "original cutoffs"),
        (r"\bidentities[- ]only\b", "Ticker-normalized CRSP matches"),
        (r"\brepaired link records to unparsed filing\b", "records linked to an unparsed filing"),
        (r"\bRepaired\b", "Extended sample"),
        (r"\brepaired lines\b", "newly matched records"),
        (r"\brepaired official line\b", "newly matched official record"),
        (r"\brepaired line\b", "newly matched record"),
        (r"\brepaired lots\b", "newly matched lots"),
        (r"\ball lots repaired\b", "all lots newly matched"),
        (r"\bbaseline / repaired\b", "baseline match / extended sample"),
        (r"\bhouseholds: baseline\b(?! match)", "households: baseline match"),
        (r"\bhouseholds: repaired\b", "households: extended sample"),
        (r"\bbaseline (?=\d)", "baseline match "),
        (r"\bidentity by ticker and date validity only\b", "CRSP matching by ticker and date validity only"),
        (r"\bexact identities\b", "exact security matches"),
        (r"\bnear-exact identities\b", "near-exact security matches"),
    ):
        text = re.sub(pattern, replacement, text)
    # Standalone table cells are labels; embedded command names remain intact.
    labels = {"Transaction": "Trade date", "Public": "Disclosure date", "Clock": "Return date", "clock": "Return date", "T": "Trade date", "P": "Disclosure date"}
    text = re.sub(r"(^|[&\n{])(\s*)(Transaction|Public|Clock|clock|T|P)(\s*)(?=&|\\\\|$)",
                  lambda m: m[1] + m[2] + labels[m[3]] + m[4], text)
    return text
