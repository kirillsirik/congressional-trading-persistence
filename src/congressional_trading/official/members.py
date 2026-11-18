"""Member identity resolution: filer name → Bioguide ID with point-in-time
party/state/district from term records.

Identity metadata comes from the congress-legislators dataset (maintained at
https://github.com/unitedstates/congress-legislators, aggregating the official
Biographical Directory / Congress.gov identifiers). It is an *enrichment*
source for member metadata — never a source of transaction observations — and
is cached locally with provenance. Matching is deterministic and time-aware;
unresolved filers keep their raw names and enter the review flow rather than
being guessed.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

LEGISLATORS_URLS = [
    "https://unitedstates.github.io/congress-legislators/legislators-current.json",
    "https://unitedstates.github.io/congress-legislators/legislators-historical.json",
]

MATCH_VERSION = "1"


def _norm_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z\s'-]", " ", text.lower())
    text = re.sub(r"\b(hon|mr|mrs|ms|dr|jr|sr|ii|iii|iv)\b\.?", " ", text)
    return " ".join(text.split())


class MemberResolver:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.people: list[dict] = []
        self._index: dict[tuple[str, str], list[dict]] = {}

    def load(self, session=None) -> None:
        """Load cached legislators files; fetch through the polite session if
        absent and a session is provided."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for url in LEGISLATORS_URLS:
            fname = self.cache_dir / url.rsplit("/", 1)[-1]
            if not fname.exists():
                if session is None:
                    raise FileNotFoundError(
                        f"{fname} missing and no session provided to fetch it"
                    )
                result = session.get(url)
                if result.error_category or result.content is None:
                    raise RuntimeError(f"failed to fetch {url}: {result.error_detail}")
                fname.write_bytes(result.content)
                logger.info("cached %s (%d bytes)", fname.name, len(result.content))
            self.people.extend(json.loads(fname.read_text()))
        self._build_index()

    def _build_index(self) -> None:
        for person in self.people:
            last = _norm_name(person.get("name", {}).get("last", ""))
            for term in person.get("terms", []):
                chamber = "house" if term.get("type") == "rep" else "senate"
                self._index.setdefault((chamber, last), []).append(
                    {"person": person, "term": term}
                )

    def resolve(
        self,
        chamber: str,
        first_name: str | None,
        last_name: str | None,
        on_date: str | None,
        state: str | None = None,
    ) -> dict:
        """Deterministic, time-aware match. Returns match record with method
        and confidence; unresolved matches keep raw names."""
        out = {
            "bioguide_id": None, "member_id": None, "party": None, "state": None,
            "district": None, "identity_match_method": "unresolved",
            "identity_match_confidence": 0.0, "match_version": MATCH_VERSION,
        }
        last = _norm_name(last_name or "")
        first = _norm_name(first_name or "")
        if not last:
            return out
        candidates = self._index.get((chamber, last), [])
        if on_date:
            in_term = [
                c for c in candidates
                if c["term"].get("start", "9999") <= on_date <= c["term"].get("end", "0000")
            ]
        else:
            in_term = candidates
        if state:
            state_u = state.strip().upper()[:2]
            in_state = [c for c in in_term if c["term"].get("state", "").upper() == state_u]
        else:
            in_state = in_term

        pool = in_state or in_term
        people = {c["person"]["id"]["bioguide"]: c for c in pool}
        if len(people) > 1 and first:
            people = {
                bid: c for bid, c in people.items()
                if _norm_name(c["person"]["name"].get("first", "")).startswith(first.split()[0])
                or first.split()[0] in _norm_name(
                    c["person"]["name"].get("nickname", "") or ""
                )
            } or people
        if len(people) == 1:
            bid, c = next(iter(people.items()))
            term = c["term"]
            method = "last+term" + ("+state" if in_state else "") + ("+first" if first else "")
            out.update(
                bioguide_id=bid,
                member_id=bid,
                party=_party_label(term.get("party")),
                state=term.get("state"),
                district=term.get("district"),
                identity_match_method=method,
                identity_match_confidence=0.95 if (in_state and first) else 0.85,
            )
        elif len(people) > 1:
            out["identity_match_method"] = "ambiguous"
        return out


def _party_label(raw: str | None) -> str | None:
    if raw is None:
        return None
    if raw in {"Democrat", "Republican"}:
        return raw
    return "Other"
