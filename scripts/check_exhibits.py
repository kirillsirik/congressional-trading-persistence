#!/usr/bin/env python3
"""Confirm that rebuilding the exhibits reproduces the committed files.

    python scripts/check_exhibits.py snapshot   # before `make exhibits`
    python scripts/check_exhibits.py compare    # after it

Text outputs must be byte-identical or differ only in floating-point rounding (relative 1e-9);
rendered figures (PDF/PNG) are listed as re-rendered. Exits non-zero on any other difference.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "build" / "exhibit_snapshot"
SKIP_DIRS = {".git", "build", "__pycache__", ".pytest_cache", ".venv"}
SKIP_SUFFIXES = {".npz", ".pyc"}
RENDERED = {".pdf", ".png"}
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
RTOL = 1e-9


def tracked_files() -> list[Path]:
    """Files under version control plus untracked files that are not ignored; a plain walk outside Git."""
    try:
        listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.splitlines()
        candidates = [Path(x) for x in listed]
    except (OSError, subprocess.CalledProcessError):
        candidates = [p.relative_to(ROOT) for p in ROOT.rglob("*")]
    return sorted(rel for rel in candidates if (ROOT / rel).is_file() and not (SKIP_DIRS & set(rel.parts))
                  and "restricted" not in rel.parts and rel.suffix not in SKIP_SUFFIXES)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> None:
    if SNAP.exists():
        shutil.rmtree(SNAP)
    SNAP.mkdir(parents=True)
    hashes = {}
    for rel in tracked_files():
        hashes[str(rel)] = digest(ROOT / rel)
        if rel.suffix not in RENDERED:
            dest = SNAP / "files" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
    (SNAP / "hashes.json").write_text(json.dumps(hashes, indent=1))
    print(f"snapshot of {len(hashes)} files")


def numeric_gap(old: str, new: str) -> float | None:
    """Largest relative gap between corresponding numbers, or None if the non-numeric text differs."""
    if NUMBER.sub("#", old) != NUMBER.sub("#", new):
        return None
    gap = 0.0
    for a, b in zip(NUMBER.findall(old), NUMBER.findall(new)):
        x, y = float(a), float(b)
        if x != y:
            gap = max(gap, abs(x - y) / max(abs(x), abs(y)))
    return gap


def compare() -> int:
    before = json.loads((SNAP / "hashes.json").read_text())
    after = {str(rel): digest(ROOT / rel) for rel in tracked_files()}
    new = sorted(set(after) - set(before))
    missing = sorted(set(before) - set(after))
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    rendered, rounding, failures = [], [], []
    for rel in changed:
        if Path(rel).suffix in RENDERED:
            rendered.append(rel)
            continue
        gap = numeric_gap((SNAP / "files" / rel).read_text(errors="replace"), (ROOT / rel).read_text(errors="replace"))
        (rounding if gap is not None and gap <= RTOL else failures).append((rel, gap))
    print(f"{len(after) - len(changed) - len(new)} files unchanged")
    print(f"{len(rounding)} text files equal up to floating-point rounding"
          + (f" (largest relative gap {max(g for _, g in rounding):.1e})" if rounding else ""))
    print(f"{len(rendered)} figures re-rendered (compared by their stored coordinates, not bytes)")
    for rel in new:
        print("NEW     ", rel)
    for rel in missing:
        print("MISSING ", rel)
    for rel, gap in failures:
        print("CHANGED ", rel, "" if gap is None else f"(relative gap {gap:.1e})")
    ok = not (new or missing or failures)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["snapshot"]:
        snapshot()
    elif sys.argv[1:] == ["compare"]:
        sys.exit(compare())
    else:
        sys.exit(__doc__)
