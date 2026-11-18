"""Release manifests: immutable dirs, comparison, rerun detection (§M).

REL-002 manifest verification, REP-004 immutable release IDs, JC-022 never
overwrite prior artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path


class ReleaseError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path) -> dict[str, str]:
    root = Path(root)
    if not root.is_dir():
        raise ReleaseError(f"{root} is not a directory")
    return {
        str(p.relative_to(root)): _sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.startswith(".")
    }


def write_manifest(root: Path, out_path: Path | None = None) -> Path:
    manifest = build_manifest(root)
    out = out_path or (root / "MANIFEST.sha256.json")
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def verify_manifest(root: Path, manifest_path: Path | None = None) -> None:
    path = manifest_path or (Path(root) / "MANIFEST.sha256.json")
    recorded = json.loads(path.read_text())
    current = {
        k: v
        for k, v in build_manifest(Path(root)).items()
        if k != path.name
    }
    recorded = {k: v for k, v in recorded.items() if k != path.name}
    if recorded != current:
        drift = compare_manifests(recorded, current)
        raise ReleaseError(f"manifest verification failed: {drift}")


def compare_manifests(old: dict[str, str], new: dict[str, str]) -> dict:
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": sorted(k for k in set(old) & set(new) if old[k] != new[k]),
    }


def create_release_dir(base: Path, release_id: str) -> Path:
    """Immutable release directory: existing IDs are never reused (REP-004)."""
    target = Path(base) / release_id
    if target.exists():
        raise ReleaseError(
            f"release id {release_id!r} already exists; ids are immutable and "
            "may not be reused"
        )
    target.mkdir(parents=True)
    return target


def assert_prior_releases_preserved(base: Path, known_manifests: dict[str, dict]) -> None:
    """Every previously recorded release must still verify byte-for-byte."""
    for release_id, manifest in known_manifests.items():
        root = Path(base) / release_id
        if not root.exists():
            raise ReleaseError(f"prior release {release_id!r} disappeared")
        current = {
            k: v
            for k, v in build_manifest(root).items()
            if k != "MANIFEST.sha256.json"
        }
        if current != {k: v for k, v in manifest.items() if k != "MANIFEST.sha256.json"}:
            raise ReleaseError(f"prior release {release_id!r} was modified")


class StagedRelease:
    """Failure-safe staging: write into a temp dir, atomic-rename on success,
    delete on failure — a crashed run leaves no half-written release."""

    def __init__(self, base: Path, release_id: str) -> None:
        self.base = Path(base)
        self.release_id = release_id
        self.final_path = self.base / release_id
        self._tmp: Path | None = None

    def __enter__(self) -> Path:
        if self.final_path.exists():
            raise ReleaseError(f"release id {self.release_id!r} already exists")
        self.base.mkdir(parents=True, exist_ok=True)
        self._tmp = Path(
            tempfile.mkdtemp(prefix=f".staging_{self.release_id}_", dir=self.base)
        )
        return self._tmp

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._tmp is not None
        if exc_type is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            return False
        write_manifest(self._tmp)
        self._tmp.rename(self.final_path)
        return False


def rerun_fingerprint(input_hashes: dict[str, str], config_hash: str) -> str:
    payload = json.dumps({"inputs": input_hashes, "config": config_hash}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def check_rerun(
    registry_path: Path, fingerprint: str, release_id: str
) -> None:
    """Same inputs+config → refuse a silent rerun; a deliberate rerun needs a
    NEW release id and is recorded."""
    registry: dict[str, str] = {}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    if fingerprint in registry and registry[fingerprint] != release_id:
        raise ReleaseError(
            f"identical inputs+config already ran as release "
            f"{registry[fingerprint]!r}; rerunning under {release_id!r} must be "
            "deliberate — record a new release id only after review"
        )
    registry[fingerprint] = release_id
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
