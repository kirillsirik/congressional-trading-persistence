"""Run manifests: every major pipeline stage records what it read and wrote.

A manifest ties an output to (data snapshot, config, git commit, code module),
which is the audit trail required for every paper table/figure.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .local_io import atomic_write_text, file_sha256


def utc_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


#: Keys that describe HOW a config was loaded rather than WHAT it configures.
#: `inherits` is a loader directive: one stage's loader pops it while merging
#: and another leaves it in place, so the same file hashed differently
#: depending on which stage read it. A config hash has to identify the
#: configuration or it identifies nothing.
#: Also excluded from the EFFECTIVE hash: the output namespace is run routing,
#: not an analytical parameter, so two runs that differ only in where they
#: wrote must share an effective configuration hash.
RUNTIME_ROUTING_KEYS = ("output_namespace",)
LOADER_DIRECTIVE_KEYS = ("inherits",)

#: ``raw_inputs`` maps input roles to FILESYSTEM PATHS on the machine that ran
#: the stage. Paths are routing, not analysis: the same data at a different
#: path is the same analysis, and a path string is not evidence about content.
#:
#: It is excluded because including it made the effective hash machine- and
#: loader-dependent. The gitignored ``configs/paths.yaml`` contributes
#: machine-local entries, so one loader that merges it and one that does not
#: produced two effective hashes from the SAME four config files at the SAME
#: byte hashes -- which is exactly the P0 finding this exclusion resolves.
#:
#: This does not weaken input identity, which never lived here: WHICH data was
#: read travels as CONTENT hashes -- ``set_source_provenance`` records the
#: stage input's sha256 on every estimate, and run manifests hash their declared
#: inputs. A changed input file moves those hashes whether or not its path
#: changed, and a changed path with identical content correctly moves neither.
MACHINE_LOCAL_ROUTING_KEYS = ("raw_inputs",)


def config_sha256(config: dict[str, Any] | None) -> str:
    """Hash of the EFFECTIVE configuration.

    Excludes only non-substantive fields: loader bookkeeping (``_``-prefixed
    keys, ``inherits``), runtime routing (``run.output_namespace``) and
    machine-local input locations (``raw_inputs``). Two runs of the same
    analysis that differ only in where they read or wrote share this hash; any
    difference in an analytical parameter moves it.

    This is one of THREE layers, none of which substitutes for another. It
    answers "did these stages run the same analysis?".
    :func:`config_file_provenance` answers "which files, byte for byte,
    produced it?". Source provenance and manifest input hashes answer "which
    data did it read?".
    """
    excluded = set(LOADER_DIRECTIVE_KEYS) | set(MACHINE_LOCAL_ROUTING_KEYS)
    payload = {k: v for k, v in (config or {}).items()
               if not k.startswith("_") and k not in excluded}
    run_cfg = payload.get("run")
    if isinstance(run_cfg, dict):
        payload = {**payload,
                   "run": {k: v for k, v in run_cfg.items()
                           if k not in RUNTIME_ROUTING_KEYS}}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def config_file_provenance(config_path: Any) -> list[dict[str, Any]]:
    """Raw SHA-256 and inheritance chain of every source configuration file.

    The effective hash deliberately ignores loader and routing fields, so on
    its own it cannot prove which files were read. This walks the ``inherits``
    chain from the entry config to its root and hashes each file verbatim.
    """
    import yaml

    chain: list[dict[str, Any]] = []
    path = Path(config_path) if config_path else None
    seen: set[Path] = set()
    depth = 0
    while path is not None and path.exists() and path not in seen:
        seen.add(path)
        raw = path.read_bytes()
        try:
            parsed = yaml.safe_load(raw.decode()) or {}
        except Exception:                                    # noqa: BLE001
            parsed = {}
        parent = parsed.get("inherits") if isinstance(parsed, dict) else None
        chain.append({
            "depth": depth,
            "path": str(path),
            "name": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "inherits": parent,
        })
        depth += 1
        path = (path.parent / parent) if parent else None
    return chain


def git_state(repo_root: Path) -> dict[str, str]:
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return "unknown"

    # `dirty` means TRACKED source differs from the commit. Counting untracked
    # files made every pinned release report dirty=yes: a pinned worktree is
    # given an untracked .venv symlink, .env and configs/paths.yaml, none of
    # which is part of the source it claims to be. Untracked paths are recorded
    # separately so nothing is hidden -- a reader can still see what was there.
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    untracked = _git("status", "--porcelain", "--untracked-files=all")
    untracked_only = [ln for ln in untracked.splitlines() if ln.startswith("??")]
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": "yes" if tracked else "no",
        "tracked_changes": tracked.splitlines(),
        "untracked_paths": untracked_only,
    }


def hash_files(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        if p.exists() and p.is_file():
            out[str(p)] = file_sha256(p)
        else:
            out[str(p)] = "missing"
    return out


class RunManifest:
    """Accumulates run metadata and writes a JSON manifest at the end."""

    def __init__(
        self,
        stage: str,
        repo_root: Path,
        config: dict[str, Any] | None = None,
        paths: Any | None = None,
        profile: str | None = None,
        release_id: str | None = None,
    ):
        self.run_id = utc_run_id(stage)
        self.stage = stage
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.repo_root = repo_root

        # Freeze §8: every manifest embeds profile, run ID, release ID, config
        # hash, code commit and input hashes.
        resolved_profile = profile or getattr(paths, "profile", None) or "unspecified"
        self.profile = resolved_profile
        self.release_id = (
            release_id
            or os.environ.get("CT_RELEASE_ID")
            or f"{resolved_profile}_{self.run_id}"
        )
        roots = {}
        if paths is not None:
            roots = {
                "data_root": str(paths.data_root),
                "output_root": str(paths.output_root),
                # The NAMESPACED output root, which is where this run actually
                # wrote. `output_root` is the profile root and is identical
                # across a primary run and every sensitivity, so a consumer
                # asking "which run produced this?" cannot answer from it.
                "output_root_namespaced": str(paths._out),
                "log_root": str(paths.log_root),
                "manifest_root": str(paths.manifest_root),
                "raw_root": str(paths.raw),
            }

        self.data: dict[str, Any] = {
            "run_id": self.run_id,
            "stage": stage,
            "profile": resolved_profile,
            "release_id": self.release_id,
            "profile_roots": roots,
            # Run identity, first-class rather than inferred from a path: the
            # gate has to identify every stage of one namespaced run, and
            # scanning output paths for a substring is not an identity. All
            # three are recorded because they can legitimately differ -- the
            # environment overrides the config, and only routing is affected.
            "output_namespace": (paths.output_namespace if paths is not None
                                 else None),
            "output_namespace_config": (
                ((config or {}).get("run") or {}).get("output_namespace")),
            "output_namespace_env": os.environ.get("CT_OUTPUT_NAMESPACE"),
            "started_utc": self.started_utc,
            "user": getpass.getuser(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git": git_state(repo_root),
            "config": {k: v for k, v in (config or {}).items() if not k.startswith("_")},
            "config_path": (config or {}).get("_config_path"),
            # TWO LAYERS. `config_sha256` is the effective analysis; the chain
            # below is the byte-level provenance of the files that produced it.
            "config_sha256": config_sha256(config),
            "config_file_provenance": config_file_provenance(
                (config or {}).get("_config_path")),
            "inputs": {},
            "outputs": {},
            "stats": {},
            "warnings": [],
            "errors": [],
        }

    def add_inputs(self, paths: list[Path]) -> None:
        self.data["inputs"].update(hash_files(paths))

    def add_outputs(self, paths: list[Path]) -> None:
        self.data["outputs"].update(hash_files(paths))

    def add_stats(self, **kwargs: Any) -> None:
        self.data["stats"].update(kwargs)

    def warn(self, msg: str) -> None:
        self.data["warnings"].append(msg)

    def error(self, msg: str) -> None:
        self.data["errors"].append(msg)

    def write(self, manifest_root: Path) -> Path:
        self.data["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_root.mkdir(parents=True, exist_ok=True)
        path = manifest_root / f"{self.run_id}.json"
        atomic_write_text(path, json.dumps(self.data, indent=2, default=str))
        return path
