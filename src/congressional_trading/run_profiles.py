"""Run profiles: enforced isolation between run tracks.

Each run belongs to a profile that owns its data, output, log and manifest roots, so
artifacts of different tracks cannot land in the same folders and every artifact records
the track it belongs to.

A profile names one run track and resolves to its own data, output, log and
manifest roots. Isolation is checked on the directories a run actually
**writes** — not on the namespace roots, which legitimately nest under a shared
``data/`` — so the check holds for the layout that already exists on disk:

* no two profiles may write into the same directory, or into a directory that
  contains another profile's write directory;
* no profile may write into a shared *input* root (raw market data, the
  official collection store), which every profile reads;
* an environment override that lands a profile's write directory inside another
  profile's territory fails the run.

Profiles are declared in :data:`PROFILES`, per decision freeze §8.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Data subdirectories that every profile reads and no profile may write.
SHARED_INPUT_SUBDIRS = ("raw", "official")

DEFAULT_PROFILE = "ct_only"

_ENV_PROFILE = "CT_RUN_PROFILE"


@dataclass(frozen=True)
class RunProfile:
    """One run track and the four roots it owns."""

    name: str
    data_subdir: str | None
    output_subdir: str | None
    log_subdir: str
    manifest_subdir: str
    release_prefix: str
    frozen: bool = False
    description: str = ""


PROFILES: dict[str, RunProfile] = {
    "ct_only": RunProfile(
        name="ct_only",
        data_subdir=None,          # the historical top-level layout
        output_subdir=None,
        log_subdir="ct_only",
        manifest_subdir="ct_only",
        release_prefix="ct",
        description="Capitol-Trades-only track (the original paper sample).",
    ),
    "official_only": RunProfile(
        name="official_only",
        data_subdir="official_analysis",
        output_subdir="official_analysis",
        log_subdir="official_only",
        manifest_subdir="official_only",
        release_prefix="off",
        description="Official House Clerk + Senate eFD filings only.",
    ),
    "hybrid_baseline": RunProfile(
        name="hybrid_baseline",
        data_subdir="hybrid_analysis",
        output_subdir="hybrid_analysis",
        log_subdir="hybrid_baseline",
        manifest_subdir="hybrid_baseline",
        release_prefix="hybb",
        frozen=True,
        description=(
            "Pre-remediation hybrid run. Archived and hashed; read-only. "
            "Its results are superseded by hybrid_remediated."
        ),
    ),
    "hybrid_remediated": RunProfile(
        name="hybrid_remediated",
        data_subdir="hybrid_remediated_analysis",
        output_subdir="hybrid_remediated_analysis",
        log_subdir="hybrid_remediated",
        manifest_subdir="hybrid_remediated",
        release_prefix="hybr",
        description="Hybrid sample rebuilt under the remediation protocol.",
    ),
    "robustness": RunProfile(
        name="robustness",
        data_subdir="robustness_analysis",
        output_subdir="robustness_analysis",
        log_subdir="robustness",
        manifest_subdir="robustness",
        release_prefix="rob",
        description="Pre-specified robustness families R1-R12 (freeze §6).",
    ),
}


class ProfileError(RuntimeError):
    """A profile is unknown, frozen, or resolves onto foreign territory."""


def get_profile(name: str) -> RunProfile:
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise ProfileError(f"unknown run profile {name!r}; declared profiles: {known}") from None


def resolve_profile_name(
    explicit: str | None = None,
    config: dict | None = None,
) -> str:
    """Resolve which profile a run belongs to.

    Precedence: explicit argument > config ``run.profile`` > ``CT_RUN_PROFILE``.
    A config/environment *disagreement* is an error rather than a silent
    choice, because picking one of them is how PATH-001 stayed invisible.
    """
    if explicit:
        return get_profile(explicit).name

    cfg_name = None
    if config:
        run_cfg = config.get("run") or {}
        if isinstance(run_cfg, dict):
            cfg_name = run_cfg.get("profile")
    env_name = os.environ.get(_ENV_PROFILE) or None

    if cfg_name and env_name and cfg_name != env_name:
        raise ProfileError(
            f"run profile conflict: config says {cfg_name!r} but {_ENV_PROFILE} "
            f"says {env_name!r}. Unset one; they may not disagree."
        )
    return get_profile(cfg_name or env_name or DEFAULT_PROFILE).name


# ── Isolation checking ───────────────────────────────────────────────────────

def _is_within(inner: Path, outer: Path) -> bool:
    """True when ``inner`` is ``outer`` or sits underneath it."""
    try:
        inner.relative_to(outer)
        return True
    except ValueError:
        return False


def profile_write_dirs(profile: RunProfile, project_root: Path) -> dict[str, Path]:
    """The directories a run under ``profile`` writes, by role.

    Isolation is asserted on these, not on ``data_root``/``output_root``: the
    per-profile data roots all nest under one shared ``data/`` by design, and
    comparing namespace roots would flag that legitimate nesting while missing
    the collisions that matter.
    """
    data_root = project_root / "data"
    if profile.data_subdir:
        data_root = data_root / profile.data_subdir
    output_root = project_root / "outputs"
    if profile.output_subdir:
        output_root = output_root / profile.output_subdir

    return {
        "interim": data_root / "interim",
        "processed": data_root / "processed",
        "analysis": data_root / "analysis",
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "latex": output_root / "latex",
        "reports": output_root / "reports",
        "logs": project_root / "logs" / profile.log_subdir,
        "manifests": project_root / "manifests" / profile.manifest_subdir,
    }


def shared_input_roots(project_root: Path) -> list[Path]:
    return [project_root / "data" / sub for sub in SHARED_INPUT_SUBDIRS]


def assert_registry_isolated(project_root: Path) -> None:
    """Every declared profile's write directories are pairwise disjoint.

    A registry-level invariant: it catches a bad profile declaration at import
    time rather than after a run has already written into another track.
    """
    dirs: dict[str, dict[str, Path]] = {
        name: profile_write_dirs(p, project_root) for name, p in PROFILES.items()
    }
    names = sorted(dirs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for role_a, path_a in dirs[a].items():
                for role_b, path_b in dirs[b].items():
                    if _is_within(path_a, path_b) or _is_within(path_b, path_a):
                        raise ProfileError(
                            f"profile isolation violated: {a}.{role_a} ({path_a}) and "
                            f"{b}.{role_b} ({path_b}) are not disjoint"
                        )
    for name, roles in dirs.items():
        for shared in shared_input_roots(project_root):
            for role, path in roles.items():
                if _is_within(path, shared) or _is_within(shared, path):
                    raise ProfileError(
                        f"profile {name}.{role} ({path}) collides with the shared "
                        f"input root {shared}, which every profile reads"
                    )


def assert_resolved_isolated(
    profile: RunProfile,
    project_root: Path,
    resolved: dict[str, Path],
) -> None:
    """The *resolved* write dirs stay inside this profile's own territory.

    ``resolved`` is what path resolution actually produced after environment
    overrides. This is the check that fires when a hybrid run is pointed at the
    legacy roots (finding PATH-001) or at another profile's root (freeze §8).
    """
    for other_name, other in PROFILES.items():
        if other_name == profile.name:
            continue
        foreign = profile_write_dirs(other, project_root)
        for role, path in resolved.items():
            for other_role, other_path in foreign.items():
                if _is_within(path, other_path) or _is_within(other_path, path):
                    raise ProfileError(
                        f"run refused: profile {profile.name!r} resolved {role} to "
                        f"{path}, which is inside profile {other_name!r}'s "
                        f"{other_role} root ({other_path}). Check "
                        f"DROPBOX_DATA_ROOT / DROPBOX_OUTPUT_ROOT / "
                        f"DROPBOX_LOG_ROOT / DROPBOX_MANIFEST_ROOT."
                    )
    for shared in shared_input_roots(project_root):
        for role, path in resolved.items():
            if _is_within(path, shared) or _is_within(shared, path):
                raise ProfileError(
                    f"run refused: profile {profile.name!r} resolved {role} to {path}, "
                    f"inside the shared read-only input root {shared}"
                )


def assert_writable(profile: RunProfile) -> None:
    if profile.frozen:
        raise ProfileError(
            f"profile {profile.name!r} is frozen: {profile.description} "
            f"Resolve it with read_only=True to read its artifacts, or choose a "
            f"writable profile."
        )
