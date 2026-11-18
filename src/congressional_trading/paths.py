"""Central path resolution.

All data lives in Dropbox; the repository holds only code. Paths are resolved
from environment variables (optionally loaded from a local ``.env``), falling
back to ``configs/paths.yaml`` if present. Nothing here ever creates files
inside the Git tree.

Every resolution belongs to a **run profile** (see :mod:`.run_profiles`), which
owns the data, output, log and manifest roots it may write. Resolution fails
when a profile lands on another profile's roots.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .run_profiles import (
    RunProfile,
    assert_registry_isolated,
    assert_resolved_isolated,
    assert_writable,
    get_profile,
    profile_write_dirs,
    resolve_profile_name,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_LOADED = False


def _load_dotenv_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:
        pass
    _ENV_LOADED = True


def _expand(value: str) -> str:
    # Support ${VAR} references inside .env values.
    return os.path.expandvars(os.path.expanduser(value))


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem layout for one run.

    ``profile`` names the run track that owns the writable roots. ``raw_root``
    is deliberately *not* derived from ``data_root``: raw market data and the
    official collection store are shared read-only inputs, so every profile
    reads the same copy instead of needing a symlink or an absolute-path
    override per profile.
    """

    project_root: Path
    data_root: Path
    output_root: Path
    log_root: Path
    manifest_root: Path
    raw_root: Path | None = None
    profile: str = "ct_only"
    #: Optional sub-namespace for the four OUTPUT directories only. A sensitivity
    #: run (e.g. the 20-day-purge comparison) belongs to the same run track as
    #: the primary — same sample, same profile — but writes fixed filenames that
    #: would otherwise overwrite the primary's tables with the results it is
    #: being compared against. Declared as ``run.output_namespace``. Data roots
    #: are deliberately unaffected: the sensitivity reads the frozen inputs.
    output_namespace: str | None = None

    # Data sub-roots
    raw: Path = field(init=False)
    interim: Path = field(init=False)
    processed: Path = field(init=False)
    analysis: Path = field(init=False)

    def __post_init__(self) -> None:
        raw = self.raw_root if self.raw_root is not None else self.data_root / "raw"
        object.__setattr__(self, "raw_root", raw)
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "interim", self.data_root / "interim")
        object.__setattr__(self, "processed", self.data_root / "processed")
        object.__setattr__(self, "analysis", self.data_root / "analysis")

    @property
    def official_store(self) -> Path:
        """Shared official-collection root (state DB, raw store, release)."""
        return self.project_root / "data" / "official"

    @property
    def raw_capitol_trades(self) -> Path:
        return self.raw / "capitol_trades"

    @property
    def raw_crsp(self) -> Path:
        return self.raw / "crsp"

    @property
    def raw_factors(self) -> Path:
        return self.raw / "factors"

    @property
    def _out(self) -> Path:
        if self.output_namespace:
            return self.output_root / self.output_namespace
        return self.output_root

    @property
    def tables(self) -> Path:
        return self._out / "tables"

    @property
    def figures(self) -> Path:
        return self._out / "figures"

    @property
    def latex(self) -> Path:
        return self._out / "latex"

    @property
    def run_reports(self) -> Path:
        return self._out / "reports"

    @property
    def output_data(self) -> Path:
        """Row-level output artifacts (OOF/holdout predictions, ledgers).

        Namespaced like the other output roots. Stages previously built this as
        ``output_root / "data"``, which bypassed the namespace and let a
        sensitivity run overwrite the primary run's predictions.
        """
        return self._out / "data"

    def ensure_output_dirs(self) -> None:
        for p in (
            self.interim,
            self.processed,
            self.analysis,
            self.tables,
            self.figures,
            self.latex,
            self.run_reports,
            self.log_root,
            self.manifest_root,
        ):
            p.mkdir(parents=True, exist_ok=True)


def get_paths(
    profile: str | None = None,
    config: dict | None = None,
    read_only: bool = False,
) -> ProjectPaths:
    """Resolve paths for one run profile from environment (`.env` supported).

    Required: ``DROPBOX_PROJECT_ROOT``. The four writable roots default to the
    profile's declared subfolders; environment overrides are honoured but are
    rejected when they land inside another profile's roots.

    Parameters
    ----------
    profile : explicit profile name; otherwise resolved from ``config``'s
        ``run.profile`` or ``CT_RUN_PROFILE`` (freeze §8).
    config : run config, consulted for ``run.profile``.
    read_only : permit resolving a frozen (archived) profile in order to read
        its artifacts. A frozen profile may never be resolved for writing.
    """
    _load_dotenv_once()
    root_env = os.environ.get("DROPBOX_PROJECT_ROOT")
    if not root_env:
        raise RuntimeError(
            "DROPBOX_PROJECT_ROOT is not set. Copy .env.example to .env and "
            "point it at your Dropbox project folder."
        )
    root = Path(_expand(root_env))

    prof: RunProfile = get_profile(resolve_profile_name(profile, config))
    if not read_only:
        assert_writable(prof)
    assert_registry_isolated(root)

    def _sub(env_key: str, default: Path) -> Path:
        val = os.environ.get(env_key)
        return Path(_expand(val)) if val else default

    data_default = root / "data"
    if prof.data_subdir:
        data_default = data_default / prof.data_subdir
    output_default = root / "outputs"
    if prof.output_subdir:
        output_default = output_default / prof.output_subdir

    resolved = ProjectPaths(
        project_root=root,
        data_root=_sub("DROPBOX_DATA_ROOT", data_default),
        output_root=_sub("DROPBOX_OUTPUT_ROOT", output_default),
        log_root=_sub("DROPBOX_LOG_ROOT", root / "logs" / prof.log_subdir),
        manifest_root=_sub("DROPBOX_MANIFEST_ROOT", root / "manifests" / prof.manifest_subdir),
        raw_root=_sub("DROPBOX_RAW_ROOT", root / "data" / "raw"),
        profile=prof.name,
        output_namespace=_output_namespace(config),
    )
    assert_resolved_isolated(prof, root, {
        "interim": resolved.interim,
        "processed": resolved.processed,
        "analysis": resolved.analysis,
        "tables": resolved.tables,
        "figures": resolved.figures,
        "latex": resolved.latex,
        "reports": resolved.run_reports,
        "logs": resolved.log_root,
        "manifests": resolved.manifest_root,
    })
    return resolved


_NAMESPACE_OK = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


def _output_namespace(config: dict | None) -> str | None:
    """``run.output_namespace``, validated.

    PATH-001 was a config key that path resolution never read. This one IS read
    here, in central resolution, and a malformed value is refused rather than
    quietly ignored — an unread namespace would send a sensitivity's tables on
    top of the primary's.
    """
    # An operator may name the namespace in the environment. A pinned release
    # runs from an immutable checkout, and rewriting a tracked config inside
    # it to set the namespace made the tree dirty and changed its config hash
    # -- so the release failed its own provenance checks. The environment
    # carries the run-specific value; the tree stays exactly as committed.
    env_ns = os.environ.get("CT_OUTPUT_NAMESPACE")
    run_cfg = (config or {}).get("run") or {}
    if not isinstance(run_cfg, dict):
        run_cfg = {}
    ns = env_ns if env_ns not in (None, "") else run_cfg.get("output_namespace")
    if ns in (None, ""):
        return None
    ns = str(ns)
    if not set(ns) <= _NAMESPACE_OK:
        raise ValueError(
            f"run.output_namespace {ns!r} may use only [a-z0-9_-]; it names a "
            f"single output subdirectory, not a path")
    return ns


def get_input_paths(config: dict | None = None, run_profile: str | None = None) -> ProjectPaths:
    """Read-only paths for the profile that supplies a run's *inputs*.

    Writes are isolated per profile, reads are not: the robustness families
    read the remediated hybrid analysis sample and write to their own roots.
    Declared as ``run.input_profile``; defaults to the run's own profile.
    """
    name = None
    if config:
        run_cfg = config.get("run") or {}
        if isinstance(run_cfg, dict):
            name = run_cfg.get("input_profile")
    return get_paths(profile=name or run_profile, config=None, read_only=True)


def profile_roots(profile: str, project_root: Path | None = None) -> dict[str, Path]:
    """Declared write directories for ``profile`` (for manifests and diffs)."""
    if project_root is None:
        _load_dotenv_once()
        root_env = os.environ.get("DROPBOX_PROJECT_ROOT")
        if not root_env:
            raise RuntimeError("DROPBOX_PROJECT_ROOT is not set.")
        project_root = Path(_expand(root_env))
    return profile_write_dirs(get_profile(profile), project_root)
