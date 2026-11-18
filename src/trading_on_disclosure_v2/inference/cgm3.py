"""Three-way Cameron-Gelbach-Miller cluster covariance (INF-001..005).

Full inclusion-exclusion over the three headline dimensions
(``filer_uid``, ``permno``, ``follower_entry_date``): the seven sandwich
terms are the three one-way covariances, minus the three pairwise
intersections, plus the triple intersection. There is no IID path (INF-003)
and month/legacy-identifier cluster fields are rejected (INF-002).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

FORBIDDEN_CLUSTER_FIELDS = frozenset(
    {"calendar_month", "disclosure_month", "month", "legislator_id"}
)

HEADLINE_CLUSTERS = ("filer_uid", "permno", "follower_entry_date")


class ClusterFieldError(ValueError):
    pass


@dataclass
class CGM3Result:
    beta: np.ndarray
    cov: np.ndarray
    cov_uncorrected: np.ndarray
    cov_raw_before_psd_fix: np.ndarray
    psd_corrected: bool
    min_eigenvalue_raw: float
    cluster_counts: dict[str, int] = field(default_factory=dict)
    correction: str = "cgm_per_term"
    dof: int = 0

    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))


def _meat(X: np.ndarray, u: np.ndarray, codes: np.ndarray) -> np.ndarray:
    k = X.shape[1]
    Xu = X * u[:, None]
    n_groups = codes.max() + 1
    sums = np.zeros((n_groups, k))
    np.add.at(sums, codes, Xu)
    return sums.T @ sums


def _group_codes(frame: pd.DataFrame, cols: tuple[str, ...]) -> tuple[np.ndarray, int]:
    if len(cols) == 1:
        codes, uniques = pd.factorize(frame[cols[0]], sort=False)
    else:
        codes, uniques = pd.factorize(
            pd.MultiIndex.from_frame(frame[list(cols)]), sort=False
        )
    return codes.astype(np.int64), len(uniques)


def cgm_cluster_cov(
    y: np.ndarray,
    X: np.ndarray,
    clusters: pd.DataFrame,
    small_sample: bool = True,
    psd_fix: str = "eigenvalue_truncation",
) -> CGM3Result:
    """OLS beta and CGM inclusion-exclusion covariance over 1-3 dimensions."""
    for col in clusters.columns:
        if col.lower() in FORBIDDEN_CLUSTER_FIELDS:
            raise ClusterFieldError(
                f"forbidden cluster field {col!r}: month and legacy legislator "
                "identifiers may not be headline cluster dimensions (INF-002)"
            )
    dims = tuple(clusters.columns)
    if not 1 <= len(dims) <= 3:
        raise ClusterFieldError("cgm_cluster_cov supports 1-3 cluster dimensions")
    if len(set(dims)) != len(dims):
        raise ClusterFieldError("cluster dimensions must be distinct fields")

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if len(clusters) != n:
        raise ValueError("clusters frame must align with X rows")
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta

    cov_corr = np.zeros((k, k))
    cov_raw = np.zeros((k, k))
    counts: dict[str, int] = {}
    min_g = None
    for r in range(1, len(dims) + 1):
        sign = (-1.0) ** (r + 1)
        for subset in combinations(dims, r):
            codes, n_groups = _group_codes(clusters, subset)
            counts["×".join(subset)] = n_groups
            meat = _meat(X, u, codes)
            term = XtX_inv @ meat @ XtX_inv
            cov_raw += sign * term
            if small_sample and n_groups > 1:
                c = (n_groups / (n_groups - 1)) * ((n - 1) / (n - k))
            else:
                c = 1.0
            cov_corr += sign * c * term
            if r == 1:
                min_g = n_groups if min_g is None else min(min_g, n_groups)

    chosen = cov_corr if small_sample else cov_raw
    eigvals = np.linalg.eigvalsh(chosen)
    min_eig = float(eigvals.min())
    psd_corrected = False
    fixed = chosen
    if min_eig < 0:
        if psd_fix != "eigenvalue_truncation":
            raise ValueError(f"unknown psd_fix rule: {psd_fix!r}")
        w, v = np.linalg.eigh(chosen)
        fixed = (v * np.clip(w, 0.0, None)) @ v.T
        psd_corrected = True

    return CGM3Result(
        beta=beta,
        cov=fixed,
        cov_uncorrected=cov_raw,
        cov_raw_before_psd_fix=chosen,
        psd_corrected=psd_corrected,
        min_eigenvalue_raw=min_eig,
        cluster_counts=counts,
        correction="cgm_per_term" if small_sample else "none",
        dof=(min_g or 1) - 1,
    )


#: primary covariance of the locked streamlined design (final directive §12):
#: two-way CGM by canonical household and EXACT public-entry trading day.
PRIMARY_TWO_WAY_CLUSTERS = ("filer_uid", "follower_entry_date")


def two_way_cluster_cov(
    y: np.ndarray,
    X: np.ndarray,
    clusters: pd.DataFrame,
    required: tuple[str, str] = PRIMARY_TWO_WAY_CLUSTERS,
    **kwargs,
) -> CGM3Result:
    """Primary headline covariance: exactly household x exact entry date."""
    if tuple(clusters.columns) != tuple(required):
        raise ClusterFieldError(
            f"primary covariance requires cluster columns {required}, "
            f"got {tuple(clusters.columns)}"
        )
    return cgm_cluster_cov(y, X, clusters, **kwargs)


def three_way_cluster_cov(
    y: np.ndarray,
    X: np.ndarray,
    clusters: pd.DataFrame,
    required: tuple[str, str, str] = HEADLINE_CLUSTERS,
    **kwargs,
) -> CGM3Result:
    """Headline covariance: requires exactly the three contract dimensions."""
    if tuple(clusters.columns) != tuple(required):
        raise ClusterFieldError(
            f"headline covariance requires cluster columns {required}, "
            f"got {tuple(clusters.columns)}"
        )
    return cgm_cluster_cov(y, X, clusters, **kwargs)
