"""Records every Cameron-Gelbach-Miller covariance the original code computes (pre- and post-truncation), by
wrapping trading_on_disclosure_v2.inference.cgm3.cgm_cluster_cov at each of its import sites."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np

import trading_on_disclosure_v2.inference.cgm3 as cgm3
from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.ext import inference as ext_inference
from trading_on_disclosure_v4.inference import cgm as v4cgm

SITES = (cgm3, v4cgm, ext_inference, rerun)
ORIGINAL = cgm3.cgm_cluster_cov


@dataclass
class Recorder:
    records: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def __call__(self, y, X, clusters, small_sample=True, psd_fix="eigenvalue_truncation"):
        res = ORIGINAL(y, X, clusters, small_sample=small_sample, psd_fix=psd_fix)
        pre = np.asarray(res.cov_raw_before_psd_fix, float)
        post = np.asarray(res.cov, float)
        eig = np.linalg.eigvalsh(pre)
        scale = float(np.max(np.abs(eig))) if len(eig) else 0.0
        self.records.append({**self.context, "record_id": len(self.records), "n": int(len(y)), "k": int(X.shape[1]),
                             "dims": "x".join(clusters.columns), "cluster_counts": dict(res.cluster_counts), "dof": int(res.dof),
                             "beta": np.asarray(res.beta, float).tolist(), "cov_pre": pre.tolist(), "cov_post": post.tolist(),
                             "eig_pre": eig.tolist(), "min_eig_pre": float(eig.min()), "max_abs_eig_pre": scale,
                             "psd_corrected": bool(res.psd_corrected), "se_pre": np.sqrt(np.clip(np.diag(pre), 0, None)).tolist(),
                             "se_post": np.sqrt(np.diag(post)).tolist(), "diag_pre": np.diag(pre).tolist()})
        return res

    @contextlib.contextmanager
    def label(self, **ctx):
        old = self.context
        self.context = {**old, **ctx}
        try:
            yield
        finally:
            self.context = old


@contextlib.contextmanager
def recording():
    rec = Recorder()
    saved = [(m, m.cgm_cluster_cov) for m in SITES]
    try:
        for m in SITES:
            m.cgm_cluster_cov = rec
        yield rec
    finally:
        for m, f in saved:
            m.cgm_cluster_cov = f
