"""Canonical two-clock pipeline: 30 declared stages behind one gated entry
point (standing directive section I). The outcome-unlock step is disabled
until every human gate is signed and the single-use token is issued by the
authorized human ceremony."""

from trading_on_disclosure_v2.pipeline.canonical import (  # noqa: F401
    STAGES,
    PipelineContext,
    run_pipeline,
)
