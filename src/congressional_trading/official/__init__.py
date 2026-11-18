"""Official-source congressional financial-disclosure pipeline.

Canonical sources are the U.S. House Clerk financial-disclosure system and the
U.S. Senate eFD public search. Third-party datasets are used for comparison
only.

Layers: discovery -> fetch (immutable raw archive + manifest) -> format
identification -> parse -> normalize -> member resolution -> amendment
reconciliation -> project-compatible export -> validation -> release.
"""

from __future__ import annotations

PIPELINE_VERSION = "0.1.0"
NORMALIZATION_VERSION = "1"
SCHEMA_VERSION = "official-1.0"
