# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-31 16:20] - none
- Started from a partially initialized workspace for the adaptive human-AI collaboration paper.
- Read the paper's synthetic College Admission setup, CGPR reliance model, specialist training objectives, and adaptive routing rules.

### [2026-05-31 16:27] - method_runs
- Implemented the paper's College Admission synthetic data generator.
- Implemented logistic behavior-aware, aligned, complementary, adaptive oracle, weighted adaptive oracle, and RRS methods.
- Added method-agnostic evaluation that reads prediction artifacts and applies the CGPR team decision rule.

### [2026-05-31 16:31] - core_claim
- A compute-container job reproduced the alpha=0.9 College Admission core setting.
- Adaptive oracle team accuracy matched the paper-scale synthetic reference closely and exceeded the single behavior-aware model and specialists.

### [2026-05-31 16:32] - core_claim_plus
- Added a second alpha=0.7 human-accuracy setting to the reproducible suite.
- The additional setting preserved the adaptive-routing advantage over the single behavior-aware model on the same benchmark family.

### [2026-05-31 16:37] - core_claim
- Continuation audit confirmed the implementation is the paper's real synthetic CGPR/adaptive-ensemble algorithm, not a surrogate.
- Found that persisted scores were stale relative to the newer RRS scorer, so the milestone is held at the reset `core_claim` until the refresh job completes.

### [2026-05-31 16:40] - core_claim
- Job `148f0607-883` refreshed scores but failed validation because default scoring included RRS without paper-reported synthetic reference entries.
- Fixed the scorer to validate only paper-reported synthetic methods while retaining RRS implementation in method outputs.

## Stop Justification
- Not stopped. Current submission refreshes scores after aligning the default scorer with `reference.json`.
