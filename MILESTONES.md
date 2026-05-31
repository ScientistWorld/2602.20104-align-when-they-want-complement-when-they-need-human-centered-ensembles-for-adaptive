# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-31 16:45] - core_claim_plus
- Audited the reset workspace and found the existing implementation intact: method code contains the paper's synthetic CGPR data generation, single-model surrogate, aligned/complementary specialists, adaptive routing, and RRS artifacts.
- Verified that `eval/` remains independent from `method/` and that `python validate.py --compare` passes on the existing reproduced scores.
- Added three more alpha-sweep settings from the authors' released synthetic results to prepare a secondary-claim reproduction job.

### [2026-05-31 16:42] - core_claim_plus
- Reproduced the paper's synthetic College Admission core setting with the actual CGPR adaptive-ensemble procedure: aligned and complementary logistic specialists, oracle adaptive routing, weighted specialist variant, and CGPR team scoring.
- Prior compute-container validation passed with `python validate.py --compare`.
- Expanded beyond the minimum by reproducing a second human-accuracy setting (`alpha=0.7`) from the authors' released synthetic sweep.
- Inspected the official code release at `https://github.com/shasanamin/aaai26-adaptive-ai` and confirmed the local College Admission implementation mirrors the released synthetic generator and specialist-training procedure.

## Stop Justification
- Not stopping yet. A CPU job is being submitted to run a broader synthetic
  human-reliability sweep and test whether the reproduction can advance beyond
  `core_claim_plus`.
