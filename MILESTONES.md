# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-31 16:42] - core_claim_plus
- Reproduced the paper's synthetic College Admission core setting with the actual CGPR adaptive-ensemble procedure: aligned and complementary logistic specialists, oracle adaptive routing, weighted specialist variant, and CGPR team scoring.
- Prior compute-container validation passed with `python validate.py --compare`.
- Expanded beyond the minimum by reproducing a second human-accuracy setting (`alpha=0.7`) from the authors' released synthetic sweep.
- Inspected the official code release at `https://github.com/shasanamin/aaai26-adaptive-ai` and confirmed the local College Admission implementation mirrors the released synthetic generator and specialist-training procedure.

## Stop Justification
- Completed at milestone `core_claim_plus`. Higher milestones would require the WoofNette image benchmark or broader synthetic sweeps; the current package is already a reusable benchmark for the paper's core synthetic claim and one additional setting.
