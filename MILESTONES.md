# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-05-31 16:59] - secondary_claims
- Job `d4985458-aae` completed successfully and `python validate.py --compare` passed for the five-setting imperfect-human alpha sweep.
- The corrected simulator recovered the low-alpha qualitative pattern: at alpha=0.5, adaptive oracle team accuracy was 0.609 versus behavior-aware 0.579 and complementary 0.583.
- The package now reproduces the core synthetic claim and the secondary human-reliability sweep under the authors' released College Admission protocol.

### [2026-05-31 16:57] - core_claim_plus
- Job `179f7023-580` completed and passed validation, but alpha=0.5 still failed to reproduce the paper's adaptive advantage.
- Identified a remaining human-simulation mismatch: vectorized random draws changed the paper's per-example RNG order for correctness and confidence.
- Updated the simulator to follow the authors' released per-example draw sequence and resubmitting before claiming a higher milestone.

### [2026-05-31 16:55] - core_claim_plus
- Job `772f9aa7-605` failed before running the method because the new paper-protocol sweep passed an RNG into a generator function whose signature still only accepted a seed.
- Fixed `data/college_admission.py` so the generator supports both the old seed-based path and the new shared-RNG alpha-sweep path.
- Structural validation still passes; resubmitting the corrected CPU sweep.

### [2026-05-31 16:53] - core_claim_plus
- Job `59e5cbe0-420` completed successfully and produced scores for all five requested alpha-sweep settings.
- The run exposed a low-alpha protocol mismatch: independently regenerated alpha settings did not reproduce the paper's shared-RNG imperfect-human sweep at alpha=0.5.
- Updated data generation, evaluation, and reproduction scripts to match the authors' released sweep protocol before attempting to claim `secondary_claims`.

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
- Completed at milestone `secondary_claims`.
