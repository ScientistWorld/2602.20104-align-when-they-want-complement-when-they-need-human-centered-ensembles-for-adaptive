# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-31 16:30] - none
- Read the paper and identified the core CGPR adaptive ensemble method.
- Cloned the authors' public repository for reference.
- Implemented a clean College Admission reproduction pipeline with data generation, method outputs, and method-agnostic evaluation.
- Prepared a CPU test job to verify end-to-end execution inside the container.

### [2026-05-31 16:31] - none
- Audited the previous implementation and found evaluation-boundary and reference-table issues.
- Moved simulated human behavior into the tracked data generator and made evaluation recompute CGPR team outcomes from held-out data.
- Replaced mixed-benchmark reference values with the authors' released College Admission synthetic values.

### [2026-05-31 16:35] - core_claim
- CPU job `9cf433ae-84c` completed successfully in the container.
- Reproduced the core College Admission setting: adaptive oracle team accuracy 0.8063 versus paper 0.806, and weighted oracle team accuracy 0.7967 versus paper 0.7947.
- Validation passed with clean method/evaluation separation and plausible baseline scores.

### [2026-05-31 16:40] - core_claim_plus
- CPU job `29fb451a-dae` completed successfully in the container.
- Added and reproduced a second College Admission human-accuracy setting at alpha=0.7.
- The added setting preserved the paper's qualitative claim: adaptive oracle routing outperformed the single behavior-aware model and individual specialists while maintaining the reliance/routing constraints.

## Stop Justification

Not stopped.
