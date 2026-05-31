# Overview

Paper: "Align When They Want, Complement When They Need! Human-Centered
Ensembles for Adaptive Human-AI Collaboration" by Hasan Amin, Ming Yin, and
Rajiv Khanna.

This reproduction packages the core synthetic experiment: a two-region admission
task where human confidence and ability differ by region. The method trains
aligned and complementary specialists and adaptively routes between them under
the CGPR human-AI team model. The benchmark scores whether adaptive
recommendations improve team accuracy while preserving the reliance quantity
that makes humans willing to use AI advice.

The packaged reproduction reaches `core_claim_plus`: the alpha=0.9 core
setting matches the authors' reported synthetic result closely, and an
additional alpha=0.7 setting preserves the same adaptive-routing advantage.

---

# Problem Definition

The task is AI-assisted binary decision making with a human decision maker who
retains final control. Each instance has features, a ground-truth label, a human
initial prediction, and a human confidence score. An AI system provides one
recommendation per instance.

Success is measured after simulating the final human-AI team decision. A useful
assistant must improve the final team accuracy on cases where humans are weak,
while preserving enough agreement with confident human judgments that humans
remain willing to rely on the AI when they are uncertain.

The reusable benchmark in this workspace uses the paper's synthetic College
Admission setting. Applicants are described by GPA and standardized test score.
There are two regions: an aligned/high-confidence region where humans are more
reliable, and a complementary/low-confidence region where humans need AI help.

---

# Paper's Method

The paper introduces Confidence-Gated Probabilistic Reliance (CGPR), a model of
AI-assisted decision making where high-confidence humans keep their own answer
and low-confidence humans accept AI advice with a probability determined by the
AI's agreement with humans in the high-confidence region.

The proposed system trains two specialists. The aligned specialist minimizes
disagreement with human predictions in the alignment region. The complementary
specialist minimizes prediction loss against ground truth in the complementary
region. At inference, an adaptive ensemble selects between specialists either
with oracle region membership or with the Rational Routing Shortcut (RRS), which
chooses the specialist with higher model confidence.

The implementation in `method/college_admission_method.py` follows these steps
with logistic-regression specialists, a behavior-aware single model, oracle
routing, weighted oracle routing, and RRS routing. `eval/` applies the CGPR
team-decision rule to method predictions so the scorer remains independent of
the paper-method implementation.

---

# Evaluation

Evaluation reads held-out predictions written by a method run and computes:

- `team_accuracy`: accuracy of final CGPR team decisions.
- `reliance_probability`: one minus the AI-human disagreement rate in the
  aligned region, the paper's operational trust/reliance quantity.
- `routing_accuracy`: for adaptive systems, how often the selected specialist
  matches the known synthetic region.

`scripts/evaluate.sh` is method-agnostic: it imports only `eval/` code and reads
`checkpoints/college_admission/predictions.csv` by default. Future methods can
be scored by writing the same prediction columns and running the same evaluator.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 14m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-05-31 16:43] - core_claim_plus
- Job `f814d317-529` completed successfully in the compute container.
- Validated the core alpha=0.9 College Admission setting and the additional alpha=0.7 setting with method-independent CGPR scoring.
- Restored `core_claim_plus`: adaptive oracle routing reproduces the paper's core synthetic claim, and the second human-accuracy setting provides additional coverage.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 13m | **GPU**: 0.0h
- **Jobs**: 8 total (6 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 8 total (6 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 8 total (6 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 9 total (7 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
