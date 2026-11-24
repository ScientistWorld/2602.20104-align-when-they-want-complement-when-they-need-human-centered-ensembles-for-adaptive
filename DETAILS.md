# Overview

Paper: "Align When They Want, Complement When They Need! Human-Centered
Ensembles for Adaptive Human-AI Collaboration" by Hasan Amin, Ming Yin, and
Rajiv Khanna.

The paper's claim is that, under a realistic model of when humans take AI
advice, an AI that agrees with humans where they are confident and corrects
them where they are not beats any single AI model on human-AI team accuracy.
This holds both when the human's region is known and when the AI must infer
it.

The gym scores this on an admissions world that extends the paper's
synthetic College Admission data:
- two applicant groups with different outcome rules;
- context features that reveal the group only partly;
- evaluators who are confident and sometimes confidently biased in one
  group, and unsure in the other;
- confidence thresholds drawn per decision.

The paper's adaptive ensemble is the reference. It beats the paper's
single-model baselines in all four scored experiments: group known or hidden,
crossed with reliable or less reliable evaluators.

The paper's own College Admission sweep is also reproduced, unscored and
exactly matching the released numbers. Milestone: `secondary_claims` (the
WoofNette image benchmark is not reproduced).

---

# Problem Definition

AI-assisted binary decision making in which a human decision maker keeps the
final say. For every case the AI gives one recommendation (admit = 1,
reject = 0). The human then decides with the following confidence-gated,
probabilistic reliance behaviour:

- Each decision has its own confidence threshold tau, drawn from U[0.6, 0.8].
- If the human's self-confidence on the case exceeds tau, the human keeps
  their own judgement. These are the cases where the human feels sure.
- Otherwise the human accepts the AI's recommendation with probability r,
  and keeps their own judgement otherwise.
- r is trust. It equals the rate at which the AI's recommendations agree
  with the human's own judgement on the cases where the human felt sure.

So an assistant is rewarded in two ways:
- for being right on cases where humans are unsure;
- for agreeing with humans where they feel sure, because that agreement is
  what makes them take its advice at all.

## Data

The setting is college admission. Applicants are described by `gpa`, `score`
(a standardised test score) and six context features: `school_resources`,
`family_income`, `neighborhood_index`, `prep_hours`, `test_retakes` and
`activities`. Applicants come from two groups. The group affects both what
predicts the outcome and how sure and accurate the human evaluators are.

Each experiment directory `data/<experiment>/` holds:

| file | rows | columns |
|---|---|---|
| `train.csv` | 7,000 | `example_id`, features, [`group`], `label`, `human_pred`, `human_conf` |
| `dev.csv` | 3,000 | same as train (a visible development split) |
| `test_inputs.csv` | 20,000 | `example_id`, features, [`group`] (no answers) |
| `task.json` | | feature list, whether `group` is given, the threshold distribution |

- `human_pred` is the evaluator's own initial judgement.
- `human_conf` is the evaluator's self-reported confidence.
- Both are available for training (think of them as pilot-study data). They
  are never available at test time.
- `group` is given (in all three files) only in the `admission_group_known*`
  experiments. In the `admission_group_hidden*` experiments the AI sees
  applicant features only.

`data/experiments.json` lists the scored experiments. The other
`data/college_admission*` directories are not scored. They hold the paper's
original two-feature generator and are used only by an unscored
reproduction check.

## What to produce

For every experiment, write one binary recommendation per `example_id` of
`dev.csv` and `test_inputs.csv` to:

    checkpoints/<method_name>/<experiment>/predictions.csv   # columns: example_id,prediction

- The ranked entry is `checkpoints/submission/`.
- Any other method directory is scored too, under its own name, for
  comparison.
- Any recommendation strategy is allowed, but it may use only the inputs
  available at test time.

---

# Paper's Method

Paper: "Align When They Want, Complement When They Need! Human-Centered
Ensembles for Adaptive Human-AI Collaboration" (Amin, Yin, Khanna, AAAI 2026).

## Team model

Confidence-Gated Probabilistic Reliance (CGPR, Eq. 2):
- a human whose confidence C^h(x) exceeds the threshold tau keeps their own
  judgement;
- otherwise they follow the AI with probability r = 1 − L_h(D_a, m), the
  AI's agreement with the human on the alignment region D_a = {C^h > tau}.

With tau drawn per decision from a distribution F_T, region membership is
probabilistic (Proposition 5). The paper's instance weights are
w_a = F_T(C^h) and w_c = 1 − F_T(C^h).

## Adaptive ensemble

- **Aligned specialist m_a.** Minimises disagreement with the human's own
  predictions, with the alignment-region weights.
- **Complementary specialist m_c.** Minimises prediction loss on the ground
  truth, with the complementarity-region weights.
- **Routing.**
  - Oracle routing (Eq. 7) sends a case to m_a when it is in the alignment
    region and to m_c otherwise.
  - The Rational Routing Shortcut (RRS, Eq. 8) sends a case to the more
    confident specialist instead.

## Implementation in this gym (`method/adaptive_team_ai.py`)

This follows the paper's synthetic recipe: logistic regression for every AI
model, sklearn defaults, deterministic.

- **Group given (`admission_group_known*`).** This is the paper's synthetic
  default, where group membership is treated as known.
  - m_a is fitted on the privileged group's human judgements, with weights
    F_T(C^h).
  - m_c is fitted on the other group's outcomes, with weights 1 − F_T(C^h).
  - Routing uses the group.
- **Group hidden (`admission_group_hidden*`).** The paper's
  behaviour-grounded protocol.
  - m_a and m_c are fitted on all training rows with Proposition 5 weights.
  - The human's confidence is unobservable at test time. As in the paper's
    pipeline, a confidence model fitted on the training (pilot) data
    estimates P(C^h > tau | x), and Eq. 7 routes to m_a when that estimate
    is at least 0.5.
  - `--router rrs` uses the paper's RRS instead. On this world it scores
    clearly lower; see HARDENING.md.
- **Baselines.** The paper's single-model paradigms, also logistic
  regression:
  - standard AI (ERM on outcomes);
  - behaviour-aware single AI (the weighted surrogate g_p of Theorem 4, with
    human targets in the alignment region);
  - aligned AI alone;
  - complementary AI alone.

`method/college_admission_method.py` and `method/aaai26-adaptive-ai/` (the
authors' release) reproduce the paper's own two-feature College Admission
sweep. They feed the unscored `scoring/paper_reproduction.json`, which matches
the authors' released numbers exactly.

---

# Evaluation

All scores are recomputed by `eval/` from the binary recommendations alone.
Nothing a method writes besides `example_id,prediction` is read.

## Metrics (per experiment)

Let P_i = P(conf_i > tau) = clip((conf_i - 0.6) / 0.2, 0, 1). This is the
probability that the human feels sure on case i.

- `team_accuracy` (primary, coefficient 1.0) is the expected accuracy of the
  final human decision. It is computed exactly over the threshold draw and the
  reliance coin, so there is no simulation noise:
  `mean_i( P_i [h_i = y_i] + (1 - P_i) (r [m_i = y_i] + (1 - r) [h_i = y_i]) )`
- `reliance_probability` (constraint: trust, coefficient 0.5) is the r above:
  `sum_i P_i [m_i = h_i] / sum_i P_i`. It is the agreement with the human's own
  judgement where the human feels sure.
- Two diagnostics, not ranked (coefficient 0):
  - `ai_accuracy` is the standalone accuracy of the recommendations;
  - `complementary_region_accuracy` is the AI accuracy weighted by 1 - P_i.

Here m_i is the AI recommendation, h_i the human's judgement, y_i the outcome
and conf_i the human's confidence.

## Rules

- Every scored `example_id` must appear exactly once with a prediction of
  exactly 0 or 1. If a file is missing or unreadable, or has missing,
  duplicate, NaN, fractional or out-of-range predictions, that experiment
  scores its worst value (0) in every metric.
- Rows for other ids are ignored.
- The ranked slot `submission` is always scored. If it is absent it gets the
  worst values.
- The baselines (`standard_ai`, `behavior_aware_ai`, `aligned_ai`,
  `complementary_ai`) are scored from predictions frozen inside `eval/`.
  Method directories with those names are ignored.

## Slices and scripts

- `bash scripts/evaluate_train.sh` (and `bash scripts/evaluate.sh`, which
  writes `scoring/scores.json`) scores the visible dev split. The answers are
  in `data/<experiment>/dev.csv`. Output: `scoring/scores_train.json`.
- `bash scripts/evaluate_test.sh` scores the hidden test rows
  (`test_inputs.csv`), whose answers are not in the workspace. Output:
  `scoring/scores_test.json`.
  - It takes no arguments and scores every method directory under
    `checkpoints/`.
  - It deletes the old file before writing.
- Train and test measure the same thing on draws from the same population.
  Dev has 3,000 rows and test has 20,000 rows per experiment.

## Ranking

The rank sums, over experiments, `weight × Σ_metrics coefficient ×
(improvement over the reference)` as set in `scoring/reference.json`:

| experiment | weight |
|---|---|
| `admission_group_hidden` | 0.35 |
| `admission_group_hidden_low_reliability` | 0.20 |
| `admission_group_known` | 0.25 |
| `admission_group_known_low_reliability` | 0.20 |

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

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
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

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 9 total (7 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-31 17:45] - secondary_claims
- Final verification passed `python validate.py --compare`.
- Tightened method-agnostic briefing and scoring documentation so from-scratch scientists see the task, metrics, and constraints without implementation hints.
- Confirmed the remaining limitation is scope: the gym reproduces the synthetic College Admission core and human-reliability sweep, not the WoofNette image benchmark.

### [2026-05-31 17:25] - secondary_claims
- Continuation audit found the existing package still implements the paper's synthetic CGPR adaptive-ensemble method and has prior passing CPU-container validation.
- Submitting a fresh CPU end-to-end reproduction job to regenerate the full College Admission alpha sweep and re-run `python validate.py --compare` in the compute container.

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

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-05-31 17:50] - secondary_claims
- Final verification confirmed the claimed `secondary_claims` milestone is justified.
- `python validate.py --compare` passes with the five College Admission alpha-sweep experiments, train/test score files, and structural separation checks.
- Cleaned final packaging by ignoring/removing tracked job logs, correcting the overview milestone, and marking AI-only train/test diagnostics as zero-coefficient metrics so they do not distort the gym score.

### [2026-05-31 17:45] - secondary_claims
- Final verification passed `python validate.py --compare`.
- Tightened method-agnostic briefing and scoring documentation so from-scratch scientists see the task, metrics, and constraints without implementation hints.
- Confirmed the remaining limitation is scope: the gym reproduces the synthetic College Admission core and human-reliability sweep, not the WoofNette image benchmark.

### [2026-05-31 17:25] - secondary_claims
- Continuation audit found the existing package still implements the paper's synthetic CGPR adaptive-ensemble method and has prior passing CPU-container validation.
- Submitting a fresh CPU end-to-end reproduction job to regenerate the full College Admission alpha sweep and re-run `python validate.py --compare` in the compute container.

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

### Iteration 10: claude-opus-5-5 (redteam)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 14m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-31 17:50] - secondary_claims
- Final verification confirmed the claimed `secondary_claims` milestone is justified.
- `python validate.py --compare` passes with the five College Admission alpha-sweep experiments, train/test score files, and structural separation checks.
- Cleaned final packaging by ignoring/removing tracked job logs, correcting the overview milestone, and marking AI-only train/test diagnostics as zero-coefficient metrics so they do not distort the gym score.

### [2026-05-31 17:45] - secondary_claims
- Final verification passed `python validate.py --compare`.
- Tightened method-agnostic briefing and scoring documentation so from-scratch scientists see the task, metrics, and constraints without implementation hints.
- Confirmed the remaining limitation is scope: the gym reproduces the synthetic College Admission core and human-reliability sweep, not the WoofNette image benchmark.

### [2026-05-31 17:25] - secondary_claims
- Continuation audit found the existing package still implements the paper's synthetic CGPR adaptive-ensemble method and has prior passing CPU-container validation.
- Submitting a fresh CPU end-to-end reproduction job to regenerate the full College Admission alpha sweep and re-run `python validate.py --compare` in the compute container.

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

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `secondary_claims` | **Status**: error
- **Working time**: 43m | **GPU**: 0.0h
- **Jobs**: 6 total (5 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-11-24 01:40] - secondary_claims
- Hardening (iteration 11): the scored hidden test was rebuilt as an extended College Admission world.
  - Answers and the generator are only in `eval/test/`.
  - Four experiments: group known or hidden, crossed with reliable or less reliable evaluators.
  - Exact expected CGPR team accuracy.
  - Strict per-id validation with worst values on failure.
  - Baselines frozen in `eval/`.
- The paper's adaptive ensemble (logistic regression, the paper's synthetic scale) beats all four paper single-model baselines in every experiment, by +0.06 to +0.11 team accuracy.
- The paper's own College Admission alpha sweep now reproduces the released numbers exactly. It is kept as an unscored check in `scoring/paper_reproduction.json`.
- See HARDENING.md.

### [2026-05-31 17:50] - secondary_claims
- Final verification confirmed the claimed `secondary_claims` milestone is justified.
- `python validate.py --compare` passes with the five College Admission alpha-sweep experiments, train/test score files, and structural separation checks.
- Cleaned final packaging by ignoring/removing tracked job logs, correcting the overview milestone, and marking AI-only train/test diagnostics as zero-coefficient metrics so they do not distort the gym score.

### [2026-05-31 17:45] - secondary_claims
- Final verification passed `python validate.py --compare`.
- Tightened method-agnostic briefing and scoring documentation so from-scratch scientists see the task, metrics, and constraints without implementation hints.
- Confirmed the remaining limitation is scope: the gym reproduces the synthetic College Admission core and human-reliability sweep, not the WoofNette image benchmark.

### [2026-05-31 17:25] - secondary_claims
- Continuation audit found the existing package still implements the paper's synthetic CGPR adaptive-ensemble method and has prior passing CPU-container validation.
- Submitting a fresh CPU end-to-end reproduction job to regenerate the full College Admission alpha sweep and re-run `python validate.py --compare` in the compute container.

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

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 7 total (6 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-11-24 06:45] - secondary_claims
- Final hardening round (job `bc60a3d0-ddf`):
  - The rebuilt gym and the reference scores are byte-identical to the committed ones.
  - `validate.py --compare --post-design` passes.
- The trust (`reliance_probability`) coefficient went from 0.4 to 0.5. A single gradient-boosted behaviour-aware model now scores −0.0022 below the reference (SE 0.0009); before, it tied at −0.0008.
- Better variants score +0.0043 to +0.0155 (SE ≤ 0.0010). The hidden-generator bound is +0.0321. Every shortcut scores 0 or the worst values.
- The authors' release is fetched by `scripts/download.sh` (pinned commit) instead of being tracked in git.
- HARDENING.md written.

### [2026-11-24 01:40] - secondary_claims
- Hardening (iteration 11): the scored hidden test was rebuilt as an extended College Admission world.
  - Answers and the generator are only in `eval/test/`.
  - Four experiments: group known or hidden, crossed with reliable or less reliable evaluators.
  - Exact expected CGPR team accuracy.
  - Strict per-id validation with worst values on failure.
  - Baselines frozen in `eval/`.
- The paper's adaptive ensemble (logistic regression, the paper's synthetic scale) beats all four paper single-model baselines in every experiment, by +0.06 to +0.11 team accuracy.
- The paper's own College Admission alpha sweep now reproduces the released numbers exactly. It is kept as an unscored check in `scoring/paper_reproduction.json`.
- See HARDENING.md.

### [2026-05-31 17:50] - secondary_claims
- Final verification confirmed the claimed `secondary_claims` milestone is justified.
- `python validate.py --compare` passes with the five College Admission alpha-sweep experiments, train/test score files, and structural separation checks.
- Cleaned final packaging by ignoring/removing tracked job logs, correcting the overview milestone, and marking AI-only train/test diagnostics as zero-coefficient metrics so they do not distort the gym score.

### [2026-05-31 17:45] - secondary_claims
- Final verification passed `python validate.py --compare`.
- Tightened method-agnostic briefing and scoring documentation so from-scratch scientists see the task, metrics, and constraints without implementation hints.
- Confirmed the remaining limitation is scope: the gym reproduces the synthetic College Admission core and human-reliability sweep, not the WoofNette image benchmark.

### [2026-05-31 17:25] - secondary_claims
- Continuation audit found the existing package still implements the paper's synthetic CGPR adaptive-ensemble method and has prior passing CPU-container validation.
- Submitting a fresh CPU end-to-end reproduction job to regenerate the full College Admission alpha sweep and re-run `python validate.py --compare` in the compute container.

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
