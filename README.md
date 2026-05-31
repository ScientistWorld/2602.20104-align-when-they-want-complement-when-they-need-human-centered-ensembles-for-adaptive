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


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
