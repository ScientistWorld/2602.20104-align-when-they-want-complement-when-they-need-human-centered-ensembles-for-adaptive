# Evaluation Targets

Every target below is an entry in `scoring/scores.json` (dev slice),
`scoring/scores_train.json` (dev slice) and `scoring/scores_test.json`
(hidden test).

## Primary: team accuracy

The expected accuracy of the final human-AI team decision under
confidence-gated probabilistic reliance (threshold tau ~ U[0.6, 0.8]).

- **Metric**: `team_accuracy` (higher is better, coefficient 1.0)
- **Experiments and weights**:
  - `admission_group_hidden` (0.35)
  - `admission_group_hidden_low_reliability` (0.20)
  - `admission_group_known` (0.25)
  - `admission_group_known_low_reliability` (0.20)
- **Run**: `bash scripts/reproduce.sh` (writes `checkpoints/submission/` and
  the dev scores), then `bash scripts/evaluate_test.sh` (hidden test).

## Constraint: trust (reliance)

The agreement between the AI and the human's own judgement where the human
feels sure. It is the probability that unsure humans take the AI's advice. An
assistant that gains accuracy by contradicting confident humans loses it here.

- **Metric**: `reliance_probability` (higher is better, coefficient 0.5)
- **Experiments**: all four.

## Paper claim checked in the scores

The paper's adaptive ensemble (the reference `submission`) should beat each
of the paper's single-model AIs (`standard_ai`, `behavior_aware_ai`,
`aligned_ai`, `complementary_ai`) on team accuracy in every experiment.

## Unscored reproduction

The paper's own College Admission imperfect-human sweep (alpha = 0.5, 0.7,
0.8, 0.9, 1.0) is written to `scoring/paper_reproduction.json` by
`scripts/reproduce.sh`. It is compared with the released numbers kept in
`scoring/reference.json["_paper_reported"]`.
