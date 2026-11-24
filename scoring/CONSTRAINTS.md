# Scientist Constraints

## Inputs

- At test time a method may use only what `test_inputs.csv` gives: the
  applicant features, plus `group` in the `admission_group_known*`
  experiments.
- The human's judgement, confidence and threshold are training-time
  information only.
- Train on `train.csv`. Tuning on `dev.csv` is fine. The dev answers are
  visible, so dev scores of a dev-tuned method are optimistic. Only the
  hidden test counts.

## Outputs

- Write `checkpoints/<method>/<experiment>/predictions.csv` with columns
  `example_id,prediction`, with exactly one 0/1 prediction for every dev and
  test-input id.
- The ranked entry is `checkpoints/submission/`.
- A missing or malformed file scores the experiment as failed.

## Evaluation

- Do not modify `eval/`. The hidden test runs the gym's own copy.
- The scorers read nothing but the predictions, so edits to `data/`,
  `scoring/` or baseline-named method directories do not change hidden-test
  scores.
- The team-decision model (threshold distribution, reliance rule) is part of
  the task, not a parameter of the method.

## Scope

- Improvements should come from recommendation methodology: modelling where
  humans feel sure, what they would decide, what the outcome is, and how to
  combine these under the reliance rule.
- Do not score or optimise wall-clock speed.

## Compliance Checklist

- [ ] Recommendations use only test-time inputs
- [ ] One binary prediction per dev and test-input id, per experiment
- [ ] `eval/` untouched and independent of `method/`
- [ ] Report `team_accuracy` and `reliance_probability`
