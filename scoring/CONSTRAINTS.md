# Scientist Constraints

## Data

Use the provided College Admission data generation protocol and held-out split.
Do not alter ground-truth labels, human predictions, human confidence values, or
the provided probabilistic reliance evaluation rule when comparing methods.

## Evaluation

Methods must write the prediction format consumed by `scripts/evaluate.sh`.
The evaluator must remain method-agnostic and must not import from `method/`.

## Scope

Do not score wall-clock speed. Improvements should come from recommendation,
human-behavior modeling, calibration, or regime-selection methodology rather
than from changing the benchmark problem.

## Compliance Checklist

- [ ] Uses the tracked College Admission generator and test split
- [ ] Does not train on the held-out test labels except through evaluation
- [ ] Leaves `eval/` independent of `method/`
- [ ] Reports `team_accuracy` and `reliance_probability`
