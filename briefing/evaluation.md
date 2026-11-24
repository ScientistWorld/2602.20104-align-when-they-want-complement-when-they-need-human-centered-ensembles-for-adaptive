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
- `reliance_probability` (constraint: trust, coefficient 0.4) is the r above:
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
