> **Superseded (iteration 11, hardening).** This file describes the gym before it was hardened (the five-setting alpha sweep scored on visible test files, with a routing metric). The scored gym is now the hidden-test admissions world described in `DETAILS.md` and `briefing/`; what changed and why is in `HARDENING.md`. Kept as a historical record.

# Train/Test Evaluation Design Notes

## Checks Performed

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train script invokes `eval/train/evaluate.py` and writes `scoring/scores_train.json`; the test script invokes `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Checked for train/test cross-imports and score-stub artifacts before the runner job; the split score files were then produced by job `2100744c-980`.
- Checked `scoring/reference.json` for paper-agreement metrics. The active metrics measure team accuracy, reliance, standalone accuracy, per-region accuracy, and routing quality rather than distance to published values.
- Confirmed the split evaluators read method-produced prediction artifacts from `checkpoints/*/predictions.csv` rather than regenerating the method inline.

## Changes Made

- Changed both train and test slice evaluators to score every method declared for the experiment in `scoring/reference.json`. This avoids hardcoded single-method scoring while keeping split score files compatible with the validator's reference-method registry.
- Added three triangulation metrics to both slices: `ai_accuracy`, `aligned_region_accuracy`, and `complementary_region_accuracy`. These make a shortcut that only optimizes the CGPR team metric or only echoes human decisions easier to detect.
- Made `routing_accuracy` total rather than optional: methods without an explicit `<method>_route_aligned` column now receive `routing_accuracy = 0.0`. This closes the loophole where a method could avoid a routed-assistance constraint by omitting the route artifact.
- Added required `description` fields to every metric entry in `scoring/reference.json` and registered the new triangulation metrics across all existing experiments.
- Updated train/test script comments to reflect prediction-column method discovery.

## Runtime Concerns For Runner Iteration

- The train/test split is a deterministic 50/50 hash partition of the existing labeled test CSVs. That is lightweight and comparable across slices, but it is not a hidden-data split; a deliberately label-memorizing method could still exploit the visible CSV labels.
- `routing_accuracy` can saturate at 1.0 for methods that output the synthetic region label directly. This matches the paper's oracle-routing synthetic condition, but future hidden tests should watch whether this metric adds discrimination beyond artifact completeness.
- Both split evaluators still assume each experiment has `data/<experiment>/test.csv`, `data/<experiment>/metadata.csv`, and `checkpoints/<experiment>/predictions.csv` with a one-to-one `example_id` merge.

## Runner Assessment

- Job `2100744c-980` exited with code 0 and produced `scoring/scores_train.json` and `scoring/scores_test.json`.
- Coverage is complete after the validator fix: 5 experiments, 5 reference-declared methods per experiment, and all 6 reference metrics populated on both train and test slices.
- Train and test method sets match exactly for every experiment. The hash split sizes are balanced, ranging from 1492/1508 to 1527/1473 examples.
- The scores are not byte-identical across slices. Maximum per-method train/test gaps are modest for the primary metric (`team_accuracy`): 0.022 on core, 0.041 at alpha 0.5, 0.010 at alpha 0.7, 0.013 at alpha 0.8, and 0.010 at alpha 1.0.
- No active benefit metric is constant across every method on either slice. Oracle `routing_accuracy` remains 1.0 by construction while non-routed methods score 0.0, so it functions as an artifact/constraint check rather than a standalone benefit score.

## Validation Fix

- The first validator run rejected `standard_ai` and `adaptive_rrs` because they were discovered from `*_pred` columns but are not registered methods in `scoring/reference.json`.
- The split evaluators now restrict the default suite to the methods declared under each experiment's `results` block in `scoring/reference.json`; this preserves multi-method scoring and removes unregistered methods from regenerated split scores.
