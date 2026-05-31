# Train/Test Evaluation Design Notes

## Checks Performed

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train script invokes `eval/train/evaluate.py` and writes `scoring/scores_train.json`; the test script invokes `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Checked for train/test cross-imports and score-stub artifacts. No `scoring/scores_train.json` or `scoring/scores_test.json` is present.
- Checked `scoring/reference.json` for paper-agreement metrics. The active metrics measure team accuracy, reliance, standalone accuracy, per-region accuracy, and routing quality rather than distance to published values.
- Confirmed the split evaluators read method-produced prediction artifacts from `checkpoints/*/predictions.csv` rather than regenerating the method inline.

## Changes Made

- Changed both train and test slice evaluators to enumerate all `*_pred` columns in each predictions file by default. This avoids silently ignoring future agent variants that add new method outputs without editing `scoring/scores.json`.
- Added three triangulation metrics to both slices: `ai_accuracy`, `aligned_region_accuracy`, and `complementary_region_accuracy`. These make a shortcut that only optimizes the CGPR team metric or only echoes human decisions easier to detect.
- Made `routing_accuracy` total rather than optional: methods without an explicit `<method>_route_aligned` column now receive `routing_accuracy = 0.0`. This closes the loophole where a method could avoid a routed-assistance constraint by omitting the route artifact.
- Added required `description` fields to every metric entry in `scoring/reference.json` and registered the new triangulation metrics across all existing experiments.
- Updated train/test script comments to reflect prediction-column method discovery.

## Runtime Concerns For Runner Iteration

- The train/test split is a deterministic 50/50 hash partition of the existing labeled test CSVs. That is lightweight and comparable across slices, but it is not a hidden-data split; a deliberately label-memorizing method could still exploit the visible CSV labels.
- `routing_accuracy` can saturate at 1.0 for methods that output the synthetic region label directly. This matches the paper's oracle-routing synthetic condition, but future hidden tests should watch whether this metric adds discrimination beyond artifact completeness.
- The added per-region metrics should be checked for saturation on the current reproduction once the runner iteration is allowed to execute evaluators.
- Both split evaluators still assume each experiment has `data/<experiment>/test.csv`, `data/<experiment>/metadata.csv`, and `checkpoints/<experiment>/predictions.csv` with a one-to-one `example_id` merge.
