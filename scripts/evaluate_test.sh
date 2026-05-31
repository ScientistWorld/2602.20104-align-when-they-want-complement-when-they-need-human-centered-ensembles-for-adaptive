#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Argument shape mirrors scripts/evaluate.sh:
#   ./evaluate_test.sh <predictions.csv> <output.json> <data_dir> <experiment_name>
#
# With no arguments, scores every experiment in scoring/reference.json by
# enumerating *_pred columns in each predictions.csv and writes
# scoring/scores_test.json.

set -e

cd /home/user

mkdir -p scoring

PREDICTIONS="${1:-}"
OUTPUT="${2:-scoring/scores_test.json}"
DATA_DIR="${3:-}"
EXPERIMENT_NAME="${4:-}"
TRAIN_SCORES="${TRAIN_SCORES:-scoring/scores_train.json}"

if [[ -n "$PREDICTIONS" ]]; then
  python eval/test/evaluate.py \
    --predictions "$PREDICTIONS" \
    --data-dir "${DATA_DIR:-data/college_admission}" \
    --experiment-name "${EXPERIMENT_NAME:-college_admission_core}" \
    --method-source "$TRAIN_SCORES" \
    --output "$OUTPUT"
else
  python eval/test/evaluate.py \
    --method-source "$TRAIN_SCORES" \
    --output "$OUTPUT"
fi
