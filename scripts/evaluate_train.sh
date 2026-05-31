#!/bin/bash
# Visible train-slice evaluation entry point.
#
# Argument shape mirrors scripts/evaluate.sh:
#   ./evaluate_train.sh <predictions.csv> <output.json> <data_dir> <experiment_name>
#
# With no arguments, scores every experiment in scoring/reference.json by
# enumerating *_pred columns in each predictions.csv and writes
# scoring/scores_train.json.

set -e

cd /home/user

mkdir -p scoring

PREDICTIONS="${1:-}"
OUTPUT="${2:-scoring/scores_train.json}"
DATA_DIR="${3:-}"
EXPERIMENT_NAME="${4:-}"

if [[ -n "$PREDICTIONS" ]]; then
  python eval/train/evaluate.py \
    --predictions "$PREDICTIONS" \
    --data-dir "${DATA_DIR:-data/college_admission}" \
    --experiment-name "${EXPERIMENT_NAME:-college_admission_core}" \
    --method-source scoring/scores.json \
    --output "$OUTPUT"
else
  python eval/train/evaluate.py \
    --method-source scoring/scores.json \
    --output "$OUTPUT"
fi
