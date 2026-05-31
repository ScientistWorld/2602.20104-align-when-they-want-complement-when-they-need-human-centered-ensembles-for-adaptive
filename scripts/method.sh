#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

DATA_DIR="${DATA_DIR:-data/college_admission}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/college_admission}"
SCORES_OUTPUT="${SCORES_OUTPUT:-scoring/scores.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-college_admission_core}"
N_INSTANCES="${N_INSTANCES:-10000}"
ALIGNED_FRACTION="${ALIGNED_FRACTION:-0.5}"
DELTA="${DELTA:-0.25}"
ALPHA="${ALPHA:-0.9}"
COMPLEMENTARY_ACCURACY="${COMPLEMENTARY_ACCURACY:-0.5}"
TAU="${TAU:-0.5}"
SEED="${SEED:-42}"
GENERATE_DATA="${GENERATE_DATA:-1}"

if [ "$GENERATE_DATA" != "0" ]; then
  python data/college_admission.py \
    --output-dir "$DATA_DIR" \
    --n-instances "$N_INSTANCES" \
    --aligned-fraction "$ALIGNED_FRACTION" \
    --delta "$DELTA" \
    --threshold "$TAU" \
    --alpha "$ALPHA" \
    --complementary-accuracy "$COMPLEMENTARY_ACCURACY" \
    --seed "$SEED"
fi

python method/college_admission_method.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --aligned-fraction "$ALIGNED_FRACTION" \
  --alpha "$ALPHA" \
  --complementary-accuracy "$COMPLEMENTARY_ACCURACY" \
  --tau "$TAU" \
  --seed "$SEED"

bash scripts/evaluate.sh "$OUTPUT_DIR/predictions.csv" "$SCORES_OUTPUT" "$DATA_DIR" "$EXPERIMENT_NAME"
