#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

mkdir -p tmp/reproduce_scores

python data/college_admission.py \
  --sweep-alphas 0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 \
  --sweep-output-root data \
  --n-instances "${N_INSTANCES:-10000}" \
  --aligned-fraction "${ALIGNED_FRACTION:-0.5}" \
  --delta "${DELTA:-0.25}" \
  --threshold "${TAU:-0.5}" \
  --complementary-accuracy "${COMPLEMENTARY_ACCURACY:-0.5}" \
  --seed "${SEED:-42}"

run_alpha_setting() {
  local alpha="$1"
  local experiment_name="$2"
  local data_dir="$3"
  local output_dir="$4"
  local scores_file="$5"

  ALPHA="$alpha" \
  DATA_DIR="$data_dir" \
  OUTPUT_DIR="$output_dir" \
  SCORES_OUTPUT="$scores_file" \
  EXPERIMENT_NAME="$experiment_name" \
  GENERATE_DATA=0 \
  bash scripts/method.sh
}

run_alpha_setting 0.5 college_admission_alpha_0_5 data/college_admission_alpha_0_5 checkpoints/college_admission_alpha_0_5 tmp/reproduce_scores/college_admission_alpha_0_5.json
run_alpha_setting 0.7 college_admission_alpha_0_7 data/college_admission_alpha_0_7 checkpoints/college_admission_alpha_0_7 tmp/reproduce_scores/college_admission_alpha_0_7.json
run_alpha_setting 0.8 college_admission_alpha_0_8 data/college_admission_alpha_0_8 checkpoints/college_admission_alpha_0_8 tmp/reproduce_scores/college_admission_alpha_0_8.json
run_alpha_setting 0.9 college_admission_core data/college_admission checkpoints/college_admission tmp/reproduce_scores/college_admission_core.json
run_alpha_setting 1.0 college_admission_alpha_1_0 data/college_admission_alpha_1_0 checkpoints/college_admission_alpha_1_0 tmp/reproduce_scores/college_admission_alpha_1_0.json

python eval/merge_scores.py \
  tmp/reproduce_scores/college_admission_alpha_0_5.json \
  tmp/reproduce_scores/college_admission_core.json \
  tmp/reproduce_scores/college_admission_alpha_0_7.json \
  tmp/reproduce_scores/college_admission_alpha_0_8.json \
  tmp/reproduce_scores/college_admission_alpha_1_0.json \
  --output scoring/scores.json
