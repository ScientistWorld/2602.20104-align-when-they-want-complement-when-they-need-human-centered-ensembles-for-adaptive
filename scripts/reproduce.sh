#!/bin/bash
# Reproduce the paper's results end-to-end (runs inside the compute container).
#
# Part 1 - the gym: the paper's adaptive ensemble on the admissions world.
#   Writes checkpoints/submission/ (the reference), scoring/scores_train.json
#   and scoring/scores.json (visible dev slice). The hidden test
#   (scripts/evaluate_test.sh) scores the same predictions on held-out rows.
#   Baselines are frozen in eval/; `bash scripts/baseline.sh` regenerates them.
#
# Part 2 - unscored check on the paper's own College Admission generator:
#   regenerates the authors' imperfect-human alpha sweep (same RNG protocol),
#   runs method/college_admission_method.py and applies the authors' CGPR
#   protocol; writes scoring/paper_reproduction.json for comparison with the
#   paper's released numbers in scoring/reference.json["_paper_reported"].

set -e

cd "$(dirname "$0")/.."

# ---- Part 1
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate.sh

# ---- Part 2
OUT=tmp/paper_reproduction
mkdir -p "$OUT"
python data/college_admission.py \
  --sweep-alphas 0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 \
  --sweep-output-root data --n-instances 10000 --aligned-fraction 0.5 \
  --delta 0.25 --threshold 0.5 --complementary-accuracy 0.5 --seed 42

paper_setting() {
  local alpha="$1" exp="$2" dir="$3"
  python method/college_admission_method.py --data-dir "data/$dir" --output-dir "$OUT/$dir" \
    --aligned-fraction 0.5 --alpha "$alpha" --complementary-accuracy 0.5 --tau 0.5 --seed 42
  python eval/paper_reproduction_eval.py --predictions "$OUT/$dir/predictions.csv" \
    --data-dir "data/$dir" --experiment-name "$exp" --output "$OUT/$exp.json"
}

paper_setting 0.5 college_admission_alpha_0_5 college_admission_alpha_0_5
paper_setting 0.7 college_admission_alpha_0_7 college_admission_alpha_0_7
paper_setting 0.8 college_admission_alpha_0_8 college_admission_alpha_0_8
paper_setting 0.9 college_admission_core college_admission
paper_setting 1.0 college_admission_alpha_1_0 college_admission_alpha_1

python eval/merge_scores.py \
  "$OUT/college_admission_alpha_0_5.json" "$OUT/college_admission_alpha_0_7.json" \
  "$OUT/college_admission_alpha_0_8.json" "$OUT/college_admission_core.json" \
  "$OUT/college_admission_alpha_1_0.json" --output scoring/paper_reproduction.json
