#!/bin/bash
# Job script: rebuild the gym, reproduce the reference end to end, score it on
# the hidden test, run the hardening checks (variants and shortcuts through
# the hidden test, see redteam/v2/), and validate.
#
# The world-design sweep that chose the gym's world is
# `python redteam/v2/design_sweep.py <worlds>` (results in redteam/v2/design_sweep*.md).

set -e

cd /home/user

# Jobs get one CPU: keep BLAS/OpenMP from starting a thread per node core.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p tmp
echo "[run] $(date) build"
bash scripts/build_world.sh
echo "[run] $(date) determinism check (empty = rebuilt files identical to the committed ones)"
git status --short data eval/test/answers eval/test/baselines eval/train/baselines || true
echo "[run] $(date) reproduce"
bash scripts/reproduce.sh
echo "[run] $(date) hidden test"
bash scripts/evaluate_test.sh
echo "[run] $(date) determinism check of the reference scores (empty = identical)"
git status --short scoring/scores_test.json scoring/scores_train.json scoring/scores.json scoring/paper_reproduction.json || true
echo "[run] $(date) hardening checks"
python redteam/v2/run_checks.py
echo "[run] $(date) validate"
python validate.py --compare --post-design
