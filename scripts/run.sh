#!/bin/bash
# Job script: rebuild the gym, reproduce the reference end to end, score it on
# the hidden test, then run the hardening checks (variants, shortcuts, world
# sensitivity) through the hidden test.

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
echo "[run] $(date) hardening checks"
python redteam/v2/run_checks.py
echo "[run] $(date) world sweep"
python redteam/v2/design_sweep.py default,wider_gate,sharp_gate,no_red_flag
echo "[run] $(date) validate"
python validate.py --compare --post-design || echo "validate.py reported issues (reference.json results are filled in after this job)"
