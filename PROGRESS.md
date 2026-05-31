# Progress

## What Works

- Paper analysis is complete for the synthetic College Admission setting.
- Implemented tracked data-generation code for the paper's two-region synthetic task, including held-out human predictions and confidence values.
- Implemented the paper's logistic single-model surrogate, aligned specialist, complementary specialist, oracle adaptive ensemble, weighted oracle ensemble, and RRS routing.
- Implemented an evaluator that reads method prediction artifacts, joins them to held-out problem data, recomputes CGPR team decisions, and writes `scoring/scores.json`.
- The method writes RRS prediction artifacts, while the default scorer includes the paper-reported synthetic methods that have reference values.
- Inspected the official repository linked by the paper and confirmed the packaged synthetic reproduction follows the released College Admission generator, single-model surrogate, specialist training, and CGPR result columns.
- Current continuation audit confirmed that the default scoring intentionally excludes synthetic RRS from `scores.json` because the paper does not report synthetic RRS reference values; RRS predictions remain implemented as reusable method artifacts.

## Results

Job `9cf433ae-84c` completed successfully in the compute container and wrote `scoring/scores.json`.

Core College Admission setting at alpha=0.9:

- `adaptive_oracle`: team accuracy 0.8063 versus paper 0.806; reliance 0.8850 versus paper 0.8795; routing accuracy 1.0.
- `adaptive_weighted_oracle`: team accuracy 0.7967 versus paper 0.7947; reliance 0.8817 versus paper 0.8737; routing accuracy 1.0.
- Baselines are numerically plausible: behavior-aware 0.7603, aligned 0.7380, complementary 0.7727 team accuracy.

The core claim is reproduced: adaptive routing between aligned and complementary specialists improves over the single behavior-aware model and component specialists while preserving the reliance signal.

Job `29fb451a-dae` completed successfully and expanded the suite to alpha=0.7.

Additional College Admission setting at alpha=0.7:

- `adaptive_oracle`: team accuracy 0.6743 versus paper 0.688; reliance 0.6764 versus paper 0.6898; routing accuracy 1.0.
- `adaptive_weighted_oracle`: team accuracy 0.6550 versus paper 0.6693; reliance 0.6757 versus paper 0.6904; routing accuracy 1.0.
- Baselines remain nontrivial: behavior-aware 0.6543, aligned 0.6220, complementary 0.6603 team accuracy.

The expanded setting is lower than the authors' reported values but keeps the core qualitative pattern: oracle adaptive routing beats the single behavior-aware model and the individual specialists under a different human-accuracy regime.

Current turn: audited the existing package against the official code release and
updated `MILESTONES.md` to reflect the previously validated `core_claim_plus`
state. RRS remains implemented in `method/`, but the synthetic reference table
does not include RRS numbers, so it is not part of the default validated
`scores.json`.

Continuation audit for the reset milestone found no integrity issues:

- `method/` implements the paper's actual synthetic CGPR/adaptive-ensemble
  procedure rather than a surrogate.
- `eval/` is method-independent and reads method prediction artifacts.
- `python validate.py --compare` passed on the existing two-experiment scores.
- The environment files are reasonable and were left unchanged.

Current expansion: added alpha=0.5, alpha=0.8, and alpha=1.0 settings from the
authors' released synthetic human-reliability sweep to `reference.json` and
`scripts/reproduce.sh`. Submitting a CPU job to regenerate `scores.json` for all
five synthetic experiments and validate the expanded suite.

Job `59e5cbe0-420` completed successfully and validated the five-alpha suite,
but the alpha=0.5 adaptive result was weaker than both the paper reference and
simple baselines. Audit against the authors' released `run_imperfect_human`
procedure found the cause: the packaged reproduction generated each alpha
setting independently, used a stratified train/test split, and used a
hash-derived team-decision RNG. The paper's sweep instead uses one shared
dataset, an unstratified split, one RNG stream across the whole alpha sweep, and
a seeded NumPy RNG for each CGPR team decision. The data generator, evaluator,
and reproduction script have been updated to follow that protocol, while still
keeping evaluation independent from `method/`.

Job `772f9aa7-605` failed immediately in data generation because the new sweep
path passed a shared RNG into `generate_college_admission`, but that function's
signature had not yet been updated. The signature now accepts an optional RNG,
which preserves the old seed-based standalone path and enables the paper-style
shared-RNG sweep.

Job `179f7023-580` completed successfully, but the alpha=0.5 adaptive and
aligned-specialist scores still did not match the released sweep qualitatively.
The remaining mismatch was in the human simulator: the released code draws
correctness and confidence sequentially per example, while the packaged version
drew all correctness Bernoulli samples first and all confidences second. The
simulator now follows the per-example RNG order from the authors' code.

Job `d4985458-aae` completed successfully after the simulator fix, and
`python validate.py --compare` passed. The five-setting alpha sweep now
reproduces the paper's secondary imperfect-human reliability pattern:

- alpha=0.5: adaptive oracle 0.609 team accuracy versus behavior-aware 0.579
  and complementary 0.583; paper reference 0.593.
- alpha=0.7: adaptive oracle 0.690 versus behavior-aware 0.667 and
  complementary 0.677; paper reference 0.688.
- alpha=0.8: adaptive oracle 0.753 versus behavior-aware 0.722 and
  complementary 0.730; paper reference 0.753.
- alpha=0.9 core: adaptive oracle 0.810 versus behavior-aware 0.764 and
  complementary 0.777; paper reference 0.806.
- alpha=1.0: adaptive oracle 0.875 versus behavior-aware 0.818 and
  complementary 0.830; paper reference 0.881.

Continuation audit: the implementation remains a real reproduction of the
paper's synthetic CGPR/adaptive-ensemble procedure, and prior CPU container jobs
validated the core and alpha=0.7 settings. A login-node attempt failed because
the host Python lacks `numpy`; dependencies are installed by
`environment/setup.sh` in the compute container.

Job `148f0607-883` failed validation because the scorer included
`adaptive_rrs` while `reference.json` correctly lacked paper-reported synthetic
RRS reference values. The fix is to exclude RRS from default synthetic scoring
while retaining the implementation and prediction artifacts.

Job `f814d317-529` completed successfully after that fix. It regenerated
`scoring/scores.json` in the container and `python validate.py --compare`
passed. The reproduced values remain:

- alpha=0.9 core: adaptive oracle team accuracy 0.8063 versus paper reference
  0.806; behavior-aware baseline 0.7603.
- alpha=0.7 expansion: adaptive oracle team accuracy 0.6743 versus paper
  reference 0.688; behavior-aware baseline 0.6543.

## Remaining

- Higher milestones would require reproducing additional axes from the paper,
  such as the WoofNette image benchmark or broader College Admission sweeps
  beyond imperfect-human reliability.
- A fresh CPU end-to-end validation job is being submitted to regenerate the
  five-setting synthetic alpha sweep and rerun `python validate.py --compare`
  in the compute container.

## Issues

- Login-node Python does not have the scientific Python stack installed. This is expected to be handled by `environment/setup.sh` during the submitted job.
- Audit found that the first implementation let `method/` generate the human behavior used for evaluation and mixed a WoofNette reference number into the College Admission score table. This turn moved human behavior into tracked data generation, made `eval/` recompute team decisions from held-out data, and replaced references with College Admission values from the authors' released CSV.
- The job log paths reported by the scheduler were mirrored under `jobs/9cf433ae-84c/` in this workspace; the direct absolute scratch paths in the notification were not readable from the current shell.
- The direct absolute scratch paths for job `59e5cbe0-420` were also not mounted
  here; the readable logs were under `/home/user/jobs/59e5cbe0-420/`.
- The first five-alpha run exposed a protocol mismatch at alpha=0.5. The fix
  preserves the paper's algorithm and changes only data/evaluation protocol to
  match the released experiment implementation.
- Job `772f9aa7-605` caught a missed function-signature change before any
  expensive work ran; fixed in `data/college_admission.py`.
- Job `179f7023-580` showed that matching the shared RNG stream is not enough
  unless the per-example draw order also matches the paper implementation.

## Deviations from Paper

- The first runnable target focuses on the paper's synthetic College Admission benchmark rather than the WoofNette image benchmark. This preserves the paper's algorithmic procedure while avoiding large image/model downloads before the core synthetic claim is verified.
