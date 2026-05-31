# Progress

## What Works

- Paper analysis is complete for the synthetic College Admission setting.
- Implemented tracked data-generation code for the paper's two-region synthetic task, including held-out human predictions and confidence values.
- Implemented the paper's logistic behavior-aware single model, aligned specialist, complementary specialist, oracle adaptive ensemble, weighted oracle ensemble, and RRS routing.
- Implemented an evaluator that reads method prediction artifacts, joins them to held-out problem data, recomputes CGPR team decisions, and writes `scoring/scores.json`.
- Updated the scorer to include the practical RRS adaptive variant in addition to oracle routing.

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

Current turn: submitting a CPU reproduction refresh so `scoring/scores.json` includes the RRS metrics already written by the method.

Continuation audit: the implementation remains a real reproduction of the
paper's synthetic CGPR/adaptive-ensemble procedure, and prior CPU container jobs
validated the core and alpha=0.7 settings. The persisted scores predate the RRS
scorer update, so a new CPU test submission is needed to refresh
`scoring/scores.json` inside the container. A login-node attempt failed because
the host Python lacks `numpy`; dependencies are installed by
`environment/setup.sh` in the compute container.

## Remaining

- Higher milestones would require reproducing additional axes from the paper, such as the WoofNette image benchmark or more synthetic sweeps. The current packaged gym covers the core synthetic claim and one additional human-accuracy setting.

## Issues

- Login-node Python does not have the scientific Python stack installed. This is expected to be handled by `environment/setup.sh` during the submitted job.
- Audit found that the first implementation let `method/` generate the human behavior used for evaluation and mixed a WoofNette reference number into the College Admission score table. This turn moved human behavior into tracked data generation, made `eval/` recompute team decisions from held-out data, and replaced references with College Admission values from the authors' released CSV.
- The job log paths reported by the scheduler were mirrored under `jobs/9cf433ae-84c/` in this workspace; the direct absolute scratch paths in the notification were not readable from the current shell.

## Deviations from Paper

- The first runnable target focuses on the paper's synthetic College Admission benchmark rather than the WoofNette image benchmark. This preserves the paper's algorithmic procedure while avoiding large image/model downloads before the core synthetic claim is verified.
