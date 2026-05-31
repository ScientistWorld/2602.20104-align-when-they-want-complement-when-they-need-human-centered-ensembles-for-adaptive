# Progress

## What Works

- Paper analysis is complete for the synthetic College Admission setting.
- Implemented tracked data-generation code for the paper's two-region synthetic task, including held-out human predictions and confidence values.
- Implemented the paper's logistic behavior-aware single model, aligned specialist, complementary specialist, oracle adaptive ensemble, weighted oracle ensemble, and RRS routing.
- Implemented an evaluator that reads method prediction artifacts, joins them to held-out problem data, recomputes CGPR team decisions, and writes `scoring/scores.json`.

## Results

Job `9cf433ae-84c` completed successfully in the compute container and wrote `scoring/scores.json`.

Core College Admission setting at alpha=0.9:

- `adaptive_oracle`: team accuracy 0.8063 versus paper 0.806; reliance 0.8850 versus paper 0.8795; routing accuracy 1.0.
- `adaptive_weighted_oracle`: team accuracy 0.7967 versus paper 0.7947; reliance 0.8817 versus paper 0.8737; routing accuracy 1.0.
- Baselines are numerically plausible: behavior-aware 0.7603, aligned 0.7380, complementary 0.7727 team accuracy.

The core claim is reproduced: adaptive routing between aligned and complementary specialists improves over the single behavior-aware model and component specialists while preserving the reliance signal.

## Remaining

- Run the added alpha=0.7 College Admission setting to expand beyond the core point.
- Inspect the merged two-experiment `scoring/scores.json` and update the milestone if the second setting is also reproduced.

## Issues

- Login-node Python does not have the scientific Python stack installed. This is expected to be handled by `environment/setup.sh` during the submitted job.
- Audit found that the first implementation let `method/` generate the human behavior used for evaluation and mixed a WoofNette reference number into the College Admission score table. This turn moved human behavior into tracked data generation, made `eval/` recompute team decisions from held-out data, and replaced references with College Admission values from the authors' released CSV.
- The job log paths reported by the scheduler were mirrored under `jobs/9cf433ae-84c/` in this workspace; the direct absolute scratch paths in the notification were not readable from the current shell.

## Deviations from Paper

- The first runnable target focuses on the paper's synthetic College Admission benchmark rather than the WoofNette image benchmark. This preserves the paper's algorithmic procedure while avoiding large image/model downloads before the core synthetic claim is verified.
