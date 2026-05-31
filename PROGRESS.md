# Progress

## What Works

- Paper analysis is complete for the synthetic College Admission setting.
- Implemented tracked data-generation code for the paper's two-region synthetic task, including held-out human predictions and confidence values.
- Implemented the paper's logistic behavior-aware single model, aligned specialist, complementary specialist, oracle adaptive ensemble, weighted oracle ensemble, and RRS routing.
- Implemented an evaluator that reads method prediction artifacts, joins them to held-out problem data, recomputes CGPR team decisions, and writes `scoring/scores.json`.

## Results

No completed job results yet. A local smoke test could not run because the login-node Python environment lacks NumPy; dependencies are declared in `environment/setup.sh` for the compute container.

## Remaining

- Run the packaged pipeline in the container.
- Inspect `scoring/scores.json` and compare against `scoring/reference.json`.
- If the core claim is supported, update milestone to `core_claim`; otherwise debug and resubmit.

## Issues

- Login-node Python does not have the scientific Python stack installed. This is expected to be handled by `environment/setup.sh` during the submitted job.
- Audit found that the first implementation let `method/` generate the human behavior used for evaluation and mixed a WoofNette reference number into the College Admission score table. This turn moved human behavior into tracked data generation, made `eval/` recompute team decisions from held-out data, and replaced references with College Admission values from the authors' released CSV.

## Deviations from Paper

- The first runnable target focuses on the paper's synthetic College Admission benchmark rather than the WoofNette image benchmark. This preserves the paper's algorithmic procedure while avoiding large image/model downloads before the core synthetic claim is verified.
