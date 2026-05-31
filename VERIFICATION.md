# Verification Report

## Initial Assessment

The workspace claimed `secondary_claims`. That claim is justified: the
implementation reproduces the paper's synthetic College Admission core and the
human-reliability alpha sweep, and the evaluator reads method-produced
artifacts rather than recomputing the method inline. The problem/evaluation
briefings and `scoring/EXPERIMENTS.md` are method-agnostic, while method
details are confined to `briefing/method.md` and `method/`.

## Issues Found

- minor: `briefing/overview.md` still described the result as
  `core_claim_plus`, even though the workspace has five alpha-sweep settings
  and a justified `secondary_claims` milestone.
- minor: `scoring/reference.json` gave nonzero coefficients to AI-only
  diagnostic metrics that are not part of the main `scores.json` contract. This
  could distort train/test scoring incentives away from the actual team and
  reliance claims.
- cosmetic: Generated `jobs/` logs and Python bytecode caches were present in
  the working tree. They are not reusable gym assets.

## Fixes Applied

- Updated `briefing/overview.md` to state the correct `secondary_claims` scope.
- Kept the diagnostic AI-only metric definitions for train/test validation, but
  set their coefficients to `0.0`. The active scoring metrics are now team
  accuracy, reliance probability, and routing accuracy.
- Added `jobs/` to `.gitignore`, removed tracked job-log files, and deleted
  Python bytecode caches.
- Re-ran `python validate.py --compare`; it passes for `reference.json`,
  `scores.json`, `scores_train.json`, `scores_test.json`, train/test split
  checks, import separation, and shared-path checks.

## Final Judgment

The workspace is ready to serve as a research gym for the reproduced synthetic
College Admission problem. Improve-mode agents can compare against the existing
method outputs, and from-scratch agents can use the method-agnostic data and
evaluation contract without seeing implementation details.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: high
- Key limitation: Synthetic College Admission only; the WoofNette/RRS image
  benchmark is not reproduced.
