# Overview

Paper: "Align When They Want, Complement When They Need! Human-Centered
Ensembles for Adaptive Human-AI Collaboration" by Hasan Amin, Ming Yin, and
Rajiv Khanna.

The paper's claim is that, under a realistic model of when humans take AI
advice, an AI that agrees with humans where they are confident and corrects
them where they are not beats any single AI model on human-AI team accuracy.
This holds both when the human's region is known and when the AI must infer
it.

The gym scores this on an admissions world that extends the paper's
synthetic College Admission data:
- two applicant groups with different outcome rules;
- context features that reveal the group only partly;
- evaluators who are confident and sometimes confidently biased in one
  group, and unsure in the other;
- confidence thresholds drawn per decision.

The paper's adaptive ensemble is the reference. It beats the paper's
single-model baselines in all four scored experiments: group known or hidden,
crossed with reliable or less reliable evaluators.

The paper's own College Admission sweep is also reproduced, unscored and
exactly matching the released numbers. Milestone: `secondary_claims` (the
WoofNette image benchmark is not reproduced).
