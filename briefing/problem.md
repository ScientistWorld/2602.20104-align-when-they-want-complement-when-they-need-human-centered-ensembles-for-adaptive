# Problem Definition

The task is AI-assisted binary decision making with a human decision maker who
retains final control. Each instance has features, a ground-truth label, a human
initial prediction, and a human confidence score. An AI system provides one
recommendation per instance.

Success is measured after simulating the final human-AI team decision. A useful
assistant must improve the final team accuracy on cases where humans are weak,
while preserving enough agreement with confident human judgments that humans
remain willing to rely on the AI when they are uncertain.

The reusable benchmark in this workspace uses a synthetic College Admission
setting. Applicants are described by GPA and standardized test score. The data
contains two regimes: one where human predictions are usually more reliable and
confidence is higher, and one where human predictions are weaker and advice is
more valuable. Methods may use any recommendation strategy, as long as they
write predictions in the format consumed by the evaluator.
