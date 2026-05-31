# Experiment Overview

## college_admission_core

This experiment checks whether an AI assistant can improve final team decisions
in a two-region admission task where humans are reliable and confident in one
region but weak in the other. The important outcome is not just team accuracy:
the assistant must also preserve the aligned-region reliance signal that
determines whether humans will use advice when they are uncertain. Adaptive
systems are scored under the paper-reported synthetic condition where routing
has access to the synthetic region label.

## college_admission_alpha_0_7

This experiment repeats the same held-out task with lower human accuracy in the
region where humans are usually reliable. It checks whether the evaluation
surface remains meaningful when human competence changes, rather than only at
the easiest high-accuracy setting.
