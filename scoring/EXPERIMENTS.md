# Experiment Overview

## college_admission_core

This experiment checks whether an AI assistant can improve final team decisions
in a two-region admission task where humans are reliable and confident in one
region but weak in the other. The important outcome is not just team accuracy:
the assistant must also preserve the agreement signal in the human-reliable
regime that determines whether humans will use advice when they are uncertain.
Methods that emit a regime choice are also checked against the known synthetic
regime label.

## college_admission_alpha_0_7

This experiment repeats the same held-out task with lower human accuracy in the
region where humans are usually reliable. It checks whether the evaluation
surface remains meaningful when human competence changes, rather than only at
the easiest high-accuracy setting.

## college_admission_alpha_0_5 / 0_8 / 1_0

These experiments extend the same human-reliability sweep to a weak
human-reliable-regime signal, a moderately strong signal, and a perfect signal.
Together with the core and alpha=0.7 settings, they test whether the benchmark
captures the claim that behavior-aware assistance becomes more valuable as the
reliable human regime provides a stronger trust signal.
