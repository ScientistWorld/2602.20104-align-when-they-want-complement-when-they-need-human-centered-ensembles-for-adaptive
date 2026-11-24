# Red-team and hardening checks

- `v2/` checks the current, hardened gym. `python redteam/v2/run_checks.py` (run by
  `scripts/run.sh` after the reference) scores variants, shortcuts and sandboxed
  workspace edits through the hidden test and writes `v2/summary.{json,md}`.
  - `summary_round1*`, `summary_round2*` and `summary_round3_coef0.4*` are earlier
    rounds of the same checks, kept for the before/after record in `HARDENING.md`.
  - `design_sweep*` holds the world-design sweep that chose the gym's world.
- Every other folder (`run_all.sh`, `shortcut_*`, `data_scaling`, `seed_spread`, ...)
  belongs to the iteration-10 audit (`REDTEAM.md`) of the gym *before* hardening. Those
  scripts target files and method names that no longer exist and are kept as a record.
