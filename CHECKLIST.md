# Reproduction Checklist

## Briefing
- [x] `briefing/problem.md` (method-agnostic)
- [x] `briefing/evaluation.md` (method-agnostic)
- [x] `briefing/method.md`
- [x] `briefing/overview.md`

## Scoring
- [x] `scoring/reference.json`
- [x] Workspace validated via `python validate.py --reference-only`
- [x] `scoring/scores.json`
- [x] `scoring/EXPERIMENTS.md`
- [x] `scoring/TARGETS.md`
- [x] `scoring/CONSTRAINTS.md`
- [x] `scoring/DIRECTION.md`

## Code
- [x] Evaluation code in `eval/`
- [x] Method implementation in `method/`
- [ ] Baseline implementation in `baseline/` (optional)

## Scripts
- [x] `scripts/evaluate.sh`
- [x] `scripts/reproduce.sh`
- [x] `scripts/download.sh`
- [ ] `scripts/baseline.sh` (optional)

## Environment
- [x] `environment/container.def` + `environment/setup.sh`

## Data
- [x] Synthetic data generator implemented
- [x] No pretrained models needed for the core synthetic benchmark

## Milestones
- [x] `method_runs`
- [x] `core_claim`
- [x] `core_claim_plus`
- [x] `secondary_claims`
- [ ] `majority`
- [ ] `near_complete`
- [ ] `full`
