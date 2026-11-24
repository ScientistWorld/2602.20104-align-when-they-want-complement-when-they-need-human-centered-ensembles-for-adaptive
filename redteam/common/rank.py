"""Score one hidden-test run against the reference hidden-test scores.

Mirrors the stated rank rule as far as it is visible: for every experiment
(weight w_e from reference.json), every metric with a non-zero coefficient c_m
and every method listed in that experiment's reference results,
    contribution = w_e * |c_m| * (run - reference), signed by higher_is_better.
Missing values are skipped (as the rank does) and listed.  The per-metric
division by the spread across future attempts is unknown here, so the numbers
are raw (unnormalised) weighted deltas; the sign is what matters.

Writes <run-dir>/rank.json and prints one line.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = HERE / "reference_spec.json"            # frozen copy of scoring/reference.json
REFERENCE = HERE / "reference_scores_test.json"  # frozen copy of scoring/scores_test.json
PROPOSED = ["adaptive_oracle", "adaptive_weighted_oracle"]


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rc", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    spec = json.loads(SPEC.read_text())["experiments"]
    ref = json.loads(REFERENCE.read_text())["experiments"]
    scores_path = run_dir / "scores_test.json"
    out = {"run": str(run_dir), "evaluate_test_rc": args.rc, "scores_file_present": scores_path.exists()}

    if not scores_path.exists():
        out["status"] = "FAILED: evaluate_test.sh wrote no scoring/scores_test.json"
        (run_dir / "rank.json").write_text(json.dumps(out, indent=2) + "\n")
        print(f"[rank] {run_dir.name}: rc={args.rc} no scores file")
        return

    run = json.loads(scores_path.read_text()).get("experiments", {})
    methods = sorted({m for e in spec.values() for m in e["results"]})
    score_by_method = {}
    missing = []
    detail = {}
    for method in methods:
        total = 0.0
        for exp, exp_spec in spec.items():
            if method not in exp_spec["results"]:
                continue
            weight = float(exp_spec["weight"])
            for metric, metric_spec in exp_spec["metrics"].items():
                coef = abs(float(metric_spec["coefficient"]))
                if coef == 0.0:
                    continue
                ref_value = ref[exp]["results"][method][metric]
                try:
                    value = run[exp]["results"][method][metric]
                except (KeyError, TypeError):
                    missing.append(f"{exp}/{method}/{metric}")
                    continue
                if not finite(value):
                    missing.append(f"{exp}/{method}/{metric} (non-finite: {value!r})")
                    continue
                sign = 1.0 if metric_spec["higher_is_better"] else -1.0
                delta = sign * (float(value) - float(ref_value))
                total += weight * coef * delta
                detail[f"{exp}/{method}/{metric}"] = {"run": value, "ref": ref_value, "delta": delta}
        score_by_method[method] = total

    ao = {}
    for exp in spec:
        row = {}
        for metric in ("team_accuracy", "reliance_probability", "routing_accuracy"):
            key = f"{exp}/adaptive_oracle/{metric}"
            if key in detail:
                row[metric] = detail[key]["run"]
                row["d_" + metric] = detail[key]["delta"]
        ao[exp] = row
        try:
            ao[exp]["n_examples"] = run[exp]["_metadata"]["n_examples"]
        except (KeyError, TypeError):
            pass

    out.update(
        {
            "status": "scored" if args.rc == 0 else "scores file present although evaluate_test.sh failed",
            "score_adaptive_oracle": score_by_method.get("adaptive_oracle"),
            "score_proposed_mean": sum(score_by_method[m] for m in PROPOSED) / len(PROPOSED),
            "score_all_methods_mean": sum(score_by_method.values()) / len(score_by_method),
            "score_by_method": score_by_method,
            "adaptive_oracle_by_experiment": ao,
            "n_missing": len(missing),
            "missing": missing,
            "experiments_in_file": sorted(run.keys()),
            "methods_in_file": {e: sorted((v or {}).get("results", {}).keys()) for e, v in run.items()},
            "detail": detail,
        }
    )
    (run_dir / "rank.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"[rank] {run_dir.name}: rc={args.rc} score(adaptive_oracle)={out['score_adaptive_oracle']:+.4f} "
        f"score(mean 5 methods)={out['score_all_methods_mean']:+.4f} missing={len(missing)}"
    )


if __name__ == "__main__":
    main()
