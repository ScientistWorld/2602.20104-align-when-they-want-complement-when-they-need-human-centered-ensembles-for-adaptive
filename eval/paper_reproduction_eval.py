"""Unscored paper-reproduction check on the paper's own College Admission generator.

Applies the authors' released protocol (tau = 0.5, reliance measured on the
privileged group, one CGPR coin-flip draw with seed 0) to predictions from
method/college_admission_method.py, so the numbers can be compared with the
paper's released results kept in scoring/reference.json["_paper_reported"].
Not part of the ranked gym (the hidden test lives in eval/test/).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_COLUMNS = {
    "behavior_aware_ai": "behavior_aware_ai",
    "aligned_ai": "aligned_ai",
    "complementary_ai": "complementary_ai",
    "adaptive_oracle": "adaptive_oracle",
    "adaptive_weighted_oracle": "adaptive_weighted_oracle",
}


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def reliance_from_alignment(human_pred, ai_pred, region) -> float:
    mask = region == 1
    if not np.any(mask):
        return 0.0
    return float(1.0 - np.mean(human_pred[mask] != ai_pred[mask]))


def cgpr_team_decision(human_pred, human_conf, ai_pred, reliance, tau: float, seed: int):
    high_conf = human_conf > tau
    rng = np.random.Generator(np.random.PCG64(seed))
    accepts_ai = rng.random(len(human_pred)) < reliance
    return np.where(high_conf, human_pred, np.where(accepts_ai, ai_pred, human_pred)).astype(int)


def evaluate_predictions(prediction_path: Path, data_dir: Path, experiment_name: str) -> dict:
    predictions = pd.read_csv(prediction_path)
    test = pd.read_csv(data_dir / "test.csv")
    metadata = pd.read_csv(data_dir / "metadata.csv").iloc[0].to_dict()

    df = test.merge(predictions, on="example_id", validate="one_to_one")
    y = df["label"].to_numpy(dtype=int)
    human = df["human_pred"].to_numpy(dtype=int)
    human_conf = df["human_conf"].to_numpy(dtype=float)
    region = df["region"].to_numpy(dtype=int)
    tau = float(metadata.get("threshold", 0.5))
    team_seed = int(metadata.get("team_seed", 0))

    results = {}
    for method_name, prefix in METHOD_COLUMNS.items():
        pred_column = f"{prefix}_pred"
        if pred_column not in df.columns:
            continue
        ai_pred = df[f"{prefix}_pred"].to_numpy(dtype=int)
        reliance = reliance_from_alignment(human, ai_pred, region)
        team_pred = cgpr_team_decision(human, human_conf, ai_pred, reliance, tau=tau, seed=team_seed)
        team_acc = accuracy(y, team_pred)
        results[method_name] = {
            "team_accuracy": team_acc,
            "reliance_probability": reliance,
        }

    if "adaptive_weighted_oracle_route_aligned" in df.columns and "adaptive_weighted_oracle" in results:
        results["adaptive_weighted_oracle"]["routing_accuracy"] = accuracy(
            region, df["adaptive_weighted_oracle_route_aligned"].to_numpy(dtype=int)
        )
    if "adaptive_oracle_route_aligned" in df.columns:
        results["adaptive_oracle"]["routing_accuracy"] = accuracy(
            region, df["adaptive_oracle_route_aligned"].to_numpy(dtype=int)
        )
    return {
        "experiments": {
            experiment_name: {
                "results": results,
            }
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="checkpoints/college_admission/predictions.csv")
    parser.add_argument("--data-dir", default="data/college_admission")
    parser.add_argument("--output", default="scoring/scores.json")
    parser.add_argument("--experiment-name", default="college_admission_core")
    args = parser.parse_args()

    scores = evaluate_predictions(Path(args.predictions), Path(args.data_dir), args.experiment_name)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
