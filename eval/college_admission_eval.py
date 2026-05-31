"""Evaluate College Admission predictions produced by a method run."""

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
    "adaptive_rrs": "adaptive_rrs",
}


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def reliance_from_alignment(human_pred, ai_pred, region) -> float:
    mask = region == 1
    if not np.any(mask):
        return 0.0
    return float(1.0 - np.mean(human_pred[mask] != ai_pred[mask]))


def deterministic_uniform(example_id, seed: int):
    values = np.asarray(example_id, dtype=np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values = values ^ (values >> np.uint64(31))
    return (values >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def cgpr_team_decision(example_id, human_pred, human_conf, ai_pred, reliance, tau: float, seed: int):
    high_conf = human_conf > tau
    accepts_ai = deterministic_uniform(example_id, seed) < reliance
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
    example_id = df["example_id"].to_numpy(dtype=int)
    tau = float(metadata.get("threshold", 0.5))
    seed = int(metadata.get("seed", 42)) + 3

    results = {}
    for method_name, prefix in METHOD_COLUMNS.items():
        pred_column = f"{prefix}_pred"
        if pred_column not in df.columns:
            continue
        ai_pred = df[f"{prefix}_pred"].to_numpy(dtype=int)
        reliance = reliance_from_alignment(human, ai_pred, region)
        team_pred = cgpr_team_decision(example_id, human, human_conf, ai_pred, reliance, tau=tau, seed=seed)
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
    if "adaptive_rrs_route_aligned" in df.columns and "adaptive_rrs" in results:
        results["adaptive_rrs"]["routing_accuracy"] = accuracy(
            region, df["adaptive_rrs_route_aligned"].to_numpy(dtype=int)
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
