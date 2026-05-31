"""Evaluate College Admission predictions produced by a method run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_COLUMNS = {
    "standard_ai": "standard_ai",
    "behavior_aware_ai": "behavior_aware_ai",
    "aligned_ai": "aligned_ai",
    "complementary_ai": "complementary_ai",
    "adaptive_oracle": "adaptive_oracle",
    "adaptive_rrs": "adaptive_rrs",
}


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def evaluate_predictions(prediction_path: Path) -> dict:
    df = pd.read_csv(prediction_path)
    y = df["label"].to_numpy(dtype=int)
    human = df["human_pred"].to_numpy(dtype=int)
    region = df["region"].to_numpy(dtype=int)

    results = {}
    for method_name, prefix in METHOD_COLUMNS.items():
        ai_pred = df[f"{prefix}_pred"].to_numpy(dtype=int)
        team_pred = df[f"{prefix}_team_pred"].to_numpy(dtype=int)
        team_acc = accuracy(y, team_pred)
        reliance = float(df[f"{prefix}_reliance"].iloc[0])
        results[method_name] = {
            "team_accuracy": team_acc,
            "reliance_probability": reliance,
        }

    if "adaptive_rrs_route_aligned" in df.columns:
        results["adaptive_rrs"]["routing_accuracy"] = accuracy(
            region, df["adaptive_rrs_route_aligned"].to_numpy(dtype=int)
        )
    if "adaptive_oracle_route_aligned" in df.columns:
        results["adaptive_oracle"]["routing_accuracy"] = accuracy(
            region, df["adaptive_oracle_route_aligned"].to_numpy(dtype=int)
        )

    return {
        "experiments": {
            "college_admission_core": {
                "results": results,
            }
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="checkpoints/college_admission/predictions.csv")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    scores = evaluate_predictions(Path(args.predictions))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
