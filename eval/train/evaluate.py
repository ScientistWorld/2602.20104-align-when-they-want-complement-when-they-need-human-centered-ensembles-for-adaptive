"""Visible train-slice scorer for College Admission method outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SLICE_NAME = "train"
SPLIT_SALT = "college-admission-eval-split-v1"
DEFAULT_REFERENCE = Path("scoring/reference.json")
DEFAULT_METHOD_SOURCE = Path("scoring/scores.json")


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def in_train_slice(experiment_name: str, example_id: int) -> bool:
    key = f"{SPLIT_SALT}:{experiment_name}:{int(example_id)}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % 2 == 0


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


def experiment_paths(experiment_name: str) -> tuple[Path, Path]:
    if experiment_name == "college_admission_core":
        suffix = "college_admission"
    else:
        suffix = experiment_name
    return Path("data") / suffix, Path("checkpoints") / suffix / "predictions.csv"


def load_reference_experiments(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    return list(payload.get("experiments", {}).keys())


def methods_from_score_file(path: Path, experiment_name: str) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text())
    experiments = payload.get("experiments", payload)
    experiment = experiments.get(experiment_name, {})
    results = experiment.get("results", experiment)
    if not isinstance(results, dict):
        return set()
    return {name for name in results if not name.startswith("_")}


def methods_from_predictions(predictions: pd.DataFrame) -> set[str]:
    methods = set()
    for column in predictions.columns:
        if column.endswith("_pred"):
            methods.add(column[: -len("_pred")])
    return methods


def evaluate_predictions(
    prediction_path: Path,
    data_dir: Path,
    experiment_name: str,
    requested_methods: set[str] | None,
) -> dict:
    predictions = pd.read_csv(prediction_path)
    test = pd.read_csv(data_dir / "test.csv")
    metadata = pd.read_csv(data_dir / "metadata.csv").iloc[0].to_dict()

    df = test.merge(predictions, on="example_id", validate="one_to_one")
    split_mask = df["example_id"].map(lambda value: in_train_slice(experiment_name, int(value)))
    df = df.loc[split_mask].copy()
    if df.empty:
        raise ValueError(f"{experiment_name}: train split is empty")

    available_methods = methods_from_predictions(predictions)
    methods = sorted(requested_methods or available_methods)
    missing = [name for name in methods if name not in available_methods]
    if missing:
        raise ValueError(f"{experiment_name}: missing prediction columns for methods: {missing}")

    y = df["label"].to_numpy(dtype=int)
    human = df["human_pred"].to_numpy(dtype=int)
    human_conf = df["human_conf"].to_numpy(dtype=float)
    region = df["region"].to_numpy(dtype=int)
    tau = float(metadata.get("threshold", 0.5))
    team_seed = int(metadata.get("team_seed", 0))

    results = {}
    for method_name in methods:
        pred_column = f"{method_name}_pred"
        ai_pred = df[pred_column].to_numpy(dtype=int)
        reliance = reliance_from_alignment(human, ai_pred, region)
        team_pred = cgpr_team_decision(human, human_conf, ai_pred, reliance, tau=tau, seed=team_seed)
        method_scores = {
            "team_accuracy": accuracy(y, team_pred),
            "reliance_probability": reliance,
        }
        route_column = f"{method_name}_route_aligned"
        if route_column in df.columns:
            method_scores["routing_accuracy"] = accuracy(region, df[route_column].to_numpy(dtype=int))
        results[method_name] = method_scores

    return {
        "results": results,
        "_metadata": {
            "slice": SLICE_NAME,
            "data_dir": str(data_dir),
            "predictions": str(prediction_path),
            "split_rule": f"sha256({SPLIT_SALT}:experiment_name:example_id) % 2 == 0",
            "n_examples": int(len(df)),
        },
    }


def evaluate_suite(method_source: Path, reference: Path) -> dict:
    scores = {
        "_metadata": {
            "slice": SLICE_NAME,
            "method_source": str(method_source),
            "split_rule": f"sha256({SPLIT_SALT}:experiment_name:example_id) % 2 == 0",
        },
        "experiments": {},
    }
    for experiment_name in load_reference_experiments(reference):
        data_dir, prediction_path = experiment_paths(experiment_name)
        requested = methods_from_score_file(method_source, experiment_name)
        if not requested:
            requested = methods_from_score_file(reference, experiment_name)
        scores["experiments"][experiment_name] = evaluate_predictions(
            prediction_path=prediction_path,
            data_dir=data_dir,
            experiment_name=experiment_name,
            requested_methods=requested,
        )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions")
    parser.add_argument("--data-dir")
    parser.add_argument("--output", default="scoring/scores_train.json")
    parser.add_argument("--experiment-name")
    parser.add_argument("--method-source", default=str(DEFAULT_METHOD_SOURCE))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    args = parser.parse_args()

    output = Path(args.output)
    if args.predictions:
        experiment_name = args.experiment_name or "college_admission_core"
        data_dir = Path(args.data_dir or "data/college_admission")
        method_source = Path(args.method_source)
        requested = methods_from_score_file(method_source, experiment_name)
        if not requested:
            requested = methods_from_score_file(Path(args.reference), experiment_name)
        scores = {
            "_metadata": {"slice": SLICE_NAME, "method_source": str(method_source)},
            "experiments": {
                experiment_name: evaluate_predictions(
                    prediction_path=Path(args.predictions),
                    data_dir=data_dir,
                    experiment_name=experiment_name,
                    requested_methods=requested,
                )
            },
        }
    else:
        scores = evaluate_suite(method_source=Path(args.method_source), reference=Path(args.reference))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
