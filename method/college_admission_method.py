"""Paper method for the synthetic College Admission experiments.

This implements the computational steps described in the paper:
single behavior-aware logistic model, aligned and complementary
specialists, oracle adaptive routing, Rational Routing Shortcut (RRS),
and CGPR team decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


FEATURE_COLUMNS = ["gpa", "score"]


def fit_logistic(x, y, sample_weight=None, seed: int = 42) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
    model.fit(x, y, sample_weight=sample_weight)
    return model


def train_single_ai(x, y, region, aligned_weight, human_pred=None, seed: int = 42):
    """Train the paper's weighted single-model surrogate g_p(theta)."""
    target = y.copy() if human_pred is None else np.where(region == 1, human_pred, y)
    n = len(y)
    n_a = max(int(np.sum(region == 1)), 1)
    n_c = max(int(np.sum(region == 0)), 1)
    sample_weight = np.where(region == 1, aligned_weight / n_a, (1.0 - aligned_weight) / n_c)
    sample_weight = sample_weight * n
    return fit_logistic(x, target, sample_weight=sample_weight, seed=seed)


def simulate_human(y, region, alpha, complementary_accuracy, seed: int):
    rng = np.random.default_rng(seed)
    p_correct = np.where(region == 1, alpha, complementary_accuracy)
    correct = rng.random(len(y)) < p_correct
    human_pred = np.where(correct, y, 1 - y).astype(int)

    low = np.clip(p_correct - 0.1, 0.0, 1.0)
    high = np.clip(p_correct + 0.1, 0.0, 1.0)
    human_conf = rng.uniform(low, high)
    return human_pred, human_conf


def hard_predictions(model, x):
    prob = model.predict_proba(x)[:, 1]
    return prob, (prob >= 0.5).astype(int)


def route_by_rrs(model_aligned, model_complementary, x):
    prob_a = model_aligned.predict_proba(x)[:, 1]
    prob_c = model_complementary.predict_proba(x)[:, 1]
    conf_a = np.maximum(prob_a, 1.0 - prob_a)
    conf_c = np.maximum(prob_c, 1.0 - prob_c)
    use_aligned = conf_a >= conf_c
    prob = np.where(use_aligned, prob_a, prob_c)
    return prob, (prob >= 0.5).astype(int), use_aligned.astype(int), conf_a, conf_c


def cgpr_team_decision(human_pred, human_conf, ai_pred, reliance, tau=0.5, seed=0):
    rng = np.random.default_rng(seed)
    high_conf = human_conf > tau
    accepts_ai = rng.random(len(human_pred)) < reliance
    return np.where(high_conf, human_pred, np.where(accepts_ai, ai_pred, human_pred)).astype(int)


def reliance_from_alignment(human_pred, ai_pred, region):
    mask = region == 1
    if not np.any(mask):
        return 0.0
    return float(1.0 - np.mean(human_pred[mask] != ai_pred[mask]))


def write_method_outputs(
    data_dir: Path,
    output_dir: Path,
    aligned_fraction: float,
    alpha: float,
    complementary_accuracy: float,
    tau: float,
    seed: int,
) -> None:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = train[FEATURE_COLUMNS].to_numpy()
    y_train = train["label"].to_numpy(dtype=int)
    region_train = train["region"].to_numpy(dtype=int)
    x_test = test[FEATURE_COLUMNS].to_numpy()
    y_test = test["label"].to_numpy(dtype=int)
    region_test = test["region"].to_numpy(dtype=int)

    human_train, conf_train = simulate_human(
        y_train, region_train, alpha=alpha, complementary_accuracy=complementary_accuracy, seed=seed + 1
    )
    human_test, conf_test = simulate_human(
        y_test, region_test, alpha=alpha, complementary_accuracy=complementary_accuracy, seed=seed + 2
    )

    single = train_single_ai(
        x_train,
        y_train,
        region_train,
        aligned_weight=aligned_fraction,
        human_pred=human_train,
        seed=seed,
    )
    standard = fit_logistic(x_train, y_train, seed=seed)
    aligned = fit_logistic(x_train[region_train == 1], human_train[region_train == 1], seed=seed)
    complementary = fit_logistic(x_train[region_train == 0], y_train[region_train == 0], seed=seed)

    weighted_aligned = fit_logistic(x_train, human_train, sample_weight=conf_train, seed=seed)
    weighted_complementary = fit_logistic(x_train, y_train, sample_weight=1.0 - conf_train, seed=seed)

    rows = pd.DataFrame(
        {
            "example_id": test["example_id"].to_numpy(dtype=int),
            "label": y_test,
            "region": region_test,
            "human_pred": human_test,
            "human_conf": conf_test,
        }
    )

    model_preds = {}
    for name, model in {
        "standard_ai": standard,
        "behavior_aware_ai": single,
        "aligned_ai": aligned,
        "complementary_ai": complementary,
    }.items():
        prob, pred = hard_predictions(model, x_test)
        rows[f"{name}_prob"] = prob
        rows[f"{name}_pred"] = pred
        model_preds[name] = pred

    oracle_prob = np.where(
        region_test == 1,
        aligned.predict_proba(x_test)[:, 1],
        complementary.predict_proba(x_test)[:, 1],
    )
    rows["adaptive_oracle_prob"] = oracle_prob
    rows["adaptive_oracle_pred"] = (oracle_prob >= 0.5).astype(int)
    rows["adaptive_oracle_route_aligned"] = region_test
    model_preds["adaptive_oracle"] = rows["adaptive_oracle_pred"].to_numpy(dtype=int)

    rrs_prob, rrs_pred, rrs_route_choice, conf_a, conf_c = route_by_rrs(
        weighted_aligned, weighted_complementary, x_test
    )
    rows["adaptive_rrs_prob"] = rrs_prob
    rows["adaptive_rrs_pred"] = rrs_pred
    rows["adaptive_rrs_route_aligned"] = rrs_route_choice
    rows["aligned_confidence"] = conf_a
    rows["complementary_confidence"] = conf_c
    model_preds["adaptive_rrs"] = rrs_pred

    for name, pred in model_preds.items():
        reliance = reliance_from_alignment(human_test, pred, region_test)
        rows[f"{name}_reliance"] = reliance
        team = cgpr_team_decision(human_test, conf_test, pred, reliance, tau=tau, seed=seed + 3)
        rows[f"{name}_team_pred"] = team

    rows.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "aligned_fraction": aligned_fraction,
                "alpha": alpha,
                "complementary_accuracy": complementary_accuracy,
                "tau": tau,
                "seed": seed,
            }
        ]
    ).to_csv(output_dir / "run_metadata.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/college_admission")
    parser.add_argument("--output-dir", default="checkpoints/college_admission")
    parser.add_argument("--aligned-fraction", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--complementary-accuracy", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_method_outputs(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        aligned_fraction=args.aligned_fraction,
        alpha=args.alpha,
        complementary_accuracy=args.complementary_accuracy,
        tau=args.tau,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
