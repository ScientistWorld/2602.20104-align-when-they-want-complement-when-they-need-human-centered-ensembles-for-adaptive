"""Synthetic College Admission data used by the adaptive AI paper.

The problem is binary admission prediction from two features:
GPA and standardized test score.  Aligned-region examples use score as
the stronger signal; complementary-region examples use GPA as the
stronger signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["gpa", "score"]


def generate_college_admission(
    n_instances: int,
    aligned_fraction: float,
    delta: float,
    threshold: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.choice([1, 0], size=n_instances, p=[aligned_fraction, 1.0 - aligned_fraction])
    features = rng.uniform(0.0, 1.0, size=(n_instances, 2))

    gpa = features[:, 0]
    score = features[:, 1]
    aligned_value = (0.5 + delta) * score + (0.5 - delta) * gpa
    complementary_value = (0.5 + delta) * gpa + (0.5 - delta) * score
    label = np.where(group == 1, aligned_value >= threshold, complementary_value >= threshold)

    return pd.DataFrame(
        {
            "example_id": np.arange(n_instances, dtype=int),
            "gpa": gpa,
            "score": score,
            "region": group.astype(int),
            "label": label.astype(int),
        }
    )


def add_human_behavior(
    df: pd.DataFrame,
    alpha: float,
    complementary_accuracy: float,
    seed: int,
) -> pd.DataFrame:
    """Attach the paper's simulated human decisions to a split."""
    rng = np.random.default_rng(seed)
    output = df.copy()
    label = output["label"].to_numpy(dtype=int)
    region = output["region"].to_numpy(dtype=int)

    p_correct = np.where(region == 1, alpha, complementary_accuracy)
    correct = rng.random(len(output)) < p_correct
    output["human_pred"] = np.where(correct, label, 1 - label).astype(int)

    low = np.clip(p_correct - 0.1, 0.0, 1.0)
    high = np.clip(p_correct + 0.1, 0.0, 1.0)
    output["human_conf"] = rng.uniform(low, high)
    return output


def split_and_write(
    output_dir: Path,
    n_instances: int = 10000,
    aligned_fraction: float = 0.5,
    delta: float = 0.25,
    threshold: float = 0.5,
    alpha: float = 0.9,
    complementary_accuracy: float = 0.5,
    test_size: float = 0.3,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_college_admission(
        n_instances=n_instances,
        aligned_fraction=aligned_fraction,
        delta=delta,
        threshold=threshold,
        seed=seed,
    )
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["region"],
        random_state=seed,
    )
    train = train.sort_values("example_id").reset_index(drop=True)
    test = test.sort_values("example_id").reset_index(drop=True)
    train = add_human_behavior(train, alpha=alpha, complementary_accuracy=complementary_accuracy, seed=seed + 1)
    test = add_human_behavior(test, alpha=alpha, complementary_accuracy=complementary_accuracy, seed=seed + 2)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "n_instances": n_instances,
                "aligned_fraction": aligned_fraction,
                "delta": delta,
                "threshold": threshold,
                "alpha": alpha,
                "complementary_accuracy": complementary_accuracy,
                "test_size": test_size,
                "seed": seed,
            }
        ]
    )
    metadata.to_csv(output_dir / "metadata.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/college_admission")
    parser.add_argument("--n-instances", type=int, default=10000)
    parser.add_argument("--aligned-fraction", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--complementary-accuracy", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_and_write(
        output_dir=Path(args.output_dir),
        n_instances=args.n_instances,
        aligned_fraction=args.aligned_fraction,
        delta=args.delta,
        threshold=args.threshold,
        alpha=args.alpha,
        complementary_accuracy=args.complementary_accuracy,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
