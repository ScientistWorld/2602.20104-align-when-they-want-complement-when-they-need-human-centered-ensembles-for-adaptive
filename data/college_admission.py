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


def split_and_write(
    output_dir: Path,
    n_instances: int = 10000,
    aligned_fraction: float = 0.5,
    delta: float = 0.25,
    threshold: float = 0.5,
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
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "n_instances": n_instances,
                "aligned_fraction": aligned_fraction,
                "delta": delta,
                "threshold": threshold,
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
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_and_write(
        output_dir=Path(args.output_dir),
        n_instances=args.n_instances,
        aligned_fraction=args.aligned_fraction,
        delta=args.delta,
        threshold=args.threshold,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
