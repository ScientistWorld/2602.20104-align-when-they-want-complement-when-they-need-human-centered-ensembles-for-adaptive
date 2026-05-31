"""Synthetic College Admission data used by the adaptive AI paper.

The problem is binary admission prediction from two features:
GPA and standardized test score.  Aligned-region examples use score as
the stronger signal; complementary-region examples use GPA as the
stronger signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

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
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if rng is None:
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
    return add_human_behavior_with_rng(
        df,
        alpha=alpha,
        complementary_accuracy=complementary_accuracy,
        rng=np.random.default_rng(seed),
    )


def add_human_behavior_with_rng(
    df: pd.DataFrame,
    alpha: float,
    complementary_accuracy: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Attach simulated human decisions using an existing RNG stream."""
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


def write_split(
    output_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float,
    complementary_accuracy: float,
    rng: np.random.Generator,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_h = add_human_behavior_with_rng(
        train, alpha=alpha, complementary_accuracy=complementary_accuracy, rng=rng
    )
    test_h = add_human_behavior_with_rng(
        test, alpha=alpha, complementary_accuracy=complementary_accuracy, rng=rng
    )
    train_h.to_csv(output_dir / "train.csv", index=False)
    test_h.to_csv(output_dir / "test.csv", index=False)

    row = dict(metadata)
    row.update(
        {
            "alpha": alpha,
            "complementary_accuracy": complementary_accuracy,
            "team_seed": 0,
        }
    )
    pd.DataFrame([row]).to_csv(output_dir / "metadata.csv", index=False)


def alpha_dir_name(alpha: float) -> str:
    if abs(alpha - 0.9) < 1e-12:
        return "college_admission"
    alpha_token = f"{alpha:g}".replace(".", "_")
    return f"college_admission_alpha_{alpha_token}"


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
    rng = np.random.default_rng(seed)
    df = generate_college_admission(
        n_instances=n_instances,
        aligned_fraction=aligned_fraction,
        delta=delta,
        threshold=threshold,
        rng=rng,
    )
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
    )
    metadata = {
        "n_instances": n_instances,
        "aligned_fraction": aligned_fraction,
        "delta": delta,
        "threshold": threshold,
        "test_size": test_size,
        "seed": seed,
    }
    write_split(
        output_dir=output_dir,
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        alpha=alpha,
        complementary_accuracy=complementary_accuracy,
        rng=rng,
        metadata=metadata,
    )


def write_alpha_sweep(
    output_root: Path,
    alphas: Iterable[float],
    n_instances: int = 10000,
    aligned_fraction: float = 0.5,
    delta: float = 0.25,
    threshold: float = 0.5,
    complementary_accuracy: float = 0.5,
    test_size: float = 0.3,
    seed: int = 42,
) -> None:
    """Write the imperfect-human alpha sweep with the paper's RNG protocol."""
    rng = np.random.default_rng(seed)
    df = generate_college_admission(
        n_instances=n_instances,
        aligned_fraction=aligned_fraction,
        delta=delta,
        threshold=threshold,
        rng=rng,
    )
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    metadata = {
        "n_instances": n_instances,
        "aligned_fraction": aligned_fraction,
        "delta": delta,
        "threshold": threshold,
        "test_size": test_size,
        "seed": seed,
        "sweep_protocol": "paper_imperfect_human",
    }
    for alpha in alphas:
        write_split(
            output_dir=output_root / alpha_dir_name(alpha),
            train=train,
            test=test,
            alpha=alpha,
            complementary_accuracy=complementary_accuracy,
            rng=rng,
            metadata=metadata,
        )


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
    parser.add_argument(
        "--sweep-alphas",
        default=None,
        help="Comma-separated alpha values; writes the paper's alpha sweep protocol.",
    )
    parser.add_argument("--sweep-output-root", default="data")
    args = parser.parse_args()
    if args.sweep_alphas:
        alphas = [float(value) for value in args.sweep_alphas.split(",")]
        write_alpha_sweep(
            output_root=Path(args.sweep_output_root),
            alphas=alphas,
            n_instances=args.n_instances,
            aligned_fraction=args.aligned_fraction,
            delta=args.delta,
            threshold=args.threshold,
            complementary_accuracy=args.complementary_accuracy,
            test_size=args.test_size,
            seed=args.seed,
        )
        return

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
