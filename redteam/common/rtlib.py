"""Method-side helpers for the red-team runs.

Everything here uses only what a future agent is given: data/ (including the
generator data/college_admission.py), the visible scorer eval/train/evaluate.py
(split salt, CGPR rule) and, in improve mode, method/.  The gym evaluator never
imports this file.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

WS = Path("/home/user")
PRISTINE = WS / "tmp" / "redteam_pristine"

# experiment name -> (data/checkpoint directory, alpha)
EXPERIMENTS = {
    "college_admission_core": ("college_admission", 0.9),
    "college_admission_alpha_0_5": ("college_admission_alpha_0_5", 0.5),
    "college_admission_alpha_0_7": ("college_admission_alpha_0_7", 0.7),
    "college_admission_alpha_0_8": ("college_admission_alpha_0_8", 0.8),
    "college_admission_alpha_1_0": ("college_admission_alpha_1_0", 1.0),
}
BASELINES = ["behavior_aware_ai", "aligned_ai", "complementary_ai"]
PROPOSED = ["adaptive_oracle", "adaptive_weighted_oracle"]
REF_METHODS = BASELINES + PROPOSED
FEATURES = ["gpa", "score"]
SPLIT_SALT = "college-admission-eval-split-v1"  # printed in eval/train/evaluate.py


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generator():
    """The gym's own data generator (data/college_admission.py)."""
    return load_module("rt_generator", WS / "data" / "college_admission.py")


def data_dir(exp: str) -> Path:
    return WS / "data" / EXPERIMENTS[exp][0]


def ckpt_path(exp: str) -> Path:
    return WS / "checkpoints" / EXPERIMENTS[exp][0] / "predictions.csv"


def load_split(exp: str):
    d = data_dir(exp)
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    meta = pd.read_csv(d / "metadata.csv").iloc[0].to_dict()
    return train, test, meta


def reference_predictions(exp: str) -> pd.DataFrame:
    return pd.read_csv(PRISTINE / "checkpoints" / EXPERIMENTS[exp][0] / "predictions.csv")


def write_predictions(exp: str, frame: pd.DataFrame) -> None:
    path = ckpt_path(exp)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def in_test_slice(exp: str, ids) -> np.ndarray:
    """Hidden-test membership, rebuilt from the rule printed in eval/train/evaluate.py."""
    out = []
    for value in ids:
        digest = hashlib.sha256(f"{SPLIT_SALT}:{exp}:{int(value)}".encode("utf-8")).digest()
        out.append(int.from_bytes(digest[:8], "big") % 2 == 1)
    return np.asarray(out, dtype=bool)


def cgpr_team(human, conf, ai, region, tau: float, seed: int):
    """The CGPR rule exactly as printed in eval/train/evaluate.py."""
    mask = region == 1
    reliance = float(1.0 - np.mean(human[mask] != ai[mask])) if mask.any() else 0.0
    rng = np.random.Generator(np.random.PCG64(seed))
    accepts = rng.random(len(human)) < reliance
    team = np.where(conf > tau, human, np.where(accepts, ai, human)).astype(int)
    return team, reliance


def generator_rule(test: pd.DataFrame, meta: dict) -> np.ndarray:
    """Label rule of data/college_admission.py applied to (gpa, score, region)."""
    delta = float(meta.get("delta", 0.25))
    threshold = float(meta.get("threshold", 0.5))
    gpa = test["gpa"].to_numpy(dtype=float)
    score = test["score"].to_numpy(dtype=float)
    aligned_value = (0.5 + delta) * score + (0.5 - delta) * gpa
    complementary_value = (0.5 + delta) * gpa + (0.5 - delta) * score
    region = test["region"].to_numpy(dtype=int)
    return np.where(region == 1, aligned_value >= threshold, complementary_value >= threshold).astype(int)


def fit(x, y, w=None, C: float = 1.0, seed: int = 42) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed, C=C)
    model.fit(x, y, sample_weight=w)
    return model


def extra_training_rows(meta: dict, n_rows: int, seed: int) -> pd.DataFrame:
    """Fresh training rows from the gym's generator with this experiment's settings."""
    gen = generator()
    rng = np.random.default_rng(seed)
    frame = gen.generate_college_admission(
        n_instances=int(n_rows),
        aligned_fraction=float(meta.get("aligned_fraction", 0.5)),
        delta=float(meta.get("delta", 0.25)),
        threshold=float(meta.get("threshold", 0.5)),
        rng=rng,
    )
    frame = gen.add_human_behavior_with_rng(
        frame,
        alpha=float(meta["alpha"]),
        complementary_accuracy=float(meta.get("complementary_accuracy", 0.5)),
        rng=rng,
    )
    frame["example_id"] = frame["example_id"] + 50_000_000 + 1_000_000 * int(seed)
    return frame


def train_models(train: pd.DataFrame, test: pd.DataFrame, C: float = 1.0, seed: int = 42,
                 aligned_fraction: float = 0.5) -> pd.DataFrame:
    """Same models as method/college_admission_method.py (C exposed); all prediction columns."""
    x = train[FEATURES].to_numpy()
    y = train["label"].to_numpy(dtype=int)
    g = train["region"].to_numpy(dtype=int)
    h = train["human_pred"].to_numpy(dtype=int)
    c = train["human_conf"].to_numpy(dtype=float)
    xt = test[FEATURES].to_numpy()
    gt = test["region"].to_numpy(dtype=int)

    target = np.where(g == 1, h, y)
    n = len(y)
    n_a = max(int(np.sum(g == 1)), 1)
    n_c = max(int(np.sum(g == 0)), 1)
    weights = np.where(g == 1, aligned_fraction / n_a, (1.0 - aligned_fraction) / n_c) * n
    single = fit(x, target, weights, C, seed)
    aligned = fit(x[g == 1], h[g == 1], None, C, seed)
    complementary = fit(x[g == 0], y[g == 0], None, C, seed)
    weighted_aligned = fit(x, h, c, C, seed)
    weighted_complementary = fit(x, y, 1.0 - c, C, seed)

    out = pd.DataFrame({"example_id": test["example_id"].to_numpy(dtype=int)})
    for name, model in {"behavior_aware_ai": single, "aligned_ai": aligned,
                        "complementary_ai": complementary}.items():
        prob = model.predict_proba(xt)[:, 1]
        out[f"{name}_prob"] = prob
        out[f"{name}_pred"] = (prob >= 0.5).astype(int)

    pa = aligned.predict_proba(xt)[:, 1]
    pc = complementary.predict_proba(xt)[:, 1]
    po = np.where(gt == 1, pa, pc)
    out["adaptive_oracle_prob"] = po
    out["adaptive_oracle_pred"] = (po >= 0.5).astype(int)
    out["adaptive_oracle_route_aligned"] = gt

    pwa = weighted_aligned.predict_proba(xt)[:, 1]
    pwc = weighted_complementary.predict_proba(xt)[:, 1]
    pw = np.where(gt == 1, pwa, pwc)
    out["adaptive_weighted_oracle_prob"] = pw
    out["adaptive_weighted_oracle_pred"] = (pw >= 0.5).astype(int)
    out["adaptive_weighted_oracle_route_aligned"] = gt

    # Rational Routing Shortcut over the weighted specialists (as in the method).
    use_w = np.maximum(pwa, 1 - pwa) >= np.maximum(pwc, 1 - pwc)
    pr = np.where(use_w, pwa, pwc)
    out["adaptive_rrs_prob"] = pr
    out["adaptive_rrs_pred"] = (pr >= 0.5).astype(int)
    out["adaptive_rrs_route_aligned"] = use_w.astype(int)

    # RRS over the unweighted specialists (used by the weaker_rrs variant).
    use_u = np.maximum(pa, 1 - pa) >= np.maximum(pc, 1 - pc)
    pu = np.where(use_u, pa, pc)
    out["rrs_unweighted_prob"] = pu
    out["rrs_unweighted_pred"] = (pu >= 0.5).astype(int)
    out["rrs_unweighted_route_aligned"] = use_u.astype(int)
    return out
