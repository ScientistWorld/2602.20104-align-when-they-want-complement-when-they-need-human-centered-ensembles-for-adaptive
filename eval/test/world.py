"""Hidden generator for the admissions world scored by the hidden test.

Maintainer-only.  This file can regenerate the hidden test answers, so it
lives in eval/test/ and nothing outside eval/test/ may import it at test
time.  It extends the paper's synthetic College Admission generator:

* two applicant groups whose admission outcome depends on GPA and test score
  with group-specific weights (score-dominant for the privileged group,
  GPA-dominant for the underprivileged group), plus small context effects,
  a non-linear term and label noise;
* group membership is not drawn independently of the features: it follows a
  partly non-linear function of six context features, so a router can
  infer it only approximately;
* human decision makers apply one judgement rule to everybody: the
  privileged-group outcome weighting of GPA and score, blind to the outcome's
  context and non-linear terms, plus a confidently held bias (they favour
  high family income and penalise test retakes, neither of which affects the
  outcome) -- systematic, confident errors -- plus random slips;
  privileged-group humans are confident except on cases that are borderline
  by their own judgement; underprivileged-group humans are unsure
  (confidence U[0.4, 0.6]) and at chance (as in the paper's simulation);
* the decision maker's self-confidence threshold is tau ~ U[0.6, 0.8] per
  decision, as in the paper's behaviour-grounded protocol.

Commands
--------
build        write data/<exp>/{train,dev,test_inputs}.csv, data/<exp>/task.json,
             data/experiments.json and eval/test/answers/<exp>.csv
ceiling      write Bayes-policy predictions computed from the generator's own
             probabilities (the "write down the generating process" bound)
extra-train  write additional training rows from the same world (red-team
             variants that test the effect of more data)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

CONTEXT = [
    "school_resources",
    "family_income",
    "neighborhood_index",
    "prep_hours",
    "test_retakes",
    "activities",
]
FEATURES = ["gpa", "score"] + CONTEXT

TAU_LOW, TAU_HIGH = 0.6, 0.8
N_TRAIN, N_DEV, N_TEST = 7000, 3000, 20000

_BASE = dict(
    # group gate: linear in school_resources, family_income, neighborhood_index
    # plus a small non-linear residual (interaction and square of prep_hours)
    gate_lin=(1.6, 0.9, 0.6, 0.0, 0.0, 0.0),
    gate_inter=0.30,
    gate_sq=0.15,
    gate_scale=6.0,
    # outcome rules: the paper's two weightings plus small context / non-linear terms
    priv_w=(0.64, 0.16),
    under_w=(0.60, 0.20),
    ctx_effect=0.03,
    wiggle=0.025,
    label_noise=0.02,
    # human judgement: the privileged-group outcome weighting, ignoring the
    # outcome's context/non-linear terms, plus a confidently held bias
    # (favours family income, penalises test retakes; neither affects the outcome)
    human_w=(0.64, 0.16),
    human_bias=0.035,
    retake_relu=0.0,
    red_flag=None,
    under_flip=0.5,
    conf_margin=0.06,
)

WORLDS = {
    # Round 3 default: wider (more non-linear) group gate, and an outcome for
    # the underprivileged group (where humans defer) with stronger context,
    # non-linear and interaction terms; the confident humans' rule is unchanged.
    "default": dict(_BASE, gate_inter=0.5, gate_sq=0.3, under_ctx=0.06, under_wiggle=0.06, under_inter=0.05),
    "under_mild": dict(_BASE, gate_inter=0.5, gate_sq=0.3, under_ctx=0.045, under_wiggle=0.045, under_inter=0.03),
    "under_strong": dict(_BASE, gate_inter=0.5, gate_sq=0.3, under_ctx=0.08, under_wiggle=0.08, under_inter=0.07),
    "under_narrow_gate": dict(_BASE, under_ctx=0.06, under_wiggle=0.06, under_inter=0.05),
    # earlier rounds, kept for the record (see HARDENING.md)
    "bias_v2": dict(_BASE),
    "redflag_v1": dict(_BASE, human_w=(0.70, 0.10), human_bias=0.0, red_flag=1.4, conf_margin=0.15),
}

# Human-reliability profiles: target accuracy of confident (privileged-group)
# humans and the range their confidence spans from borderline to clear cases.
RELIABILITY = {
    "high": dict(alpha=0.90, conf_lo=0.66, conf_hi=0.98),
    "low": dict(alpha=0.75, conf_lo=0.55, conf_hi=0.88),
}

EXPERIMENTS = {
    "admission_group_hidden": dict(group_given=False, reliability="high", seed=1101),
    "admission_group_hidden_low_reliability": dict(group_given=False, reliability="low", seed=1202),
    "admission_group_known": dict(group_given=True, reliability="high", seed=1303),
    "admission_group_known_low_reliability": dict(group_given=True, reliability="low", seed=1404),
}

CALIBRATION_SEED = 7
CONF_NOISE = 0.03


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def f_t(conf):
    return np.clip((conf - TAU_LOW) / (TAU_HIGH - TAU_LOW), 0.0, 1.0)


def _latent(gpa, score, ctx, w):
    """Group-specific merit scores, the human heuristic score and its red flag."""
    ctx_term = w["ctx_effect"] * np.tanh(1.5 * ctx[:, 5])
    a_s, a_g = w["priv_w"]
    b_g, b_s = w["under_w"]
    s_priv = a_s * score + a_g * gpa + ctx_term + w["wiggle"] * np.sin(6.0 * gpa) - 0.5 * (a_s + a_g)
    u_ctx = w.get("under_ctx", w["ctx_effect"]) * np.tanh(1.5 * ctx[:, 5])
    u_wig = w.get("under_wiggle", w["wiggle"]) * np.sin(6.0 * score)
    u_int = w.get("under_inter", 0.0) * np.tanh(ctx[:, 3]) * (gpa - 0.5) * 2.0
    s_under = b_g * gpa + b_s * score + u_ctx + u_wig + u_int - 0.5 * (b_g + b_s)
    h_s, h_g = w["human_w"]
    s_human = h_s * score + h_g * gpa - 0.5 * (h_s + h_g)
    s_human = s_human + w["human_bias"] * (ctx[:, 1] - ctx[:, 4])
    s_human = s_human - w["retake_relu"] * np.maximum(ctx[:, 4] - 1.0, 0.0)
    if w["red_flag"] is None:
        red = np.zeros(len(gpa), dtype=bool)
    else:
        red = ctx[:, 4] > w["red_flag"]
    return s_priv, s_under, s_human, red


def _gate(ctx, w):
    f = ctx @ np.asarray(w["gate_lin"], dtype=float)
    f = f + w["gate_inter"] * ctx[:, 2] * ctx[:, 3]
    f = f + w["gate_sq"] * (ctx[:, 3] ** 2 - 1.0)
    return sigmoid(w["gate_scale"] * f)


def draw_population(n: int, rng: np.random.Generator, w: dict) -> dict:
    gpa = rng.uniform(0.0, 1.0, n)
    score = rng.uniform(0.0, 1.0, n)
    ctx = rng.normal(0.0, 1.0, (n, len(CONTEXT)))
    p_priv = _gate(ctx, w)
    group = (rng.random(n) < p_priv).astype(int)
    s_priv, s_under, s_human, red = _latent(gpa, score, ctx, w)
    y_priv = (s_priv >= 0).astype(int)
    y_under = (s_under >= 0).astype(int)
    clean = np.where(group == 1, y_priv, y_under)
    label_flip = rng.random(n) < w["label_noise"]
    label = np.where(label_flip, 1 - clean, clean).astype(int)
    heur = ((s_human >= 0) & ~red).astype(int)
    return dict(
        gpa=gpa, score=score, ctx=ctx, p_priv=p_priv, group=group,
        y_priv=y_priv, y_under=y_under, label=label, heur=heur, s_human=s_human, red=red,
    )


def add_humans(pop: dict, rng: np.random.Generator, w: dict, rel: dict, eps_priv: float) -> dict:
    n = len(pop["label"])
    flip_p = np.where(pop["group"] == 1, eps_priv, w["under_flip"])
    slip = rng.random(n) < flip_p
    human_pred = np.where(slip, 1 - pop["heur"], pop["heur"]).astype(int)
    # Confident humans are sure of red-flag rejections and of clear cases,
    # less sure of cases that are borderline by their own heuristic.
    margin = np.where(pop["red"], 1.0, np.minimum(1.0, np.abs(pop["s_human"]) / w["conf_margin"]))
    center = rel["conf_lo"] + (rel["conf_hi"] - rel["conf_lo"]) * margin
    conf_priv = center + rng.uniform(-CONF_NOISE, CONF_NOISE, n)
    conf_under = rng.uniform(0.40, 0.60, n)
    human_conf = np.clip(np.where(pop["group"] == 1, conf_priv, conf_under), 0.0, 1.0)
    out = dict(pop)
    out.update(human_pred=human_pred, human_conf=human_conf, conf_center=center)
    return out


def calibrate_slip(w: dict, alpha: float) -> tuple[float, float]:
    """Random-slip rate that brings privileged-group human accuracy to alpha."""
    rng = np.random.default_rng(CALIBRATION_SEED)
    pop = draw_population(400_000, rng, w)
    mask = pop["group"] == 1
    acc_heur = float(np.mean(pop["heur"][mask] == pop["label"][mask]))
    eps = max(0.0, (acc_heur - alpha) / (2.0 * acc_heur - 1.0))
    return eps, acc_heur


def generate_experiment(name: str, world: str = "default") -> tuple[pd.DataFrame, dict]:
    spec = EXPERIMENTS[name]
    w = WORLDS[world]
    rel = RELIABILITY[spec["reliability"]]
    eps, acc_heur = calibrate_slip(w, rel["alpha"])
    rng = np.random.default_rng(spec["seed"])
    n_total = N_TRAIN + N_DEV + N_TEST
    pop = add_humans(draw_population(n_total, rng, w), rng, w, rel, eps)
    ids = rng.choice(np.arange(100_000, 1_000_000), size=n_total, replace=False)
    split = np.array(["train"] * N_TRAIN + ["dev"] * N_DEV + ["test"] * N_TEST)
    split = split[rng.permutation(n_total)]
    df = pd.DataFrame({"example_id": ids.astype(np.int64)})
    df["gpa"] = pop["gpa"]
    df["score"] = pop["score"]
    for j, col in enumerate(CONTEXT):
        df[col] = pop["ctx"][:, j]
    df["group"] = pop["group"]
    df["label"] = pop["label"]
    df["human_pred"] = pop["human_pred"]
    df["human_conf"] = pop["human_conf"]
    df["split"] = split
    # Generator-only quantities used by the Bayes-policy ceiling.
    hidden = dict(
        p_priv=pop["p_priv"], y_priv=pop["y_priv"], y_under=pop["y_under"],
        heur=pop["heur"], conf_center=pop["conf_center"],
    )
    meta = dict(
        world=world, eps_priv=eps, heuristic_accuracy_priv=acc_heur,
        alpha_target=rel["alpha"], reliability=spec["reliability"],
        group_given=spec["group_given"], seed=spec["seed"],
    )
    return df, dict(hidden=hidden, meta=meta)


def _round(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURES + ["human_conf"]:
        if col in out.columns:
            out[col] = out[col].round(5)
    return out


def build(world: str, data_root: Path, answers_root: Path) -> dict:
    data_root.mkdir(parents=True, exist_ok=True)
    answers_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, spec in EXPERIMENTS.items():
        df, extra = generate_experiment(name, world)
        df = _round(df)
        cols_in = ["example_id"] + FEATURES + (["group"] if spec["group_given"] else [])
        cols_train = cols_in + ["label", "human_pred", "human_conf"]
        out_dir = data_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        df.loc[df.split == "train", cols_train].to_csv(out_dir / "train.csv", index=False)
        df.loc[df.split == "dev", cols_train].to_csv(out_dir / "dev.csv", index=False)
        df.loc[df.split == "test", cols_in].to_csv(out_dir / "test_inputs.csv", index=False)
        task = {
            "experiment": name,
            "features": FEATURES,
            "group_column_given": bool(spec["group_given"]),
            "files": {
                "train": "train.csv",
                "dev": "dev.csv",
                "test_inputs": "test_inputs.csv",
            },
            "team_model": {
                "confidence_threshold_distribution": "uniform",
                "tau_low": TAU_LOW,
                "tau_high": TAU_HIGH,
            },
        }
        (out_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")
        ans = df.loc[df.split == "test", ["example_id", "label", "human_pred", "human_conf", "group"]]
        ans.to_csv(answers_root / f"{name}.csv", index=False)
        t = df[df.split == "test"]
        priv = t.group == 1
        summary[name] = dict(
            extra["meta"],
            n_train=int((df.split == "train").sum()),
            n_dev=int((df.split == "dev").sum()),
            n_test=int(len(t)),
            privileged_fraction=float(priv.mean()),
            human_acc_priv=float((t.human_pred[priv] == t.label[priv]).mean()),
            human_acc_under=float((t.human_pred[~priv] == t.label[~priv]).mean()),
            mean_p_alignment=float(f_t(t.human_conf.to_numpy()).mean()),
        )
    (data_root / "experiments.json").write_text(json.dumps({"experiments": list(EXPERIMENTS)}, indent=2) + "\n")
    return summary


def _expected_ft_priv(center: np.ndarray) -> np.ndarray:
    grid = np.linspace(-CONF_NOISE, CONF_NOISE, 41)
    vals = f_t(np.clip(center[:, None] + grid[None, :], 0.0, 1.0))
    return vals.mean(axis=1)


def team_decision(a1, a0, b1, b0, c, pi_total, iters: int = 50) -> np.ndarray:
    """Maximise R(m) * A(m) over binary recommendations (expected CGPR team gain).

    a_k: expected alignment-region agreement mass if recommending k,
    b_k: expected complementarity-region correct mass if recommending k,
    c:   expected complementarity-region human-correct mass (constant in m).
    """
    m = (b1 > b0).astype(int)
    for _ in range(iters):
        a = np.where(m == 1, a1, a0)
        b = np.where(m == 1, b1, b0)
        r = a.sum() / max(pi_total, 1e-12)
        adv = (b - c).sum()
        lam = max(adv, 0.0) / max(pi_total, 1e-12)
        new = ((lam * a1 + r * b1) > (lam * a0 + r * b0)).astype(int)
        if np.array_equal(new, m):
            break
        m = new
    return m


def ceiling(world: str, out_root: Path, name: str) -> dict:
    """Bayes policy from the generator's own probabilities (dev + test rows)."""
    w = WORLDS[world]
    out = {}
    for exp, spec in EXPERIMENTS.items():
        df, extra = generate_experiment(exp, world)
        hid = extra["hidden"]
        eps = extra["meta"]["eps_priv"]
        eta = w["label_noise"]
        sel = (df.split != "train").to_numpy()
        p_priv = np.where(df.group == 1, 1.0, 0.0) if spec["group_given"] else hid["p_priv"]
        p_priv = np.asarray(p_priv)[sel]
        pa_priv = _expected_ft_priv(hid["conf_center"][sel])
        heur = hid["heur"][sel]
        yp, yu = hid["y_priv"][sel], hid["y_under"][sel]
        ph1 = heur * (1 - eps) + (1 - heur) * eps
        py1_priv = yp * (1 - eta) + (1 - yp) * eta
        py1_under = yu * (1 - eta) + (1 - yu) * eta
        a1 = p_priv * pa_priv * ph1
        a0 = p_priv * pa_priv * (1 - ph1)
        dc_priv = p_priv * (1 - pa_priv)
        dc_under = 1 - p_priv
        b1 = dc_priv * py1_priv + dc_under * py1_under
        b0 = dc_priv * (1 - py1_priv) + dc_under * (1 - py1_under)

        def p_h_eq_y(h, y, e_h):
            same = (1 - e_h) * (1 - eta) + e_h * eta
            diff = e_h * (1 - eta) + (1 - e_h) * eta
            return np.where(h == y, same, diff)

        c = dc_priv * p_h_eq_y(heur, yp, eps) + dc_under * p_h_eq_y(heur, yu, w["under_flip"])
        pi_total = float((p_priv * pa_priv).sum())
        m = team_decision(a1, a0, b1, b0, c, pi_total)
        pred = pd.DataFrame({"example_id": df.example_id.to_numpy()[sel], "prediction": m})
        d = out_root / name / exp
        d.mkdir(parents=True, exist_ok=True)
        pred.to_csv(d / "predictions.csv", index=False)
        out[exp] = int(len(pred))
    return out


def extra_train(world: str, mult: int, out_root: Path) -> None:
    """Additional training rows from the same world (fresh seeds)."""
    for exp, spec in EXPERIMENTS.items():
        w = WORLDS[world]
        rel = RELIABILITY[spec["reliability"]]
        eps, _ = calibrate_slip(w, rel["alpha"])
        rng = np.random.default_rng(spec["seed"] + 50_000)
        n = N_TRAIN * mult
        pop = add_humans(draw_population(n, rng, w), rng, w, rel, eps)
        df = pd.DataFrame({"example_id": np.arange(5_000_000, 5_000_000 + n)})
        df["gpa"] = pop["gpa"]
        df["score"] = pop["score"]
        for j, col in enumerate(CONTEXT):
            df[col] = pop["ctx"][:, j]
        if spec["group_given"]:
            df["group"] = pop["group"]
        df["label"] = pop["label"]
        df["human_pred"] = pop["human_pred"]
        df["human_conf"] = pop["human_conf"]
        d = out_root / exp
        d.mkdir(parents=True, exist_ok=True)
        _round(df).to_csv(d / "train.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--world", default="default")
    b.add_argument("--data-root", default=str(ROOT / "data"))
    b.add_argument("--answers-root", default=str(HERE / "answers"))
    b.add_argument("--summary", default=None)
    c = sub.add_parser("ceiling")
    c.add_argument("--world", default="default")
    c.add_argument("--out-root", required=True)
    c.add_argument("--name", default="generator_bayes_policy")
    e = sub.add_parser("extra-train")
    e.add_argument("--world", default="default")
    e.add_argument("--mult", type=int, default=2)
    e.add_argument("--out-root", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        summary = build(args.world, Path(args.data_root), Path(args.answers_root))
        text = json.dumps(summary, indent=2)
        print(text)
        if args.summary:
            Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary).write_text(text + "\n")
    elif args.cmd == "ceiling":
        print(json.dumps(ceiling(args.world, Path(args.out_root), args.name), indent=2))
    elif args.cmd == "extra-train":
        extra_train(args.world, args.mult, Path(args.out_root))


if __name__ == "__main__":
    main()
