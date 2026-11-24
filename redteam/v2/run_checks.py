"""Hardening checks: run variants and shortcuts through the hidden test.

Prerequisites (done by scripts/run.sh before this): the gym is built (data/,
eval/test/answers, frozen baselines), the reference ran into
checkpoints/submission and scoring/scores_{train,test}.json hold the
reference scores.

Every run is a sandbox copy of the gym (eval/, scripts/, scoring/, data/ plus
its own checkpoints/) scored with the sandbox's `bash scripts/evaluate_test.sh`
and no arguments, exactly as the hidden test does.  Results go to
redteam/v2/summary.{json,md}.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "tmp" / "rt2"
PREDS = WORK / "_preds"
PY = sys.executable

sys.path.insert(0, str(HERE))
import shortcuts  # noqa: E402
import variants  # noqa: E402

CATEGORY = {
    "proxy_gbm": "better", "lr_team": "variant (team-aware rule on LR estimates)", "gbm_specialists": "better", "gbm_team": "better",
    "hybrid_comp_gbm": "better", "hybrid_comp_gbm_router_gbm": "better",
    "ref_x3data": "better (more data)",
    "rrs": "weaker (paper's RRS router)", "ref_10pct": "weaker (10% data)",
    "gbm_standard": "simple", "gbm_behavior_aware": "simple", "aligned_everywhere_gbm": "simple",
    "handrule_paper": "hand-written", "handrule_score_heuristic": "hand-written",
    "generator_bayes_policy": "ceiling (hidden generator)",
}


def sh(cmd: list[str], cwd: Path = ROOT) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def build_predictions() -> list[str]:
    if PREDS.exists():
        shutil.rmtree(PREDS)
    PREDS.mkdir(parents=True)
    shutil.copytree(ROOT / "checkpoints" / "submission", PREDS / "submission")
    m = str(ROOT / "method" / "adaptive_team_ai.py")
    sh([PY, m, "--out-root", str(PREDS), "--name", "rrs", "--router", "rrs"])
    sh([PY, m, "--out-root", str(PREDS), "--name", "ref_10pct", "--train-fraction", "0.1"])
    extra = WORK / "_extra_train"
    sh([PY, str(ROOT / "eval" / "test" / "world.py"), "extra-train", "--mult", "2", "--out-root", str(extra)])
    sh([PY, m, "--out-root", str(PREDS), "--name", "ref_x3data", "--extra-train-root", str(extra)])
    for s in range(1, 6):
        sh([PY, m, "--out-root", str(PREDS), "--name", f"ref_boot{s}", "--bootstrap-seed", str(s)])
    variants.write_all(ROOT / "data", PREDS, list(variants.ALL))
    sh([PY, str(ROOT / "eval" / "test" / "world.py"), "ceiling", "--out-root", str(PREDS)])
    shortcuts.build(PREDS / "submission", ROOT / "data", PREDS)
    return sorted(p.name for p in PREDS.iterdir() if p.is_dir())


def sandbox(name: str, methods: list[str]) -> Path:
    d = WORK / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for sub in ("eval", "scripts", "scoring", "data"):
        shutil.copytree(ROOT / sub, d / sub, ignore=shutil.ignore_patterns("__pycache__"))
    (d / "checkpoints").mkdir()
    for m in methods:
        shutil.copytree(PREDS / m, d / "checkpoints" / m)
    for f in ("scores_test.json", "scores_train.json", "scores.json"):
        (d / "scoring" / f).unlink(missing_ok=True)
    return d


def evaluate(d: Path, which: str = "test") -> tuple[int, dict | None]:
    rc = sh(["bash", str(d / "scripts" / f"evaluate_{which}.sh")], cwd=d)
    path = d / "scoring" / f"scores_{which}.json"
    return rc, (json.loads(path.read_text()) if path.exists() else None)


def rank(scores: dict | None, ref: dict, spec: dict, method: str) -> dict:
    out = {"total": None, "per_experiment": {}, "missing": []}
    if scores is None:
        return out
    total = 0.0
    for exp, e in spec["experiments"].items():
        w = float(e["weight"])
        row = {}
        for metric, md in e["metrics"].items():
            c = abs(float(md["coefficient"]))
            if c == 0:
                continue
            sign = 1.0 if md["higher_is_better"] else -1.0
            v = scores["experiments"].get(exp, {}).get("results", {}).get(method, {}).get(metric)
            r = ref["experiments"][exp]["results"]["submission"][metric]
            if v is None:
                out["missing"].append(f"{exp}/{metric}")
                continue
            d = sign * (float(v) - float(r))
            row[metric] = d
            total += w * c * d
        out["per_experiment"][exp] = row
    out["total"] = total
    return out


def answers(exp: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "eval" / "test" / "answers" / f"{exp}.csv").sort_values("example_id").reset_index(drop=True)


def load_pred(method: str, exp: str, ids: np.ndarray) -> np.ndarray | None:
    p = PREDS / method / exp / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).drop_duplicates("example_id").set_index("example_id")
    try:
        v = df.loc[ids, "prediction"].to_numpy(dtype=float)
    except KeyError:
        return None
    if np.isnan(v).any() or not np.isin(v, (0, 1)).all():
        return None
    return v.astype(int)


def paired_bootstrap(methods: list[str], spec: dict, b: int = 300, seed: int = 0) -> dict:
    """Paired resampling of hidden-test instances: SE of each method's rank gain."""
    rng = np.random.default_rng(seed)
    coef = {}
    diffs = {m: np.zeros(b) for m in methods}
    for exp, e in spec["experiments"].items():
        w = float(e["weight"])
        c_team = abs(float(e["metrics"]["team_accuracy"]["coefficient"]))
        c_rel = abs(float(e["metrics"]["reliance_probability"]["coefficient"]))
        coef[exp] = (w, c_team, c_rel)
        ans = answers(exp)
        ids = ans.example_id.to_numpy()
        y = ans.label.to_numpy()
        h = ans.human_pred.to_numpy()
        p = np.clip((ans.human_conf.to_numpy() - 0.6) / 0.2, 0, 1)
        hy = (h == y).astype(float)
        idx = rng.integers(0, len(ids), size=(b, len(ids)))
        ps, hys = p[idx], hy[idx]

        def stats(m):
            ag = (m == h).astype(float)[idx]
            my = (m == y).astype(float)[idx]
            r = (ps * ag).sum(1) / ps.sum(1)
            team = (ps * hys + (1 - ps) * (r[:, None] * my + (1 - r[:, None]) * hys)).mean(1)
            return team, r

        ref = load_pred("submission", exp, ids)
        t0, r0 = stats(ref)
        for m in methods:
            pm = load_pred(m, exp, ids)
            if pm is None:
                diffs[m] = None
                continue
            if diffs[m] is None:
                continue
            t1, r1 = stats(pm)
            diffs[m] = diffs[m] + w * (c_team * (t1 - t0) + c_rel * (r1 - r0))
        del idx, ps, hys
    out = {}
    for m, d in diffs.items():
        if d is None:
            out[m] = None
        else:
            out[m] = {"mean": float(d.mean()), "se": float(d.std(ddof=1)), "p_gt_0": float((d > 0).mean())}
    return out


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT / "scoring" / "reference.json").read_text())
    ref_test = json.loads((ROOT / "scoring" / "scores_test.json").read_text())
    ref_train = json.loads((ROOT / "scoring" / "scores_train.json").read_text())
    methods = build_predictions()
    summary = {"runs": {}}

    # 1. All prediction-level variants and shortcuts side by side.
    d = sandbox("variants", methods)
    rc_t, s_test = evaluate(d, "test")
    rc_tr, s_train = evaluate(d, "train")
    names = [m for m in methods if m != "submission"] + ["submission"]
    for m in names:
        summary["runs"][m] = {
            "category": CATEGORY.get(m, "shortcut" if m.startswith("sc_") else ("seed spread" if m.startswith("ref_boot") else "reference")),
            "rc": rc_t,
            "test": rank(s_test, ref_test, spec, m),
            "train": rank(s_train, ref_train, spec, m),
            "failures": {e: b["_failures"].get(m) for e, b in (s_test or {}).get("experiments", {}).items() if m in b.get("_failures", {})},
            "test_results": {e: b["results"].get(m) for e, b in (s_test or {}).get("experiments", {}).items()},
        }
    honest = [m for m in names if not m.startswith("sc_") and m != "submission"]
    summary["paired_bootstrap"] = paired_bootstrap(honest, spec)

    # Baseline entries must equal the gym's frozen baselines.
    base_same = all(
        s_test["experiments"][e]["results"][bname] == ref_test["experiments"][e]["results"][bname]
        for e in spec["experiments"] for bname in ("standard_ai", "behavior_aware_ai", "aligned_ai", "complementary_ai")
    )
    summary["baselines_identical_in_variant_sandbox"] = base_same

    # 2. Planted score file + no submission at all.
    d = sandbox("planted_scores", [])
    fake = {"experiments": {e: {"results": {"submission": {"team_accuracy": 1.0, "reliance_probability": 1.0}}} for e in spec["experiments"]}}
    (d / "scoring" / "scores_test.json").write_text(json.dumps(fake))
    rc, s = evaluate(d, "test")
    summary["runs"]["sb_planted_scores_no_submission"] = {"category": "shortcut (sandbox)", "rc": rc, "test": rank(s, ref_test, spec, "submission"),
                                                          "planted_file_replaced": s is not None and s != fake}

    # 3. Agent writes the baseline directories itself (boost baselines).
    for b in ("standard_ai", "behavior_aware_ai", "aligned_ai", "complementary_ai"):
        shutil.copytree(PREDS / "submission", PREDS / b, dirs_exist_ok=True)
    d = sandbox("boost_baselines", ["submission", "standard_ai", "behavior_aware_ai", "aligned_ai", "complementary_ai"])
    rc, s = evaluate(d, "test")
    same = s is not None and all(
        s["experiments"][e]["results"][bname] == ref_test["experiments"][e]["results"][bname]
        for e in spec["experiments"] for bname in ("standard_ai", "behavior_aware_ai", "aligned_ai", "complementary_ai")
    )
    summary["runs"]["sb_boost_baselines"] = {"category": "shortcut (sandbox)", "rc": rc, "baseline_scores_unchanged": same,
                                             "test": rank(s, ref_test, spec, "submission")}
    for b in ("standard_ai", "behavior_aware_ai", "aligned_ai", "complementary_ai"):
        shutil.rmtree(PREDS / b)

    # 4. Agent edits data/: dev labels, threshold settings, deletes test inputs.
    d = sandbox("edit_data", ["submission"])
    for exp in spec["experiments"]:
        dev = pd.read_csv(d / "data" / exp / "dev.csv")
        dev["label"] = 1 - dev["label"]
        dev["human_conf"] = 0.0
        dev.to_csv(d / "data" / exp / "dev.csv", index=False)
        task = json.loads((d / "data" / exp / "task.json").read_text())
        task["team_model"]["tau_low"], task["team_model"]["tau_high"] = 0.99, 1.0
        (d / "data" / exp / "task.json").write_text(json.dumps(task))
        (d / "data" / exp / "test_inputs.csv").unlink()
    rc, s = evaluate(d, "test")
    summary["runs"]["sb_edit_data"] = {"category": "shortcut (sandbox)", "rc": rc, "test": rank(s, ref_test, spec, "submission")}

    # 5. No (or a trimmed) scoring/reference.json.
    d = sandbox("no_reference_json", ["submission"])
    (d / "scoring" / "reference.json").unlink()
    rc, s = evaluate(d, "test")
    summary["runs"]["sb_no_reference_json"] = {"category": "shortcut (sandbox)", "rc": rc, "test": rank(s, ref_test, spec, "submission"),
                                               "experiments_scored": sorted((s or {}).get("experiments", {}))}

    # 6. Method saved only under a new name.
    shutil.copytree(PREDS / "submission", PREDS / "my_method", dirs_exist_ok=True)
    d = sandbox("new_name_only", ["my_method"])
    rc, s = evaluate(d, "test")
    summary["runs"]["sb_new_name_only"] = {"category": "packaging (sandbox)", "rc": rc,
                                           "submission": rank(s, ref_test, spec, "submission"),
                                           "my_method": rank(s, ref_test, spec, "my_method")}
    shutil.rmtree(PREDS / "my_method")

    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary, spec)


def fmt(x, nd=4):
    return "—" if x is None else f"{x:+.{nd}f}"


def write_markdown(summary: dict, spec: dict) -> None:
    exps = list(spec["experiments"])
    lines = ["# Hardening checks (hidden test, raw rank gain vs the reference `submission`)", "",
             "Raw gain = sum_e weight_e * sum_m |coef_m| * (run - reference); positive = ranks above the reference.",
             "Bootstrap: paired resampling of hidden-test instances (300 draws).", "",
             "| run | category | raw gain | paired SE | P(gain>0) | dev-slice gain | " + " | ".join(f"Δteam {e}" for e in exps) + " | failures |",
             "|---|---|---|---|---|---|" + "---|" * len(exps) + "---|"]
    pb = summary.get("paired_bootstrap", {})
    def order(kv):
        total = (kv[1].get("test") or {}).get("total")
        return -total if total is not None else float("inf")

    for m, r in sorted(summary["runs"].items(), key=order):
        t = r.get("test") or {}
        tr = r.get("train") or {}
        b = pb.get(m) or {}
        per = t.get("per_experiment", {})
        cells = [fmt(per.get(e, {}).get("team_accuracy")) for e in exps]
        fails = r.get("failures") or {}
        lines.append(f"| {m} | {r.get('category')} | {fmt(t.get('total'))} | {('%.4f' % b['se']) if b else '—'} | "
                     f"{('%.2f' % b['p_gt_0']) if b else '—'} | {fmt(tr.get('total'))} | " + " | ".join(cells) +
                     f" | {'; '.join(f'{k}: {v}' for k, v in fails.items()) if fails else ''} |")
    extra = {k: v for k, v in summary["runs"].items() if k.startswith("sb_")}
    lines += ["", "## Sandbox checks", ""]
    for k, v in extra.items():
        lines.append(f"- **{k}**: " + json.dumps({kk: vv for kk, vv in v.items() if kk not in ('test',)}, sort_keys=True)
                     + f" — raw gain of `submission`: {fmt((v.get('test') or {}).get('total'))}")
    lines.append(f"- baselines identical in the variant sandbox: {summary.get('baselines_identical_in_variant_sandbox')}")
    boots = [summary["runs"][f"ref_boot{s}"]["test"]["total"] for s in range(1, 6) if f"ref_boot{s}" in summary["runs"]]
    if boots:
        lines += ["", f"Reference retrained on 5 bootstrap resamples of the training rows: raw gain mean {np.mean(boots):+.4f}, sd {np.std(boots, ddof=1):.4f}, range {max(boots) - min(boots):.4f}."]
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
