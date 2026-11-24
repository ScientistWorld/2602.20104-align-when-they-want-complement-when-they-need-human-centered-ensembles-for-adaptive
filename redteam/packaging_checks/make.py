"""Packaging checks: what the hidden test does with honest outputs it was not
written for.

new_method_name_only  the reference predictions under a new method name
                      (a from-scratch agent is never told the five names)
extra_better_column   reference columns + a strictly better method under a
                      new name: is the new method scored at all?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import rtlib  # noqa: E402


def main(mode: str) -> None:
    for exp in rtlib.EXPERIMENTS:
        _, test, meta = rtlib.load_split(exp)
        ref = rtlib.reference_predictions(exp)
        if mode == "new_method_name_only":
            out = ref[["example_id"]].copy()
            out["my_method_prob"] = ref["adaptive_oracle_prob"]
            out["my_method_pred"] = ref["adaptive_oracle_pred"]
            out["my_method_route_aligned"] = ref["adaptive_oracle_route_aligned"]
        elif mode == "extra_better_column":
            out = ref.copy()
            ordered = test.set_index("example_id").loc[out["example_id"]].reset_index()
            out["improved_method_pred"] = rtlib.generator_rule(ordered, meta)
            out["improved_method_route_aligned"] = ordered["region"].to_numpy(dtype=int)
        else:
            raise SystemExit(f"unknown mode {mode}")
        rtlib.write_predictions(exp, out)


if __name__ == "__main__":
    main(sys.argv[1])
