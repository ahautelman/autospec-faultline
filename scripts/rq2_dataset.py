"""Build the RQ2 table: one row per logged edge, with features and labels.

    python scripts/rq2_dataset.py reports/faultline_v1            # -> reports/faultline_v1/rq2/dataset.csv
    python scripts/rq2_dataset.py reports/faultline_v1 --phases decision

Rows with identical (instance, seed, edge, phase) are written once. Label columns:
``fault_kind``, ``touches_locus``, ``expected_operators``, ``committed_operator``,
``operators_clearing`` and ``score_<operator>`` (``None`` when the operator refused).
A natural first target is ``operators_clearing`` (multi-label) on ``decision`` rows, or
``fault_kind`` on weak edges of ``gate`` rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from faultline.rq2 import edge_features

LABELS = ("instance", "gold", "fault_kind", "level", "severity", "phase", "seed", "edge", "is_weak",
          "touches_locus", "expected_operators", "committed_operator", "operators_clearing")
OPERATORS = ("seq", "add", "past", "or")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--phases", nargs="+", default=["gold", "gate", "initial", "decision"])
    args = parser.parse_args()
    rq2 = args.out / "rq2"
    seen, table = set(), []
    for path in sorted(rq2.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["instance"], row["seed"], row["edge"], row["phase"])
            if row["phase"] not in args.phases or not row["learned"] or key in seen:
                continue
            seen.add(key)
            out = {k: row.get(k) for k in LABELS}
            for name in ("expected_operators", "operators_clearing"):
                out[name] = "|".join(out[name] or [])
            for op in OPERATORS:
                out[f"score_{op}"] = (row.get("operators") or {}).get(op, {}).get("score")
            out.update(edge_features(row, rq2))
            table.append(out)
    if not table:
        raise SystemExit(f"No rows under {rq2}")
    fields = list(dict.fromkeys(k for row in table for k in row))
    target = rq2 / "dataset.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    print(f"{len(table)} rows -> {target}")


if __name__ == "__main__":
    main()
