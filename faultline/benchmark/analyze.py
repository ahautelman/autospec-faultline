"""Aggregate benchmark records into the RQ1 tables and figures.

    python -m faultline.benchmark analyze --out reports/faultline_v1

Reads ``<out>/*.json`` (evaluation records), ``<out>/screen/*.json`` and ``funnel.json`` and writes
``<out>/analysis/``: ``summary.md`` (every table, ready to paste into the results), ``runs.csv``
(one row per instance x seed x method) and the figures ``f1_by_family`` and ``operator_confusion``.

Definitions (per run):

* ``proxy_success``: the repaired graph's best-path success clears the threshold ``p``. This
  is what AutoSpec optimises.
* ``f1``: harmonic mean of fault removal and collateral retention, averaged over the
  instance's region loci (goal inflation, dead start, hazard unlabeled). Structural families
  (waypoint contraction, route pruning) have no region locus and are scored by GED and the
  trajectory audit instead.
* ``operator_match``: the first committed operator on an edge touching the fault locus is one
  of the family's expected operators (``EXPECTED_OPERATORS``).
* ``false_repair`` (nulls only): some refinement was committed on an edge that is not weak by
  design.

Uncertainty: rates get exact Clopper-Pearson 95% intervals over runs; means get percentile
bootstrap intervals over instances (seeds are averaged within an instance first, so the
resampling unit is the independent one). AutoSpec vs random operator is a paired Wilcoxon
signed-rank test on instance-mean F1 per family, Holm-corrected across families.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from faultline.faults.ground_truth import EXPECTED_OPERATORS, FaultKind
from faultline.metrics.inference import bootstrap_ci, holm, wilcoxon_signed_rank
from faultline.stats import two_sided

METHODS = ("autospec", "random_operator")
BASELINES = ("unrefined", "oracle", "trivializer")
FAMILY_NAMES = {
    "goal_inflation": "Goal inflation", "hazard_unlabeled": "Hazard unlabeled", "dead_start": "Dead start",
    "waypoint_contraction": "Waypoint contraction", "route_pruning": "Route pruning", "null": "Null (no fault)",
}
METHOD_NAMES = {"autospec": "AutoSpec", "random_operator": "Random operator", "unrefined": "Unrefined",
                "oracle": "Oracle", "trivializer": "Trivializer"}


def _mean_f1(evaluation: dict) -> float | None:
    values = [locus["f1"] for locus in evaluation["loci"].values() if locus.get("f1") is not None]
    return float(np.mean(values)) if values else None


def _touches(edge: str, locus: list) -> bool:
    kind, name = locus
    if kind == "edge":
        return edge == name
    return kind == "vertex" and name in edge.split("->")


def run_rows(record: dict, threshold: float) -> list[dict[str, Any]]:
    """Flatten one record into one row per (seed, method) plus one per graph baseline."""
    base = {k: record[k] for k in ("id", "gold", "kind", "level", "severity")}
    base["stratum"] = record.get("stratum", record["gate"]["stage"])
    expected = EXPECTED_OPERATORS[FaultKind(record["kind"])]
    weak = set(record.get("weak_edges", []))
    rows = []
    for run in record.get("runs", []):
        for method in METHODS:
            result, evaluation = run[method], run[f"{method}_eval"]
            committed = [s for s in result["steps"] if s["operator"] is not None]
            on_locus = [s for s in committed if _touches(s["edge"], record["locus"])]
            rows.append({
                **base, "seed": run["seed"], "method": method, "seconds": run["seconds"],
                "best_path_success": result["best_path_success"],
                "proxy_success": result["best_path_success"] >= threshold,
                "f1": _mean_f1(evaluation), "ged": evaluation["ged"],
                "gold_kept": evaluation["audit"]["gold_kept"], "fault_rejected": evaluation["audit"]["fault_rejected"],
                "n_refinements": len(committed),
                "first_operator": committed[0]["operator"] if committed else None,
                "locus_operator": on_locus[0]["operator"] if on_locus else None,
                "operator_match": (on_locus[0]["operator"] in expected) if on_locus and expected else None,
                "false_repair": (any(s["edge"] not in weak for s in committed)
                                 if record["kind"] == FaultKind.NULL.value else None),
            })
    for name, evaluation in record.get("baselines", {}).items():
        rows.append({**base, "seed": None, "method": name, "f1": _mean_f1(evaluation), "ged": evaluation["ged"],
                     "gold_kept": evaluation["audit"]["gold_kept"],
                     "fault_rejected": evaluation["audit"]["fault_rejected"]})
    return rows


def _rate(values: list[bool]) -> str:
    if not values:
        return "–"
    k, n = sum(values), len(values)
    low, high = two_sided(k, n, 0.95)
    return f"{k}/{n} = {k / n:.2f} [{low:.2f}, {high:.2f}]"


def _instance_means(rows: list[dict], metric: str) -> dict[str, float]:
    by_instance: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            by_instance[row["id"]].append(float(row[metric]))
    return {k: float(np.mean(v)) for k, v in by_instance.items()}


def _mean_ci(rows: list[dict], metric: str, rng: np.random.Generator, resamples: int) -> tuple[float, float, float, int] | None:
    means = list(_instance_means(rows, metric).values())
    if not means:
        return None
    if len(means) == 1:
        return means[0], math.nan, math.nan, 1
    low, high = bootstrap_ci(means, rng, resamples=resamples)
    return float(np.mean(means)), low, high, len(means)


def _fmt_ci(value) -> str:
    if value is None:
        return "–"
    mean, low, high, n = value
    return f"{mean:.2f} (n={n})" if math.isnan(low) else f"{mean:.2f} [{low:.2f}, {high:.2f}] (n={n})"


def analyze(out: Path, threshold: float = 0.8, resamples: int = 10_000) -> dict[str, Any]:
    """Write the analysis of every record under ``out`` and return the summary numbers."""
    records = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))
               if p.name not in {"certificates.json", "funnel.json", "screen_funnel.json"}]
    records = [r for r in records if "gate" in r]
    rows = [row for record in records for row in run_rows(record, threshold)]
    target = out / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = sorted({k for row in rows for k in row})
        with (target / "runs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    rng = np.random.default_rng(0)
    families = [k for k in FAMILY_NAMES if any(r["kind"] == k for r in rows)]
    lines = ["# FaultLine benchmark: preliminary results", "",
             f"Records: {len(records)} instances, {sum(1 for r in rows if r['method'] == 'autospec')} AutoSpec runs. "
             f"Threshold p = {threshold}.", ""]
    lines += _funnel_section(out)

    lines += ["## Recovery per family", "",
              "| Family | Method | Proxy success | F1 (removal x retention) | GED to gold | Operator match |",
              "|---|---|---|---|---|---|"]
    summary: dict[str, Any] = {}
    for family in families:
        for method in (*METHODS, *BASELINES):
            subset = [r for r in rows if r["kind"] == family and r["method"] == method]
            if not subset:
                continue
            f1 = _mean_ci(subset, "f1", rng, resamples)
            ged = _mean_ci(subset, "ged", rng, resamples)
            proxy = [r["proxy_success"] for r in subset if r.get("proxy_success") is not None]
            match = [r["operator_match"] for r in subset if r.get("operator_match") is not None]
            summary[(family, method)] = {"f1": f1, "ged": ged}
            lines.append(f"| {FAMILY_NAMES[family]} | {METHOD_NAMES[method]} | {_rate(proxy)} | {_fmt_ci(f1)} | "
                         f"{_fmt_ci(ged)} | {_rate(match)} |")
    lines += ["", "Proxy success: Clopper-Pearson 95% over runs. F1 and GED: mean over instances with a 95% "
              "bootstrap interval over instances. Operator match: first committed operator on a locus edge is the "
              "family's expected operator.", ""]

    lines += _paired_tests(rows, families)
    lines += _confusion_section(rows, families)
    nulls = [r["false_repair"] for r in rows if r["kind"] == "null" and r["method"] == "autospec"]
    lines += ["## Nulls (gold itself)", "", f"False-repair rate (AutoSpec): {_rate(nulls)}", ""]
    seconds = [r["seconds"] for r in rows if r["method"] == "autospec" and r.get("seconds")]
    if seconds:
        lines += ["## Runtime", "", f"Seconds per evaluation seed (AutoSpec + random operator): median "
                  f"{np.median(seconds):.0f}, max {np.max(seconds):.0f}, n = {len(seconds)}.", ""]
    (target / "summary.md").write_text("\n".join(lines))
    _figures(rows, families, rng, resamples, target)
    print("\n".join(lines))
    return summary


def _funnel_section(out: Path) -> list[str]:
    lines = []
    path = out / "screen_funnel.json"
    if path.exists():
        counts = json.loads(path.read_text())
        stages = sorted({s for c in counts.values() for s in c})
        lines += ["## Screening funnel (gates on the unrefined faulty graph)", "",
                  "| Family | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
        for kind, c in counts.items():
            lines.append(f"| {FAMILY_NAMES.get(kind, kind)} | " + " | ".join(str(c.get(s, 0)) for s in stages) + " |")
        lines.append("")
    return lines


def _paired_tests(rows: list[dict], families: list[str]) -> list[str]:
    p_values, ns = {}, {}
    for family in families:
        a = _instance_means([r for r in rows if r["kind"] == family and r["method"] == "autospec"], "f1")
        b = _instance_means([r for r in rows if r["kind"] == family and r["method"] == "random_operator"], "f1")
        common = sorted(set(a) & set(b))
        if len(common) >= 2 and any(a[k] != b[k] for k in common):
            p_values[family] = wilcoxon_signed_rank([a[k] for k in common], [b[k] for k in common])
            ns[family] = len(common)
    if not p_values:
        return ["## AutoSpec vs random operator", "", "Not enough paired instances with differing F1 for a test.", ""]
    adjusted = holm(p_values)
    lines = ["## AutoSpec vs random operator (paired Wilcoxon on instance-mean F1, Holm)", "",
             "| Family | n | p | p (Holm) |", "|---|---|---|---|"]
    lines += [f"| {FAMILY_NAMES[f]} | {ns[f]} | {p_values[f]:.3f} | {adjusted[f]:.3f} |" for f in p_values]
    return lines + [""]


def _confusion(rows: list[dict], families: list[str]) -> tuple[list[str], list[str], np.ndarray]:
    operators = ["seq", "add", "past", "or", "none"]
    fams = [f for f in families if f != "null"]
    matrix = np.zeros((len(fams), len(operators)), dtype=int)
    for r in rows:
        if r["method"] == "autospec" and r["kind"] in fams:
            matrix[fams.index(r["kind"]), operators.index(r["locus_operator"] or "none")] += 1
    return fams, operators, matrix


def _confusion_section(rows: list[dict], families: list[str]) -> list[str]:
    fams, operators, matrix = _confusion(rows, families)
    if not matrix.any():
        return []
    lines = ["## Operator chosen on the fault locus (AutoSpec runs)", "",
             "| Family (expected) | " + " | ".join(operators) + " |", "|---|" + "---|" * len(operators)]
    for f, counts in zip(fams, matrix):
        expected = ",".join(sorted(EXPECTED_OPERATORS[FaultKind(f)]))
        lines.append(f"| {FAMILY_NAMES[f]} ({expected}) | " + " | ".join(str(c) for c in counts) + " |")
    return lines + [""]


# Okabe-Ito hues as in faultline.viz.style; each method also has its own marker, so identity
# never rests on colour alone. Unrefined is a hollow grey reference marker.
_METHOD_STYLE = {
    "unrefined": ("#7f7f7f", "o", True), "autospec": ("#0072B2", "D", False),
    "random_operator": ("#E69F00", "s", False), "oracle": ("#009E73", "^", False),
    "trivializer": ("#CC79A7", "v", False),
}


def _figures(rows, families, rng, resamples, target: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from faultline.viz.style import GRID_COLOUR, TEXT_COLOUR

    region_families = [f for f in families if any(r["kind"] == f and r.get("f1") is not None for r in rows)]
    if region_families:
        fig, ax = plt.subplots(figsize=(7.0, 0.6 + 0.55 * len(region_families) * 1.6))
        methods = ("unrefined", "autospec", "random_operator", "oracle", "trivializer")
        offsets = np.linspace(-0.3, 0.3, len(methods))
        for offset, method in zip(offsets, methods):
            colour, marker, hollow = _METHOD_STYLE[method]
            for y, family in enumerate(region_families):
                ci = _mean_ci([r for r in rows if r["kind"] == family and r["method"] == method], "f1", rng, resamples)
                if ci is None:
                    continue
                mean, low, high, _ = ci
                if not math.isnan(low):
                    ax.plot([low, high], [y + offset] * 2, color=colour, linewidth=2, solid_capstyle="round")
                ax.plot(mean, y + offset, marker=marker, markersize=8, color=colour,
                        markerfacecolor="white" if hollow else colour, markeredgewidth=1.5, linestyle="none",
                        label=METHOD_NAMES[method] if y == 0 else None)
        ax.set_yticks(range(len(region_families)), [FAMILY_NAMES[f] for f in region_families], color=TEXT_COLOUR)
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("F1 of fault removal and collateral retention (mean over instances, 95% bootstrap CI)",
                      color=TEXT_COLOUR)
        ax.grid(axis="x", color=GRID_COLOUR, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(target / f"f1_by_family.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fams, operators, matrix = _confusion(rows, families)
    if matrix.any():
        fig, ax = plt.subplots(figsize=(4.6, 0.8 + 0.5 * len(fams)))
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()), aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                dark = matrix[i, j] > 0.6 * matrix.max()
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if dark else TEXT_COLOUR)
        expected = [",".join(sorted(EXPECTED_OPERATORS[FaultKind(f)])) for f in fams]
        ax.set_yticks(range(len(fams)), [f"{FAMILY_NAMES[f]} ({e})" for f, e in zip(fams, expected)], fontsize=8)
        ax.set_xticks(range(len(operators)), operators, fontsize=8)
        ax.set_xlabel("Operator AutoSpec committed on the fault locus (runs)")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(target / f"operator_confusion.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
