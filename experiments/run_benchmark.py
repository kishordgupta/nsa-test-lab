"""Run NSA benchmark experiments and write a Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from experiments.datasets import dataset_names, load_datasets
from experiments.registry import model_names, model_specs


RESULT_FIELDS = [
    "dataset",
    "model",
    "family",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "fit_seconds",
    "score_seconds",
    "train_size",
    "test_size",
    "n_features",
    "contamination",
    "error",
]


def _metric(value: float | None) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), 6)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def _run_one(dataset, spec, detectors: int, optimization_iter: int, seed: int) -> Dict[str, object]:
    model_seed = zlib.crc32(f"{dataset.name}:{spec.name}:{seed}".encode("utf-8")) % (2**31 - 1)
    model = spec.build(dataset.contamination, detectors, optimization_iter, model_seed)
    started = time.perf_counter()
    model.fit(dataset.X_train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scores = model.decision_function(dataset.X_test)
    labels = model.predict(dataset.X_test)
    score_seconds = time.perf_counter() - started

    return {
        "dataset": dataset.name,
        "model": spec.name,
        "family": spec.family,
        "roc_auc": _metric(_safe_auc(dataset.y_test, scores)),
        "precision": _metric(precision_score(dataset.y_test, labels, zero_division=0)),
        "recall": _metric(recall_score(dataset.y_test, labels, zero_division=0)),
        "f1": _metric(f1_score(dataset.y_test, labels, zero_division=0)),
        "fit_seconds": _metric(fit_seconds),
        "score_seconds": _metric(score_seconds),
        "train_size": int(dataset.X_train.shape[0]),
        "test_size": int(dataset.X_test.shape[0]),
        "n_features": int(dataset.X_train.shape[1]),
        "contamination": _metric(dataset.contamination),
        "error": "",
    }


def run_benchmark(
    dataset_filter: Iterable[str],
    model_filter: Iterable[str],
    detectors: int,
    optimization_iter: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Run all requested benchmark combinations."""
    datasets = load_datasets(dataset_filter, seed)
    specs = model_specs(model_filter)
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for spec in specs:
            try:
                rows.append(_run_one(dataset, spec, detectors, optimization_iter, seed))
            except Exception as exc:  # pragma: no cover - retained for long benchmark robustness.
                rows.append(
                    {
                        "dataset": dataset.name,
                        "model": spec.name,
                        "family": spec.family,
                        "roc_auc": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "fit_seconds": None,
                        "score_seconds": None,
                        "train_size": int(dataset.X_train.shape[0]),
                        "test_size": int(dataset.X_test.shape[0]),
                        "n_features": int(dataset.X_train.shape[1]),
                        "contamination": _metric(dataset.contamination),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    """Write benchmark rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: List[Dict[str, object]], path: Path) -> None:
    """Write benchmark rows as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def _average_by(rows: List[Dict[str, object]], key: str, metric: str) -> List[Dict[str, object]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    labels: Dict[str, str] = {}
    for row in rows:
        value = row.get(metric)
        if value is None or value == "" or isinstance(value, float) and math.isnan(value):
            continue
        grouped[str(row[key])].append(float(value))
        if key == "model":
            labels[str(row[key])] = str(row["family"])
    summary = []
    for name, values in grouped.items():
        item = {key: name, metric: mean(values), "n": len(values)}
        if key == "model":
            item["family"] = labels.get(name, "")
        summary.append(item)
    return sorted(summary, key=lambda item: item[metric], reverse=True)


def _best_by_dataset(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    best = {}
    for row in rows:
        auc = row.get("roc_auc")
        if auc is None:
            continue
        current = best.get(row["dataset"])
        if current is None or float(auc) > float(current["roc_auc"]):
            best[row["dataset"]] = row
    return [best[name] for name in sorted(best)]


def write_report(
    rows: List[Dict[str, object]],
    path: Path,
    detectors: int,
    optimization_iter: int,
    seed: int,
) -> None:
    """Write a concise Markdown experimental report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    successful = [row for row in rows if not row.get("error")]
    failed = [row for row in rows if row.get("error")]
    model_auc = _average_by(successful, "model", "roc_auc")
    model_f1 = {row["model"]: row["f1"] for row in _average_by(successful, "model", "f1")}
    dataset_best = _best_by_dataset(successful)
    dataset_auc = _average_by(successful, "dataset", "roc_auc")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# NSA Experimental Report",
        "",
        f"Generated: {generated}",
        "",
        "## Configuration",
        "",
        f"- Datasets: {len(set(row['dataset'] for row in rows))}",
        f"- NSA variants: {len(set(row['model'] for row in rows))}",
        f"- Experiment runs: {len(rows)}",
        f"- Detectors per model: {detectors}",
        f"- Optimization iterations: {optimization_iter}",
        f"- Seed: {seed}",
        f"- Successful runs: {len(successful)}",
        f"- Failed runs: {len(failed)}",
        "",
        "## Top Models By Mean ROC AUC",
        "",
        "| Rank | Model | Family | Mean ROC AUC | Mean F1 | Runs |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(model_auc[:12], start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['family']} | {row['roc_auc']:.4f} | "
            f"{model_f1.get(row['model'], 0.0):.4f} | {row['n']} |"
        )

    lines.extend(
        [
            "",
            "## Best Model Per Dataset",
            "",
            "| Dataset | Best Model | ROC AUC | Precision | Recall | F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dataset_best:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {float(row['roc_auc']):.4f} | "
            f"{float(row['precision']):.4f} | {float(row['recall']):.4f} | {float(row['f1']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Dataset Difficulty By Mean ROC AUC",
            "",
            "| Dataset | Mean ROC AUC | Runs |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in sorted(dataset_auc, key=lambda item: item["roc_auc"]):
        lines.append(f"| {row['dataset']} | {row['roc_auc']:.4f} | {row['n']} |")

    if failed:
        lines.extend(
            [
                "",
                "## Failed Runs",
                "",
                "| Dataset | Model | Error |",
                "| --- | --- | --- |",
            ]
        )
        for row in failed:
            lines.append(f"| {row['dataset']} | {row['model']} | `{row['error']}` |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The benchmark is synthetic and deterministic. Results are useful for comparing "
            "relative NSA behavior across geometric data shapes, but they should not be "
            "treated as a substitute for domain-specific validation. NSA detectors are "
            "distance- and coverage-driven, so separated low-dimensional anomalies tend "
            "to favor detector families such as V-detector, density-enhanced NSA, and "
            "optimization-based variants. High-dimensional sparse problems are more "
            "sensitive to feature scaling and detector coverage.",
            "",
            "Full per-run results are available in `reports/results.csv` and "
            "`reports/results.json`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all", help=f"Comma-separated dataset names or all. Available: {', '.join(dataset_names())}")
    parser.add_argument("--models", default="all", help=f"Comma-separated model names or all. Available: {', '.join(model_names())}")
    parser.add_argument("--detectors", type=int, default=32, help="Number of detectors per NSA model.")
    parser.add_argument("--optimization-iter", type=int, default=3, help="Iterations for optimization-style NSA models.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--csv", default="reports/results.csv", help="CSV output path.")
    parser.add_argument("--json", default="reports/results.json", help="JSON output path.")
    parser.add_argument("--report", default="reports/experimental_report.md", help="Markdown report path.")
    return parser


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    rows = run_benchmark(
        dataset_filter=_split_csv(args.datasets),
        model_filter=_split_csv(args.models),
        detectors=args.detectors,
        optimization_iter=args.optimization_iter,
        seed=args.seed,
    )
    write_csv(rows, Path(args.csv))
    write_json(rows, Path(args.json))
    write_report(rows, Path(args.report), args.detectors, args.optimization_iter, args.seed)
    print(f"Wrote {len(rows)} runs to {args.csv}, {args.json}, and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
