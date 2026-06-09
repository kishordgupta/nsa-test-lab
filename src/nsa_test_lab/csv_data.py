"""CSV loading helpers for the NSA GUI."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


DEFAULT_ANOMALY_VALUES = ("1", "true", "yes", "anomaly", "outlier", "attack")


@dataclass(frozen=True)
class CsvDataset:
    """Numeric features and optional labels loaded from a CSV file."""

    path: Path
    headers: Sequence[str]
    feature_columns: Sequence[str]
    skipped_columns: Sequence[str]
    X: np.ndarray
    y: Optional[np.ndarray]
    row_count: int


def read_csv_header(path: str | Path) -> List[str]:
    """Read the header row from a CSV file."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV file is empty") from exc
    header = [name.strip() for name in header]
    if not header or any(not name for name in header):
        raise ValueError("CSV file must have a non-empty header row")
    return header


def load_csv_dataset(
    path: str | Path,
    label_column: str = "",
    anomaly_values: Iterable[str] = DEFAULT_ANOMALY_VALUES,
) -> CsvDataset:
    """Load numeric feature columns and optional binary labels from CSV."""
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    if not headers:
        raise ValueError("CSV file must have a header row")
    if not rows:
        raise ValueError("CSV file contains no data rows")
    if label_column and label_column not in headers:
        raise ValueError(f"Label column not found: {label_column}")

    feature_columns: List[str] = []
    skipped_columns: List[str] = []
    numeric_data: List[List[float]] = []
    for column in headers:
        if column == label_column:
            continue
        values = []
        try:
            for row in rows:
                raw = (row.get(column) or "").strip()
                if raw == "":
                    raise ValueError
                values.append(float(raw))
        except ValueError:
            skipped_columns.append(column)
            continue
        feature_columns.append(column)
        numeric_data.append(values)

    if not numeric_data:
        raise ValueError("CSV file does not contain usable numeric feature columns")

    X = np.asarray(numeric_data, dtype=float).T
    y = None
    if label_column:
        anomaly_set = {value.strip().lower() for value in anomaly_values if value.strip()}
        y = np.asarray(
            [1 if (row.get(label_column) or "").strip().lower() in anomaly_set else 0 for row in rows],
            dtype=int,
        )

    return CsvDataset(
        path=csv_path,
        headers=headers,
        feature_columns=feature_columns,
        skipped_columns=skipped_columns,
        X=X,
        y=y,
        row_count=len(rows),
    )


def write_prediction_csv(
    path: str | Path,
    scores: np.ndarray,
    predictions: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> None:
    """Write row-level NSA scores and predictions."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["row_index", "anomaly_score", "predicted_label"]
        if labels is not None:
            fieldnames.append("true_label")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, score in enumerate(scores):
            row = {
                "row_index": index,
                "anomaly_score": float(score),
                "predicted_label": int(predictions[index]),
            }
            if labels is not None:
                row["true_label"] = int(labels[index])
            writer.writerow(row)
