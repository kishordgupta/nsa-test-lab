"""Tkinter GUI for running NSA detectors on CSV files."""

from __future__ import annotations

import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .csv_data import DEFAULT_ANOMALY_VALUES, CsvDataset, load_csv_dataset, read_csv_header, write_prediction_csv
from .factory import build_model, model_names


PARAMETERS = [
    ("contamination", "0.10"),
    ("n_detectors", "64"),
    ("radius", ""),
    ("sampling_margin", "0.50"),
    ("n_grid", "8"),
    ("n_bits", "32"),
    ("r", "4"),
    ("n_clusters", "6"),
    ("n_estimators", "3"),
    ("feature_subsample", "0.70"),
    ("optimization_iter", "3"),
    ("random_state", "42"),
]


@dataclass
class AnalysisResult:
    """Latest GUI analysis output."""

    dataset: CsvDataset
    scores: np.ndarray
    predictions: np.ndarray
    model_name: str
    fit_seconds: float
    score_seconds: float


class NSAGuiApp(tk.Tk):
    """Desktop app for NSA CSV scoring."""

    def __init__(self) -> None:
        super().__init__()
        self.title("NSA Test Lab")
        self.geometry("980x720")
        self.minsize(860, 640)
        self.csv_path = tk.StringVar()
        self.label_column = tk.StringVar()
        self.anomaly_values = tk.StringVar(value=",".join(DEFAULT_ANOMALY_VALUES))
        self.model_name = tk.StringVar(value=model_names()[0])
        self.match_rule = tk.StringVar(value="hamming")
        self.status = tk.StringVar(value="Ready")
        self.parameter_vars: Dict[str, tk.StringVar] = {
            name: tk.StringVar(value=default) for name, default in PARAMETERS
        }
        self._last_result: Optional[AnalysisResult] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)

        source = ttk.LabelFrame(root, text="Dataset")
        source.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        source.columnconfigure(1, weight=1)
        ttk.Label(source, text="CSV file").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(source, textvariable=self.csv_path).grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(source, text="Browse", command=self._browse_csv).grid(row=0, column=2, padx=8, pady=8)
        ttk.Label(source, text="Label column").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.label_combo = ttk.Combobox(source, textvariable=self.label_column, state="readonly", values=[""])
        self.label_combo.grid(row=1, column=1, sticky="ew", padx=8, pady=8)
        ttk.Label(source, text="Anomaly values").grid(row=1, column=2, sticky="w", padx=8, pady=8)
        ttk.Entry(source, textvariable=self.anomaly_values, width=28).grid(row=1, column=3, sticky="ew", padx=8, pady=8)

        model = ttk.LabelFrame(root, text="Model")
        model.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        model.columnconfigure(1, weight=1)
        ttk.Label(model, text="NSA variant").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Combobox(model, textvariable=self.model_name, state="readonly", values=model_names()).grid(
            row=0, column=1, sticky="ew", padx=8, pady=8
        )
        ttk.Label(model, text="Match rule").grid(row=0, column=2, sticky="w", padx=8, pady=8)
        ttk.Combobox(model, textvariable=self.match_rule, state="readonly", values=["hamming", "rcb", "rchunk"]).grid(
            row=0, column=3, sticky="ew", padx=8, pady=8
        )

        params = ttk.LabelFrame(root, text="Parameters")
        params.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for index, (name, _) in enumerate(PARAMETERS):
            label_col = 0 if index < 6 else 2
            entry_col = 1 if index < 6 else 3
            row = index if index < 6 else index - 6
            ttk.Label(params, text=name).grid(row=row, column=label_col, sticky="w", padx=8, pady=4)
            ttk.Entry(params, textvariable=self.parameter_vars[name], width=16).grid(
                row=row, column=entry_col, sticky="ew", padx=8, pady=4
            )
        params.columnconfigure(1, weight=1)
        params.columnconfigure(3, weight=1)

        actions = ttk.Frame(root)
        actions.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        actions.columnconfigure(2, weight=1)
        ttk.Button(actions, text="Run", command=self._run_analysis).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text="Save Results", command=self._save_results).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(actions, textvariable=self.status).grid(row=0, column=2, sticky="w")

        output = ttk.LabelFrame(root, text="Results")
        output.grid(row=4, column=0, sticky="nsew")
        output.rowconfigure(0, weight=1)
        output.columnconfigure(0, weight=1)
        self.results_text = tk.Text(output, wrap="word", height=18)
        scrollbar = ttk.Scrollbar(output, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CSV dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            headers = read_csv_header(path)
        except Exception as exc:
            messagebox.showerror("CSV error", str(exc))
            return
        self.csv_path.set(path)
        self.label_combo.configure(values=[""] + headers)
        self.label_column.set("")
        self.status.set(f"Loaded header with {len(headers)} columns")

    def _run_analysis(self) -> None:
        if not self.csv_path.get():
            messagebox.showwarning("Missing CSV", "Select a CSV file first.")
            return
        try:
            dataset = load_csv_dataset(
                self.csv_path.get(),
                label_column=self.label_column.get(),
                anomaly_values=self._anomaly_values(),
            )
            params = self._model_params()
            model = build_model(self.model_name.get(), **params)
            X_train = self._training_data(dataset)
            self.configure(cursor="watch")
            self.update_idletasks()
            fit_started = time.perf_counter()
            model.fit(X_train)
            fit_seconds = time.perf_counter() - fit_started
            score_started = time.perf_counter()
            scores = model.decision_function(dataset.X)
            predictions = model.predict(dataset.X)
            score_seconds = time.perf_counter() - score_started
            self._last_result = AnalysisResult(dataset, scores, predictions, self.model_name.get(), fit_seconds, score_seconds)
            self._show_summary()
            self.status.set("Analysis complete")
        except Exception as exc:
            messagebox.showerror("Analysis error", str(exc))
            self.status.set("Analysis failed")
        finally:
            self.configure(cursor="")

    def _anomaly_values(self) -> list[str]:
        return [value.strip() for value in self.anomaly_values.get().split(",") if value.strip()]

    def _model_params(self) -> Dict[str, object]:
        params: Dict[str, object] = {"match_rule": self.match_rule.get()}
        int_fields = {"n_detectors", "n_grid", "n_bits", "r", "n_clusters", "n_estimators", "optimization_iter", "random_state"}
        for name, var in self.parameter_vars.items():
            raw = var.get().strip()
            if raw == "":
                if name == "radius":
                    params[name] = None
                continue
            params[name] = int(raw) if name in int_fields else float(raw)
        return params

    def _training_data(self, dataset: CsvDataset) -> np.ndarray:
        if dataset.y is None:
            return dataset.X
        normal = dataset.X[dataset.y == 0]
        if normal.shape[0] == 0:
            raise ValueError("Label column marks every row as anomaly; no normal rows are available for NSA training")
        return normal

    def _show_summary(self) -> None:
        if self._last_result is None:
            return
        result = self._last_result
        dataset = result.dataset
        lines = [
            f"Model: {result.model_name}",
            f"Rows: {dataset.row_count}",
            f"Numeric features: {len(dataset.feature_columns)}",
            f"Feature columns: {', '.join(dataset.feature_columns)}",
            f"Skipped columns: {', '.join(dataset.skipped_columns) if dataset.skipped_columns else 'None'}",
            f"Predicted anomalies: {int(np.sum(result.predictions))}",
            f"Score min/mean/max: {np.min(result.scores):.6f} / {np.mean(result.scores):.6f} / {np.max(result.scores):.6f}",
            f"Fit seconds: {result.fit_seconds:.4f}",
            f"Score seconds: {result.score_seconds:.4f}",
        ]
        if dataset.y is not None and len(np.unique(dataset.y)) == 2:
            lines.extend(
                [
                    "",
                    "Metrics",
                    f"ROC AUC: {roc_auc_score(dataset.y, result.scores):.4f}",
                    f"Precision: {precision_score(dataset.y, result.predictions, zero_division=0):.4f}",
                    f"Recall: {recall_score(dataset.y, result.predictions, zero_division=0):.4f}",
                    f"F1: {f1_score(dataset.y, result.predictions, zero_division=0):.4f}",
                ]
            )
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "\n".join(lines))

    def _save_results(self) -> None:
        if self._last_result is None:
            messagebox.showwarning("No results", "Run analysis before saving.")
            return
        path = filedialog.asksaveasfilename(
            title="Save predictions",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        result = self._last_result
        write_prediction_csv(path, result.scores, result.predictions, result.dataset.y)
        self.status.set(f"Saved {path}")


def main() -> None:
    """Launch the GUI."""
    app = NSAGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
