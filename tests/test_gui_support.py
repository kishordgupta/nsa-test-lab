import csv

import numpy as np

from nsa_test_lab.csv_data import load_csv_dataset, read_csv_header, write_prediction_csv
from nsa_test_lab.factory import build_model, model_names, parameter_names


def test_factory_builds_supported_models():
    names = model_names()
    assert "VDetector" in names
    assert "BinaryNSA" in names
    model = build_model(
        "VDetector",
        contamination=0.1,
        n_detectors=8,
        n_grid=99,
        optimization_iter=1,
        random_state=42,
    )
    assert model.__class__.__name__ == "VDetector"
    assert "n_detectors" in parameter_names("VDetector")


def test_csv_loader_reads_features_and_labels(tmp_path):
    path = tmp_path / "sample.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x1", "x2", "label", "note"])
        writer.writerow([0.0, 1.0, "normal", "a"])
        writer.writerow([1.0, 2.0, "attack", "b"])
        writer.writerow([2.0, 3.0, "0", "c"])

    assert read_csv_header(path) == ["x1", "x2", "label", "note"]
    dataset = load_csv_dataset(path, label_column="label", anomaly_values=["attack"])
    assert dataset.feature_columns == ["x1", "x2"]
    assert dataset.skipped_columns == ["note"]
    assert dataset.X.shape == (3, 2)
    assert dataset.y.tolist() == [0, 1, 0]


def test_prediction_writer(tmp_path):
    path = tmp_path / "predictions.csv"
    write_prediction_csv(path, np.array([0.2, 1.5]), np.array([0, 1]), np.array([0, 1]))
    text = path.read_text(encoding="utf-8")
    assert "anomaly_score" in text
    assert "true_label" in text
