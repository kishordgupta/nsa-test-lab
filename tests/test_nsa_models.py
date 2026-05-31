import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score

from experiments.datasets import dataset_names, load_datasets
from experiments.registry import model_names, model_specs
from nsa_test_lab import BinaryNSA, GridNSA, MNSA, RNSA, VDetector
from nsa_test_lab.models.binary_nsa import BinaryNSA as FileBinaryNSA
from nsa_test_lab.models.v_detector import VDetector as FileVDetector


def _sample_data(seed=7):
    rng = np.random.default_rng(seed)
    X_in, _ = make_blobs(
        n_samples=360,
        centers=[[-1.0, -1.0], [1.0, 1.0]],
        cluster_std=[0.25, 0.30],
        random_state=seed,
    )
    pool = rng.uniform(-3.0, 3.0, size=(200, 2))
    X_out = pool[np.linalg.norm(pool, axis=1) > 2.2][:40]
    order = rng.permutation(len(X_in))
    X_train = X_in[order[:240]]
    X_test = np.vstack([X_in[order[240:]], X_out])
    y_test = np.r_[np.zeros(len(X_in) - 240, dtype=int), np.ones(len(X_out), dtype=int)]
    return X_train, X_test, y_test, float(np.mean(y_test))


def test_primary_models_fit_score_predict():
    X_train, X_test, y_test, contamination = _sample_data()
    models = [
        BinaryNSA(contamination=contamination, n_detectors=16, random_state=1),
        RNSA(contamination=contamination, n_detectors=16, random_state=1),
        VDetector(contamination=contamination, n_detectors=16, random_state=1),
        GridNSA(contamination=contamination, n_detectors=16, random_state=1),
        MNSA(contamination=contamination, n_detectors=16, n_estimators=2, random_state=1),
    ]
    for model in models:
        model.fit(X_train)
        scores = model.decision_function(X_test)
        labels = model.predict(X_test)
        assert scores.shape == (X_test.shape[0],)
        assert labels.shape == (X_test.shape[0],)
        assert np.all(np.isfinite(scores))
        assert set(np.unique(labels)).issubset({0, 1})
        assert roc_auc_score(y_test, scores) >= 0.55


def test_per_variant_files_export_expected_classes():
    assert FileBinaryNSA is BinaryNSA
    assert FileVDetector is VDetector


def test_registry_covers_all_variants():
    specs = model_specs(["all"])
    names = model_names()
    assert len(specs) == 32
    assert len(names) == len(set(names))
    assert "VDetector" in names
    assert "CNSA" in names


def test_dataset_registry_has_10_to_20_datasets():
    names = dataset_names()
    assert 10 <= len(names) <= 20
    bundles = load_datasets(["remote_blobs", "s_curve_3d"], seed=42)
    for bundle in bundles:
        assert bundle.X_train.ndim == 2
        assert bundle.X_test.ndim == 2
        assert bundle.y_test.shape == (bundle.X_test.shape[0],)
        assert 0.0 < bundle.contamination < 0.5
