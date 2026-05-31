"""Deterministic synthetic datasets for NSA benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons, make_s_curve


@dataclass(frozen=True)
class DatasetBundle:
    """Train/test split for one novelty-detection experiment."""

    name: str
    description: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    contamination: float


DatasetFactory = Callable[[int], DatasetBundle]


def _bundle(
    name: str,
    description: str,
    X_in: np.ndarray,
    X_out: np.ndarray,
    seed: int,
    train_size: int = 300,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    X_in = np.asarray(X_in, dtype=float)
    X_out = np.asarray(X_out, dtype=float)
    train_size = min(train_size, max(1, X_in.shape[0] - 120))
    order = rng.permutation(X_in.shape[0])
    train_idx = order[:train_size]
    test_idx = order[train_size:]
    X_train = X_in[train_idx]
    X_test = np.vstack([X_in[test_idx], X_out])
    y_test = np.r_[np.zeros(len(test_idx), dtype=int), np.ones(X_out.shape[0], dtype=int)]
    return DatasetBundle(
        name=name,
        description=description,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        contamination=float(np.mean(y_test)),
    )


def _remote_blobs(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    X_in, _ = make_blobs(
        n_samples=520,
        centers=[[-1.0, -1.0], [1.0, 1.0]],
        cluster_std=[0.25, 0.30],
        random_state=seed,
    )
    pool = rng.uniform(-3.5, 3.5, size=(400, 2))
    X_out = pool[np.linalg.norm(pool, axis=1) > 2.4][:70]
    return _bundle("remote_blobs", "Separated Gaussian clusters with far uniform anomalies.", X_in, X_out, seed)


def _near_blobs(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 1)
    X_in, _ = make_blobs(
        n_samples=520,
        centers=[[-1.0, -1.0], [1.0, 1.0]],
        cluster_std=[0.22, 0.26],
        random_state=seed + 1,
    )
    offsets = rng.normal(0.0, 0.18, size=(80, 2))
    anchors = np.repeat([[-1.55, -1.55], [1.55, 1.55]], repeats=40, axis=0)
    X_out = anchors + offsets
    return _bundle("near_blobs", "Anomalies close to normal clusters but just outside the self boundary.", X_in, X_out, seed)


def _anisotropic_blobs(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 2)
    X_in, _ = make_blobs(n_samples=540, centers=3, cluster_std=0.45, random_state=seed + 2)
    X_in = X_in @ np.array([[0.6, -0.8], [1.7, 0.4]])
    X_out = rng.uniform(X_in.min(axis=0) - 2.0, X_in.max(axis=0) + 2.0, size=(90, 2))
    center = X_in.mean(axis=0)
    X_out = X_out[np.linalg.norm(X_out - center, axis=1) > np.percentile(np.linalg.norm(X_in - center, axis=1), 88)][:75]
    return _bundle("anisotropic_blobs", "Rotated anisotropic clusters with broad off-manifold anomalies.", X_in, X_out, seed)


def _varied_density(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 3)
    X_in, _ = make_blobs(
        n_samples=560,
        centers=[[-2.0, 0.0], [0.5, 1.2], [1.5, -1.0]],
        cluster_std=[0.12, 0.45, 0.75],
        random_state=seed + 3,
    )
    X_out = rng.uniform([-3.2, -2.8], [3.1, 2.8], size=(350, 2))
    d = np.min(np.linalg.norm(X_out[:, None, :] - np.array([[-2.0, 0.0], [0.5, 1.2], [1.5, -1.0]])[None, :, :], axis=2), axis=1)
    X_out = X_out[d > 1.25][:80]
    return _bundle("varied_density", "Three clusters with unequal density and dispersed anomalies.", X_in, X_out, seed)


def _moons(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 4)
    X_in, _ = make_moons(n_samples=540, noise=0.06, random_state=seed + 4)
    pool = rng.uniform([-1.5, -1.0], [2.5, 1.6], size=(600, 2))
    d = np.min(np.linalg.norm(pool[:, None, :] - X_in[None, :, :], axis=2), axis=1)
    X_out = pool[d > 0.28][:80]
    return _bundle("two_moons", "Nonlinear two-moon self shape with off-curve anomalies.", X_in, X_out, seed)


def _circles(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 5)
    X_in, _ = make_circles(n_samples=560, factor=0.45, noise=0.04, random_state=seed + 5)
    pool = rng.uniform(-1.45, 1.45, size=(500, 2))
    radius = np.linalg.norm(pool, axis=1)
    X_out = pool[(radius > 1.12) | ((radius > 0.62) & (radius < 0.82))][:80]
    return _bundle("nested_circles", "Nested circular self manifolds with annular and external anomalies.", X_in, X_out, seed)


def _ring_outliers(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 6)
    X_in = rng.normal(0.0, 0.35, size=(520, 2))
    angle = rng.uniform(0, 2 * np.pi, size=80)
    radius = rng.normal(2.1, 0.12, size=80)
    X_out = np.c_[np.cos(angle) * radius, np.sin(angle) * radius]
    return _bundle("ring_outliers", "Compact Gaussian self region with ring-shaped anomalies.", X_in, X_out, seed)


def _local_density(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 7)
    X_left = rng.normal([-1.4, 0.0], [0.20, 0.28], size=(260, 2))
    X_right = rng.normal([1.4, 0.0], [0.20, 0.28], size=(260, 2))
    X_in = np.vstack([X_left, X_right])
    X_out = rng.normal([0.0, 0.0], [0.12, 0.55], size=(80, 2))
    return _bundle("local_density_bridge", "Anomalies lie in a low-density bridge between dense self clusters.", X_in, X_out, seed)


def _high_dimensional(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 8)
    X_in, _ = make_blobs(n_samples=540, centers=3, n_features=10, cluster_std=0.35, random_state=seed + 8)
    X_out = rng.normal(0.0, 1.0, size=(80, 10))
    X_out[:, :3] += rng.choice([-4.0, 4.0], size=(80, 3))
    return _bundle("high_dimensional_blobs", "Ten-dimensional Gaussian clusters with shifted-feature anomalies.", X_in, X_out, seed)


def _sparse_high_dimensional(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 9)
    X_in = rng.normal(0.0, 0.04, size=(540, 20))
    active = rng.integers(0, 20, size=(80, 3))
    X_out = rng.normal(0.0, 0.04, size=(80, 20))
    for row, cols in enumerate(active):
        X_out[row, cols] += rng.choice([-1.4, 1.4], size=3)
    return _bundle("sparse_high_dimensional", "Twenty-dimensional sparse baseline with rare feature spikes.", X_in, X_out, seed)


def _correlated_gaussian(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 10)
    cov = np.array([[1.0, 0.92], [0.92, 1.0]]) * 0.22
    X_in = rng.multivariate_normal([0.0, 0.0], cov, size=540)
    X_out = rng.multivariate_normal([0.0, 0.0], np.array([[0.22, -0.20], [-0.20, 0.22]]), size=120)
    X_out = X_out[np.abs(X_out[:, 0] - X_out[:, 1]) > 0.7][:80]
    return _bundle("correlated_gaussian", "Strong positive-correlation self cloud with anti-correlated anomalies.", X_in, X_out, seed)


def _line_manifold(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 11)
    t = rng.uniform(-2.0, 2.0, size=540)
    X_in = np.c_[t, 0.65 * t + rng.normal(0.0, 0.06, size=t.shape)]
    t_out = rng.uniform(-2.0, 2.0, size=80)
    X_out = np.c_[t_out, 0.65 * t_out + rng.choice([-0.65, 0.65], size=80) + rng.normal(0.0, 0.05, size=80)]
    return _bundle("line_manifold", "Noisy linear self manifold with parallel-offset anomalies.", X_in, X_out, seed)


def _sine_stripe(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 12)
    x = rng.uniform(-3.0, 3.0, size=540)
    y = np.sin(x) + rng.normal(0.0, 0.08, size=540)
    X_in = np.c_[x, y]
    x_out = rng.uniform(-3.0, 3.0, size=80)
    y_out = np.sin(x_out) + rng.choice([-0.75, 0.75], size=80) + rng.normal(0.0, 0.05, size=80)
    X_out = np.c_[x_out, y_out]
    return _bundle("sine_stripe", "Sine-curve self band with vertically offset anomalies.", X_in, X_out, seed)


def _s_curve(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 13)
    X_in, _ = make_s_curve(n_samples=540, noise=0.04, random_state=seed + 13)
    X_in = X_in[:, [0, 1, 2]]
    X_out = rng.uniform(X_in.min(axis=0) - 1.0, X_in.max(axis=0) + 1.0, size=(500, 3))
    d = np.min(np.linalg.norm(X_out[:, None, :] - X_in[None, :, :], axis=2), axis=1)
    X_out = X_out[d > 0.95][:80]
    return _bundle("s_curve_3d", "Three-dimensional S-curve self manifold with off-surface anomalies.", X_in, X_out, seed)


def _mixed_scale(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 14)
    base = rng.normal(0.0, 1.0, size=(540, 4))
    X_in = np.c_[base[:, 0] * 0.1, base[:, 1] * 3.0, base[:, 2] + base[:, 0], base[:, 3] * 0.4]
    X_out = rng.normal(0.0, 1.0, size=(80, 4))
    X_out = np.c_[X_out[:, 0] * 0.1 + 0.5, X_out[:, 1] * 3.0 - 8.0, X_out[:, 2] + 3.5, X_out[:, 3] * 0.4]
    return _bundle("mixed_scale", "Four features with different scales and shifted multivariate anomalies.", X_in, X_out, seed)


DATASETS: Dict[str, DatasetFactory] = {
    "remote_blobs": _remote_blobs,
    "near_blobs": _near_blobs,
    "anisotropic_blobs": _anisotropic_blobs,
    "varied_density": _varied_density,
    "two_moons": _moons,
    "nested_circles": _circles,
    "ring_outliers": _ring_outliers,
    "local_density_bridge": _local_density,
    "high_dimensional_blobs": _high_dimensional,
    "sparse_high_dimensional": _sparse_high_dimensional,
    "correlated_gaussian": _correlated_gaussian,
    "line_manifold": _line_manifold,
    "sine_stripe": _sine_stripe,
    "s_curve_3d": _s_curve,
    "mixed_scale": _mixed_scale,
}


def dataset_names() -> List[str]:
    """Return dataset names in benchmark order."""
    return list(DATASETS)


def load_datasets(names: Iterable[str], seed: int) -> List[DatasetBundle]:
    """Build datasets by name."""
    requested = dataset_names() if "all" in names else list(names)
    unknown = sorted(set(requested) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    return [DATASETS[name](seed) for name in requested]
