# NSA Test Lab

Standalone Negative Selection Algorithm (NSA) detectors plus a reproducible
benchmark suite for comparing NSA variants on synthetic anomaly-detection
datasets.

The package exposes each NSA family in its own file under
`nsa_test_lab.models`, while sharing one tested core implementation. This keeps
the repository easy to inspect and import as a library.

## Install

```bash
pip install -e ".[dev]"
```

## Library Usage

```python
from nsa_test_lab.models.v_detector import VDetector

model = VDetector(contamination=0.1, n_detectors=64, random_state=42)
model.fit(X_train)
scores = model.decision_function(X_test)
labels = model.predict(X_test)  # 1 = anomaly, 0 = inlier
```

Contrastive one-class detectors are also available from the package:

```python
from nsa_test_lab import InfoNCEContrastiveDetector

model = InfoNCEContrastiveDetector(contamination=0.1, random_state=42)
model.fit(X_train)
scores = model.decision_function(X_test)
```

You can also import NSA models from the top-level package:

```python
from nsa_test_lab import BinaryNSA, RNSA, VDetector, GridNSA, MNSA
```

## NSA Variants

The repo includes separate public files for these NSA types:

`BinaryNSA`, `RNSA`, `RRNSA`, `VDetector`, `GridNSA`, `GFRNSA`,
`MatrixNSA`, `ANSA`, `EvoSeedRNSA`, `ORNSA`, `OptimizedNSA`,
`FtNSA`, `IVRNSA`, `CBNSA`, `PRR2NSA`, `DENSElectionNSA`, `DENSA`,
`AntigenNSA`, `NSADE`, `NSAPSO`, `IORNSA`, `BIORVNSA`, `HNSAIDSA`,
`NSAII`, `OALFBNSA`, `FBNSA`, `MNSA`, `NSNAD`, `RENNSA`, `AINSA`,
`ODNSA`, `CNSA`, and `VORNSA`.

## Run Experiments

```bash
nsa-benchmark --detectors 32 --optimization-iter 3 --datasets all
```

From an uninstalled checkout, use:

```bash
PYTHONPATH=src:. python -m experiments.run_benchmark --detectors 32 --optimization-iter 3 --datasets all
```

The command writes:

* `reports/results.csv`
* `reports/results.json`
* `reports/experimental_report.md`

The default benchmark uses 15 deterministic datasets spanning separated blobs,
close anomalies, anisotropic clusters, moons, circles, high-dimensional sparse
data, correlated Gaussian data, ring outliers, local-density outliers, linear
manifold anomalies, and S-curve anomalies.

## Contrastive and Baseline Comparison

The extended benchmark compares the existing NSA registry with:

* `ContrastiveInfoNCE`
* `ContrastiveTriplet`
* `GaussianInfoNCE`
* `IsolationForest`
* `LocalOutlierFactor`
* `OneClassSVM`

Run a fast real-data comparison:

```bash
nsa-contrastive-benchmark --dataset-source real --datasets all --models contrastive,baseline --detectors 24 --optimization-iter 1
```

Run selected NSA models against the new contrastive detectors:

```bash
PYTHONPATH=src:. python -m experiments.run_contrastive_benchmark \
  --dataset-source synthetic \
  --datasets two_moons,nested_circles,high_dimensional_blobs \
  --models VDetector,DENSA,MNSA,contrastive,baseline \
  --detectors 32 \
  --optimization-iter 3
```

The contrastive benchmark writes:

* `reports/contrastive_results.csv`
* `reports/contrastive_results.json`
* `reports/contrastive_benchmark_report.md`

The additional built-in real-data novelty tasks are `breast_cancer_malignant`,
`wine_class_1_vs_rest`, `digits_zero_vs_odd`, and `digits_even_vs_nine`.

## Test

```bash
pytest
```

## Citation

K. D. Gupta and D. Dasgupta, "Negative Selection Algorithm Research and Applications in the Last Decade: A Review," in IEEE Transactions on Artificial Intelligence, vol. 3, no. 2, pp. 110-128, April 2022, doi: 10.1109/TAI.2021.3114661.

## Notes

These implementations are engineering approximations of NSA families exposed
through a scikit-learn-like API. They are intended for comparative experiments,
teaching, and applied anomaly-detection prototyping.
