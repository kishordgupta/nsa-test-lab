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

You can also import from the top-level package:

```python
from nsa_test_lab import BinaryNSA, RNSA, VDetector, GridNSA, MNSA
```

## NSA Variants

The repo includes separate public files for these NSA types:

`BinaryNSA`, `RNSA`, `RRNSA`, `VDetector`, `GridNSA`, `GFRNSA`,
`MatrixNSA`, `ANSA`, `EvoSeedRNSA`, `ORNSA`, `OptimizedNSA`, `FtNSA`,
`IVRNSA`, `CBNSA`, `PRR2NSA`, `DENSElectionNSA`, `DENSA`, `AntigenNSA`,
`NSADE`, `NSAPSO`, `IORNSA`, `BIORVNSA`, `HNSAIDSA`, `NSAII`, `OALFBNSA`,
`FBNSA`, `MNSA`, `NSNAD`, `RENNSA`, `AINSA`, `ODNSA`, `CNSA`, and `VORNSA`.

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

## Launch GUI

```bash
nsa-gui
```

From an uninstalled checkout, use:

```bash
PYTHONPATH=src:. python -m nsa_test_lab.gui
```

The GUI lets users choose an NSA variant, edit detector parameters, select a
CSV file, choose an optional label column, run scoring, and export row-level
anomaly scores. CSV files should include a header row and numeric feature
columns. If a label column is selected, values matching
`1,true,yes,anomaly,outlier,attack` are treated as anomalies by default.

## Test

```bash
pytest
```

## Notes

These implementations are engineering approximations of NSA families exposed
through a scikit-learn-like API. They are intended for comparative experiments,
teaching, and applied anomaly-detection prototyping.
