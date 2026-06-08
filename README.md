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

## Test

```bash
pytest
```

##cite
K. D. Gupta and D. Dasgupta, "Negative Selection Algorithm Research and Applications in the Last Decade: A Review," in IEEE Transactions on Artificial Intelligence, vol. 3, no. 2, pp. 110-128, April 2022, doi: 10.1109/TAI.2021.3114661.
Abstract: The negative selection algorithm (NSA) is one of the important methods in the field of immunological computation (or artificial immune systems). Over the years, some progress was made that turns this algorithm (NSA) into an efficient approach to solve problems in different domain. This review takes into account these signs of progress during the last decade and categorizes those based on different characteristics and performances. Our study shows that NSA’s evolution can be labeled in four ways highlighting the most notable NSA variations and their limitations in different application domains. We also present alternative approaches to NSA for comparison and analysis. It is evident that NSA performs better for nonlinear representation than most of the other methods, and it can outperform neural-based models in computation time. We summarize NSA’s development and highlight challenges in NSA research in comparison with other similar models.
keywords: {Detectors;Immune system;Artificial intelligence;Complexity theory;Anomaly detection;Hamming distance;Computational modeling;Artificial immune system (AIS);immunological computation;negative data representation;negative selection algorithm (NSA)},
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9546626&isnumber=9741091




## Notes

These implementations are engineering approximations of NSA families exposed
through a scikit-learn-like API. They are intended for comparative experiments,
teaching, and applied anomaly-detection prototyping.
