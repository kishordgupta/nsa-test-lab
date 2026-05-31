# NSA Experimental Report

Generated: 2026-05-31 04:22 UTC

## Configuration

- Datasets: 15
- NSA variants: 32
- Experiment runs: 480
- Detectors per model: 32
- Optimization iterations: 3
- Seed: 42
- Successful runs: 480
- Failed runs: 0

## Top Models By Mean ROC AUC

| Rank | Model | Family | Mean ROC AUC | Mean F1 | Runs |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | ORNSA | boundary-aware | 0.9955 | 0.6505 | 15 |
| 2 | FtNSA | feature/self-space | 0.9954 | 0.6537 | 15 |
| 3 | BIORVNSA | inhibition | 0.9949 | 0.7235 | 15 |
| 4 | IVRNSA | variable-radius | 0.9944 | 0.7233 | 15 |
| 5 | OptimizedNSA | optimization | 0.9938 | 0.7212 | 15 |
| 6 | IORNSA | optimization | 0.9934 | 0.7272 | 15 |
| 7 | AINSA | adaptive | 0.9934 | 0.7264 | 15 |
| 8 | OALFBNSA | online/feedback | 0.9934 | 0.7257 | 15 |
| 9 | ANSA | adaptive | 0.9934 | 0.7255 | 15 |
| 10 | AntigenNSA | density | 0.9934 | 0.7261 | 15 |
| 11 | EvoSeedRNSA | optimization | 0.9934 | 0.7255 | 15 |
| 12 | VDetector | v-detector | 0.9934 | 0.7265 | 15 |

## Best Model Per Dataset

| Dataset | Best Model | ROC AUC | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| anisotropic_blobs | BIORVNSA | 0.9982 | 0.5238 | 1.0000 | 0.6875 |
| correlated_gaussian | BinaryNSA | 1.0000 | 0.5765 | 1.0000 | 0.7313 |
| high_dimensional_blobs | RNSA | 1.0000 | 0.5634 | 1.0000 | 0.7207 |
| line_manifold | RNSA | 1.0000 | 0.5926 | 1.0000 | 0.7442 |
| local_density_bridge | RNSA | 1.0000 | 0.5031 | 1.0000 | 0.6695 |
| mixed_scale | ORNSA | 0.9996 | 0.4734 | 1.0000 | 0.6426 |
| near_blobs | FtNSA | 0.9694 | 0.5417 | 0.9750 | 0.6964 |
| nested_circles | CNSA | 1.0000 | 0.6250 | 1.0000 | 0.7692 |
| remote_blobs | RNSA | 0.9998 | 0.5469 | 1.0000 | 0.7071 |
| ring_outliers | RNSA | 1.0000 | 0.5556 | 1.0000 | 0.7143 |
| s_curve_3d | RNSA | 1.0000 | 0.6452 | 1.0000 | 0.7843 |
| sine_stripe | RNSA | 1.0000 | 0.5714 | 1.0000 | 0.7273 |
| sparse_high_dimensional | RNSA | 1.0000 | 0.5479 | 1.0000 | 0.7080 |
| two_moons | RNSA | 1.0000 | 0.6299 | 1.0000 | 0.7729 |
| varied_density | ORNSA | 0.9820 | 0.4762 | 1.0000 | 0.6452 |

## Dataset Difficulty By Mean ROC AUC

| Dataset | Mean ROC AUC | Runs |
| --- | ---: | ---: |
| near_blobs | 0.9222 | 32 |
| varied_density | 0.9362 | 32 |
| local_density_bridge | 0.9738 | 32 |
| sine_stripe | 0.9748 | 32 |
| mixed_scale | 0.9749 | 32 |
| line_manifold | 0.9779 | 32 |
| anisotropic_blobs | 0.9812 | 32 |
| correlated_gaussian | 0.9826 | 32 |
| ring_outliers | 0.9826 | 32 |
| sparse_high_dimensional | 0.9864 | 32 |
| nested_circles | 0.9891 | 32 |
| remote_blobs | 0.9899 | 32 |
| two_moons | 0.9909 | 32 |
| s_curve_3d | 0.9957 | 32 |
| high_dimensional_blobs | 0.9997 | 32 |

## Interpretation

The benchmark is synthetic and deterministic. Results are useful for comparing relative NSA behavior across geometric data shapes, but they should not be treated as a substitute for domain-specific validation. NSA detectors are distance- and coverage-driven, so separated low-dimensional anomalies tend to favor detector families such as V-detector, density-enhanced NSA, and optimization-based variants. High-dimensional sparse problems are more sensitive to feature scaling and detector coverage.

Full per-run results are available in `reports/results.csv` and `reports/results.json`.
