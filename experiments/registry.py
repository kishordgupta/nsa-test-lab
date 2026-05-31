"""Registry for all NSA detector variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

from nsa_test_lab import (
    AINSA,
    ANSA,
    BIORVNSA,
    BinaryNSA,
    CBNSA,
    CNSA,
    DENSA,
    EvoSeedRNSA,
    FBNSA,
    FtNSA,
    GFRNSA,
    GridNSA,
    HNSAIDSA,
    IORNSA,
    IVRNSA,
    MNSA,
    MatrixNSA,
    NSADE,
    NSAII,
    NSAPSO,
    NSNAD,
    OALFBNSA,
    ODNSA,
    ORNSA,
    OptimizedNSA,
    PRR2NSA,
    RENNSA,
    RNSA,
    RRNSA,
    VDetector,
    VORNSA,
    AntigenNSA,
)


@dataclass(frozen=True)
class ModelSpec:
    """Factory metadata for one NSA model."""

    name: str
    family: str
    build: Callable[[float, int, int, int], object]


def model_specs(names: Iterable[str] | None = None) -> List[ModelSpec]:
    """Return model specs in benchmark order."""
    specs = [
        ModelSpec("BinaryNSA", "binary/string", lambda c, n, opt, seed: BinaryNSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("RNSA", "real-valued", lambda c, n, opt, seed: RNSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("RRNSA", "real-valued", lambda c, n, opt, seed: RRNSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("VDetector", "v-detector", lambda c, n, opt, seed: VDetector(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("GridNSA", "grid/matrix", lambda c, n, opt, seed: GridNSA(contamination=c, n_detectors=n, n_grid=8, random_state=seed)),
        ModelSpec("GFRNSA", "grid/matrix", lambda c, n, opt, seed: GFRNSA(contamination=c, n_detectors=n, n_grid=8, random_state=seed)),
        ModelSpec("MatrixNSA", "grid/matrix", lambda c, n, opt, seed: MatrixNSA(contamination=c, n_detectors=n, n_grid=8, random_state=seed)),
        ModelSpec("ANSA", "adaptive", lambda c, n, opt, seed: ANSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("EvoSeedRNSA", "optimization", lambda c, n, opt, seed: EvoSeedRNSA(contamination=c, n_detectors=n, optimization_iter=opt, random_state=seed)),
        ModelSpec("ORNSA", "boundary-aware", lambda c, n, opt, seed: ORNSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("OptimizedNSA", "optimization", lambda c, n, opt, seed: OptimizedNSA(contamination=c, n_detectors=n, optimization_iter=opt, random_state=seed)),
        ModelSpec("FtNSA", "feature/self-space", lambda c, n, opt, seed: FtNSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("IVRNSA", "variable-radius", lambda c, n, opt, seed: IVRNSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("CBNSA", "clustering", lambda c, n, opt, seed: CBNSA(contamination=c, n_detectors=n, n_clusters=6, random_state=seed)),
        ModelSpec("PRR2NSA", "clustering", lambda c, n, opt, seed: PRR2NSA(contamination=c, n_detectors=n, n_clusters=6, random_state=seed)),
        ModelSpec("DENSA", "density", lambda c, n, opt, seed: DENSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("AntigenNSA", "density", lambda c, n, opt, seed: AntigenNSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("NSADE", "optimization", lambda c, n, opt, seed: NSADE(contamination=c, n_detectors=n, optimization_iter=opt, random_state=seed)),
        ModelSpec("NSAPSO", "optimization", lambda c, n, opt, seed: NSAPSO(contamination=c, n_detectors=n, optimization_iter=opt, random_state=seed)),
        ModelSpec("IORNSA", "optimization", lambda c, n, opt, seed: IORNSA(contamination=c, n_detectors=n, optimization_iter=opt, random_state=seed)),
        ModelSpec("BIORVNSA", "inhibition", lambda c, n, opt, seed: BIORVNSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("HNSAIDSA", "hybrid", lambda c, n, opt, seed: HNSAIDSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("NSAII", "hybrid", lambda c, n, opt, seed: NSAII(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("OALFBNSA", "online/feedback", lambda c, n, opt, seed: OALFBNSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("FBNSA", "online/feedback", lambda c, n, opt, seed: FBNSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("MNSA", "ensemble", lambda c, n, opt, seed: MNSA(contamination=c, n_detectors=n, n_estimators=3, random_state=seed)),
        ModelSpec("NSNAD", "feature-subspace", lambda c, n, opt, seed: NSNAD(contamination=c, n_detectors=n, feature_subsample=0.7, random_state=seed)),
        ModelSpec("RENNSA", "reduced-overlap", lambda c, n, opt, seed: RENNSA(contamination=c, n_detectors=n, random_state=seed)),
        ModelSpec("AINSA", "adaptive", lambda c, n, opt, seed: AINSA(contamination=c, n_detectors=n, optimization_iter=opt, random_state=seed)),
        ModelSpec("ODNSA", "optimization", lambda c, n, opt, seed: ODNSA(contamination=c, n_detectors=n, optimization_iter=opt, random_state=seed)),
        ModelSpec("CNSA", "clustering", lambda c, n, opt, seed: CNSA(contamination=c, n_detectors=n, n_clusters=6, optimization_iter=opt, random_state=seed)),
        ModelSpec("VORNSA", "vectorized", lambda c, n, opt, seed: VORNSA(contamination=c, n_detectors=n, random_state=seed)),
    ]
    if names is None or "all" in names:
        return specs
    requested = set(names)
    known = {spec.name for spec in specs}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    return [spec for spec in specs if spec.name in requested]


def model_names() -> List[str]:
    """Return all benchmark model names."""
    return [spec.name for spec in model_specs()]
