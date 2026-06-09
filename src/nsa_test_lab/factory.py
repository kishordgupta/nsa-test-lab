"""Model factory for NSA detector variants."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Type

from .core import (
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
    NegativeSelection,
)


MODEL_CLASSES: Dict[str, Type[NegativeSelection]] = {
    "BinaryNSA": BinaryNSA,
    "RNSA": RNSA,
    "RRNSA": RRNSA,
    "VDetector": VDetector,
    "GridNSA": GridNSA,
    "GFRNSA": GFRNSA,
    "MatrixNSA": MatrixNSA,
    "ANSA": ANSA,
    "EvoSeedRNSA": EvoSeedRNSA,
    "ORNSA": ORNSA,
    "OptimizedNSA": OptimizedNSA,
    "FtNSA": FtNSA,
    "IVRNSA": IVRNSA,
    "CBNSA": CBNSA,
    "PRR2NSA": PRR2NSA,
    "DENSA": DENSA,
    "AntigenNSA": AntigenNSA,
    "NSADE": NSADE,
    "NSAPSO": NSAPSO,
    "IORNSA": IORNSA,
    "BIORVNSA": BIORVNSA,
    "HNSAIDSA": HNSAIDSA,
    "NSAII": NSAII,
    "OALFBNSA": OALFBNSA,
    "FBNSA": FBNSA,
    "MNSA": MNSA,
    "NSNAD": NSNAD,
    "RENNSA": RENNSA,
    "AINSA": AINSA,
    "ODNSA": ODNSA,
    "CNSA": CNSA,
    "VORNSA": VORNSA,
}


def model_names() -> List[str]:
    """Return public NSA model names in UI order."""
    return list(MODEL_CLASSES)


def parameter_names(model_name: str) -> List[str]:
    """Return constructor parameters accepted by a model."""
    model_cls = _model_class(model_name)
    signature = inspect.signature(model_cls.__init__)
    return [name for name in signature.parameters if name != "self"]


def build_model(model_name: str, **params: Any) -> NegativeSelection:
    """Instantiate an NSA model, dropping parameters it does not accept."""
    model_cls = _model_class(model_name)
    allowed = set(parameter_names(model_name))
    filtered = {key: value for key, value in params.items() if key in allowed}
    return model_cls(**filtered)


def _model_class(model_name: str) -> Type[NegativeSelection]:
    try:
        return MODEL_CLASSES[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown NSA model: {model_name}") from exc
