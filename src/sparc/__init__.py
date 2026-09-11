"""SPARC's reusable scaling core, not the full paper experiment suite."""

from .adapters import LinearLastLayerAdapter, idct_design
from .calibration import SplitConformal
from .covariance import graph_temporal_factor, inflate_factor
from .diagnostics import LeverageExplanation, explain_leverage
from .leverage import FeatureLeverage
from .model import SPARC, Forecast, Prediction

__all__ = [
    "LeverageExplanation",
    "explain_leverage",
    "SPARC",
    "Forecast",
    "Prediction",
    "FeatureLeverage",
    "SplitConformal",
    "LinearLastLayerAdapter",
    "idct_design",
    "graph_temporal_factor",
    "inflate_factor",
]
