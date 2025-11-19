"""Optional mitigation strategies for NullBench decoder evaluations."""
from .calibration import build_mitigated_predict_fn, MitigationConfig

__all__ = ["build_mitigated_predict_fn", "MitigationConfig"]
