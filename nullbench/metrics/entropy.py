import numpy as np

def compute_entropy(probs: np.ndarray) -> float:
    """
    Mean entropy H(p(.|x)) over null inputs.
    Higher = more uncertainty (good under null).
    """
    eps = 1e-12
    logp = np.log(probs + eps)
    ent = -np.sum(probs * logp, axis=1)  # [N]
    return float(ent.mean())
