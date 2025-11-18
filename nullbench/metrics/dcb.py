import numpy as np

def compute_dcb(probs: np.ndarray) -> float:
    """
    Default Class Bias = max_y E_x p(y | x) over null inputs.
    probs: [N, K] numpy array of probabilities
    """

    if probs.ndim != 2:
        raise ValueError("probs must be an 2D Array")
    
    avg = probs.mean(axis=0)
    return float(avg.max())