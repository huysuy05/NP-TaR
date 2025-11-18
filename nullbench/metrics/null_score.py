def compute_null_score(dcb: float, entropy: float, abstention: float, K: int,
                       alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> float:
    """
    Example combined score:
      NRS = α * entropy + β * abstention - γ * (DCB - 1/K)
    Larger is better.
    """
    dcb_norm = dcb - 1.0 / K
    return float(alpha * entropy + beta * abstention - gamma * dcb_norm)
