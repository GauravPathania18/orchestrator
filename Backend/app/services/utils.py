"""
Shared utilities for the services layer.

This module provides common functions used across multiple service modules
to avoid code duplication and ensure consistency.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def cosine_similarity(a, b) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector (list or numpy array)
        b: Second vector (list or numpy array)
    
    Returns:
        float between -1.0 and 1.0
        Returns 0.0 safely if either vector is zero (no crash)
    
    Single source of truth — import this everywhere.
    Never copy this function into another file.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Zero vector guard — division by zero protection
    if norm_a == 0 or norm_b == 0:
        logger.warning("cosine_similarity received a zero vector — returning 0.0")
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))
