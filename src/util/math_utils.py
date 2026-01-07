"""
Mathematical Utility Functions

Vector operations and filtering for storm motion calculations.
"""

import math
from typing import List, Tuple


def vector_magnitude(u: float, v: float) -> float:
    """
    Compute vector magnitude.
    
    Args:
        u: Zonal component
        v: Meridional component
        
    Returns:
        Vector magnitude: sqrt(u² + v²)
    """
    return math.sqrt(u * u + v * v)


def vector_direction(u: float, v: float) -> float:
    """
    Compute meteorological direction (degrees from north, clockwise).
    
    Args:
        u: Zonal component (positive = east)
        v: Meridional component (positive = north)
        
    Returns:
        Direction in degrees [0, 360)
    """
    return (270.0 - math.degrees(math.atan2(v, u))) % 360.0


def rotate_vector_90(u: float, v: float, clockwise: bool = True) -> Tuple[float, float]:
    """
    Rotate vector 90 degrees.
    
    Args:
        u: Zonal component
        v: Meridional component
        clockwise: If True, rotate clockwise (right); otherwise counter-clockwise (left)
        
    Returns:
        Rotated (u, v) tuple
    """
    if clockwise:
        # Clockwise: (u, v) -> (v, -u)
        return (v, -u)
    else:
        # Counter-clockwise: (u, v) -> (-v, u)
        return (-v, u)


def unit_vector(u: float, v: float) -> Tuple[float, float]:
    """
    Normalize vector to unit length.
    
    Args:
        u: Zonal component
        v: Meridional component
        
    Returns:
        Unit vector (u/|V|, v/|V|). Returns (0, 0) if magnitude is zero.
    """
    mag = vector_magnitude(u, v)
    if mag < 1e-10:
        return (0.0, 0.0)
    return (u / mag, v / mag)


def exponential_filter(
    values: List[Tuple[float, float]], 
    alpha: float = 0.3
) -> Tuple[float, float]:
    """
    Apply exponential smoothing filter to a sequence of (u, v) vectors.
    
    Higher alpha gives more weight to recent observations.
    
    Args:
        values: List of (u, v) tuples in chronological order
        alpha: Smoothing parameter in (0, 1]. Higher = more responsive.
        
    Returns:
        Smoothed (u, v) tuple
        
    Raises:
        ValueError: If values is empty or alpha is out of range
    """
    if not values:
        raise ValueError("Cannot filter empty sequence")
    if not 0 < alpha <= 1:
        raise ValueError(f"Alpha must be in (0, 1], got {alpha}")
    
    u_smooth, v_smooth = values[0]
    
    for u, v in values[1:]:
        u_smooth = alpha * u + (1 - alpha) * u_smooth
        v_smooth = alpha * v + (1 - alpha) * v_smooth
    
    return (u_smooth, v_smooth)


def gaussian(x: float, mu: float, sigma: float) -> float:
    """
    Compute Gaussian function value.
    
    Args:
        x: Input value
        mu: Mean (center)
        sigma: Standard deviation (spread)
        
    Returns:
        exp(-0.5 * ((x - mu) / sigma)²)
    """
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z)


def linear_interpolate(
    x: float, 
    x1: float, 
    x2: float, 
    y1: float, 
    y2: float
) -> float:
    """
    Linear interpolation between two points.
    
    Args:
        x: Input value
        x1, x2: X bounds
        y1, y2: Y values at bounds
        
    Returns:
        Interpolated y value, clamped to [y1, y2] range
    """
    if x <= x1:
        return y1
    if x >= x2:
        return y2
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)
