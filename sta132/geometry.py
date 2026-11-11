"""Geometry of straight line elements in the global x-z plane."""

import numpy as np


def geometry_line_2d(xa, xb):
    """Return the length and direction cosines from endpoints a and b.

    xa: (2,) global endpoint coordinates [x, z] of node a
    xb: (2,) global endpoint coordinates [x, z] of node b

    Returns (L, c, s), where L is the line length and c, s are the direction
    cosines from a to b.
    """
    xa = np.asarray(xa, dtype=float)
    xb = np.asarray(xb, dtype=float)
    if xa.shape != (2,) or xb.shape != (2,):
        raise ValueError("xa and xb must each contain two coordinates: [x, z].")
    if not np.isfinite(xa).all() or not np.isfinite(xb).all():
        raise ValueError("Endpoint coordinates must be finite.")

    dx, dz = xb - xa
    L = np.hypot(dx, dz)
    if L == 0:
        raise ValueError("The two endpoints of a line must not coincide.")

    return L, dx / L, dz / L
