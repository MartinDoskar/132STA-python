"""Elementary planar truss element functions for 132STA.

Coordinates are [x, z], with x positive to the right and z positive downwards.
Global element displacements are ordered as [u_a, w_a, u_b, w_b].
Use a consistent system of units for all inputs and outputs.
"""

import numpy as np

from .geometry import geometry_line_2d


def local_stiffness_matrix_truss(E, A, L):
    """Return the local stiffness matrix of a truss element.

    E: Young's modulus
    A: cross-sectional area
    L: bar length

    Returns Kloc (2x2), ordered as [u_a_local, u_b_local].
    """
    if not np.isfinite([E, A, L]).all() or E <= 0 or A <= 0 or L <= 0:
        raise ValueError("E, A, and L must be finite and positive.")

    return E * A / L * np.array([[1., -1.],
                                 [-1., 1.]])


def transformation_matrix_truss(xa, xb):
    """Return the transformation matrix of a truss element.

    xa: (2,) global endpoint coordinates [x, z] of node a
    xb: (2,) global endpoint coordinates [x, z] of node b

    Returns T (2x4), mapping global to local axial displacements:
    [u_a_local, u_b_local] = T @ [u_a, w_a, u_b, w_b].
    """
    _, c, s = geometry_line_2d(xa, xb)

    return np.array([[c, s, 0., 0.],
                     [0., 0., c, s]])


def stiffness_matrix_truss(E, A, xa, xb):
    """Return the stiffness and transformation matrices of a truss element.

    E:  Young's modulus
    A:  cross-sectional area
    xa: (2,) global endpoint coordinates [x, z] of node a
    xb: (2,) global endpoint coordinates [x, z] of node b

    Returns (Kglob, Kloc, T), with shapes 4x4, 2x2, and 2x4.
    Global displacements are ordered as [u_a, w_a, u_b, w_b].
    """
    L, _, _ = geometry_line_2d(xa, xb)
    Kloc = local_stiffness_matrix_truss(E, A, L)
    T = transformation_matrix_truss(xa, xb)
    Kglob = T.T @ Kloc @ T

    return Kglob, Kloc, T


def axial_force_truss(E, A, xa, xb, de):
    """Return the axial force of a truss element.

    E:  Young's modulus
    A:  cross-sectional area
    xa: (2,) global endpoint coordinates [x, z] of node a
    xb: (2,) global endpoint coordinates [x, z] of node b
    de: (4,) global displacements [u_a, w_a, u_b, w_b]

    Returns the scalar axial force N, positive in tension.
    """
    L, c, s = geometry_line_2d(xa, xb)
    if not np.isfinite([E, A]).all() or E <= 0 or A <= 0:
        raise ValueError("E and A must be finite and positive.")

    de = np.asarray(de, dtype=float)
    if de.shape != (4,) or not np.isfinite(de).all():
        raise ValueError("de must contain four finite element displacements.")

    ua, wa, ub, wb = de
    axial_deformation = c * (ub - ua) + s * (wb - wa)
    axial_force = E * A / L * axial_deformation

    return axial_force
