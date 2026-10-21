"""Elementary planar truss element functions for 132STA.

Coordinates are [x, z], with x positive to the right and z positive downwards.
Global element displacements are ordered as [u_a, w_a, u_b, w_b].
Use a consistent system of units for all inputs and outputs.
"""

import numpy as np


def geometry_truss(xa, xb):
    """Return the bar length and direction cosines from endpoints a and b.

    xa and xb contain the endpoint coordinates [x, z].
    Returns L, c, s, where c and s are the direction cosines from a to b.
    """
    dx, dz = np.asarray(xb, dtype=float) - np.asarray(xa, dtype=float)
    L = np.hypot(dx, dz)
    if L == 0:
        raise ValueError("The two nodes of a bar must not coincide.")

    return L, dx / L, dz / L


def local_stiffness_matrix_truss(E, A, L):
    """Return the local stiffness matrix of a truss element.

    E is Young's modulus, A is the cross-sectional area, and L is the bar length.
    Returns a 2x2 matrix for the local axial displacements [u_a_local, u_b_local].
    """
    if E <= 0 or A <= 0 or L <= 0:
        raise ValueError("E, A, and L must be positive.")

    return E * A / L * np.array([[1., -1.],
                                 [-1., 1.]])


def transformation_matrix_truss(xa, xb):
    """Return the transformation matrix of a truss element.

    xa and xb contain the endpoint coordinates [x, z].
    Returns T (2x4), mapping global displacements to local axial displacements:
    [u_a_local, u_b_local] = T @ [u_a, w_a, u_b, w_b].
    """
    _, c, s = geometry_truss(xa, xb)

    return np.array([[c, s, 0., 0.],
                     [0., 0., c, s]])


def stiffness_matrix_truss(E, A, xa, xb):
    """Return the stiffness and transformation matrices of a truss element.

    E is Young's modulus and A is the cross-sectional area.
    xa and xb contain the endpoint coordinates [x, z].
    Returns Kglob (4x4), Kloc (2x2), and T (2x4), with global displacements
    ordered as [u_a, w_a, u_b, w_b].
    """
    L, _, _ = geometry_truss(xa, xb)
    Kloc = local_stiffness_matrix_truss(E, A, L)
    T = transformation_matrix_truss(xa, xb)
    Kglob = T.T @ Kloc @ T

    return Kglob, Kloc, T


def axial_force_truss(E, A, xa, xb, de):
    """Return the axial force of a truss element.

    E is Young's modulus and A is the cross-sectional area.
    xa and xb contain the endpoint coordinates [x, z].
    de contains the global element displacements [u_a, w_a, u_b, w_b].
    Returns the scalar axial force N, positive in tension.
    """
    L, c, s = geometry_truss(xa, xb)
    if E <= 0 or A <= 0:
        raise ValueError("E and A must be positive.")

    ua, wa, ub, wb = np.asarray(de, dtype=float)
    elongation = c * (ub - ua) + s * (wb - wa)
    axialForce = E * A / L * elongation

    return axialForce
