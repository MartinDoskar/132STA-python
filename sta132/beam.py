"""Elementary planar Euler-Bernoulli beam element functions for 132STA.

The beam element combines axial extension and bending in the local x-z plane.
Members are straight and prismatic, with linear elasticity and small displacements.
Coordinates are [x, z], with x positive to the right and z positive downwards.
Global element displacements are ordered as [u_a, w_a, phi_a, u_b, w_b, phi_b].
Rotations and end moments are positive counterclockwise, so phi = -dw/dx.
Member loads use local components and the undeformed actual member length.
Uniform intensities are qx, qz; midpoint point-force components are fx, fz.
Use a consistent system of units for all inputs and outputs.
"""

import numpy as np

from .geometry import geometry_line_2d


def local_stiffness_matrix_membrane_beam(E, A, L):
    """Return the local membrane stiffness contribution of a beam element.

    E: Young's modulus
    A: cross-sectional area
    L: beam length

    Returns the membrane contribution Kmem (6x6), ordered as
    [u_a_local, w_a_local, phi_a, u_b_local, w_b_local, phi_b].
    Axial entries correspond to u_a_local and u_b_local.
    """
    if not np.isfinite([E, A, L]).all() or E <= 0 or A <= 0 or L <= 0:
        raise ValueError("E, A, and L must be finite and positive.")

    return E * A / L * np.array([[1., 0., 0., -1., 0., 0.],
                                 [0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0.],
                                 [-1., 0., 0., 1., 0., 0.],
                                 [0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0.]])


def local_stiffness_matrix_bending_beam(E, I, L):
    """Return the local bending stiffness contribution of a beam element.

    E: Young's modulus
    I: second moment of area for bending in the local x-z plane
    L: beam length

    Returns the bending contribution Kbend (6x6), ordered as
    [u_a_local, w_a_local, phi_a, u_b_local, w_b_local, phi_b].
    Bending entries correspond to w_a_local, phi_a, w_b_local, phi_b.
    The factor 2EI/L and rotation signs follow Lecture 6.
    """
    if not np.isfinite([E, I, L]).all() or E <= 0 or I <= 0 or L <= 0:
        raise ValueError("E, I, and L must be finite and positive.")

    return 2 * E * I / L * np.array([[0., 0., 0., 0., 0., 0.],
                                     [0., 6/L**2, -3/L, 0., -6/L**2, -3/L],
                                     [0., -3/L, 2., 0., 3/L, 1.],
                                     [0., 0., 0., 0., 0., 0.],
                                     [0., -6/L**2, 3/L, 0., 6/L**2, 3/L],
                                     [0., -3/L, 1., 0., 3/L, 2.]])


def local_stiffness_matrix_beam(E, A, I, L):
    """Return the local stiffness matrix of a beam element.

    E: Young's modulus
    A: cross-sectional area
    I: second moment of area for bending in the local x-z plane
    L: beam length

    Returns Kloc (6x6), the sum of the membrane and bending contributions,
    ordered as [u_a_local, w_a_local, phi_a, u_b_local, w_b_local, phi_b].
    """
    Kmem = local_stiffness_matrix_membrane_beam(E, A, L)
    Kbend = local_stiffness_matrix_bending_beam(E, I, L)

    return Kmem + Kbend


def transformation_matrix_beam(xa, xb):
    """Return the transformation matrix of a beam element.

    xa: (2,) global endpoint coordinates [x, z] of node a
    xb: (2,) global endpoint coordinates [x, z] of node b

    Returns T (6x6), mapping global to local displacements:
    [u_a_local, w_a_local, phi_a, u_b_local, w_b_local, phi_b]
    = T @ [u_a, w_a, phi_a, u_b, w_b, phi_b].
    """
    _, c, s = geometry_line_2d(xa, xb)

    return np.array([[c, s, 0., 0., 0., 0.],
                     [-s, c, 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0.],
                     [0., 0., 0., c, s, 0.],
                     [0., 0., 0., -s, c, 0.],
                     [0., 0., 0., 0., 0., 1.]])


def stiffness_matrix_beam(E, A, I, xa, xb):
    """Return the stiffness and transformation matrices of a beam element.

    E:  Young's modulus
    A:  cross-sectional area
    I:  second moment of area for bending in the local x-z plane
    xa: (2,) global endpoint coordinates [x, z] of node a
    xb: (2,) global endpoint coordinates [x, z] of node b

    Returns (Kglob, Kloc, T), each with shape 6x6.
    Global displacements are ordered as [u_a, w_a, phi_a, u_b, w_b, phi_b].
    """
    L, _, _ = geometry_line_2d(xa, xb)
    Kloc = local_stiffness_matrix_beam(E, A, I, L)
    T = transformation_matrix_beam(xa, xb)
    Kglob = T.T @ Kloc @ T

    return Kglob, Kloc, T


def fixed_end_forces_uniform_load_beam(qx, qz, xa, xb):
    """Return the fixed-end actions and transformation for a uniform beam load.

    qx: uniform force per unit actual member length along local x
    qz: uniform force per unit actual member length along local z
    xa: (2,) global endpoint coordinates [x, z] of node a
    xb: (2,) global endpoint coordinates [x, z] of node b

    Returns (forces_global, forces_local, T), with shapes (6,), (6,), (6, 6).
    Both action vectors are ordered as [X_a, Z_a, M_a, X_b, Z_b, M_b]
    in their respective coordinate systems, with forces_global = T.T @ forces_local.
    T maps global displacements to local displacements.
    These actions hold all endpoint displacements at zero.
    Equivalent applied nodal loads are -forces_global.
    """
    L, _, _ = geometry_line_2d(xa, xb)
    if not np.isfinite([qx, qz]).all():
        raise ValueError("qx and qz must be finite.")

    forces_local = np.array([-qx*L/2, -qz*L/2, qz*L**2/12,
                             -qx*L/2, -qz*L/2, -qz*L**2/12])
    T = transformation_matrix_beam(xa, xb)
    forces_global = T.T @ forces_local

    return forces_global, forces_local, T


def fixed_end_forces_midpoint_load_beam(fx, fz, xa, xb):
    """Return the fixed-end actions and transformation for a midpoint beam force.

    fx: point force at x = L/2 along local x
    fz: point force at x = L/2 along local z
    xa: (2,) global endpoint coordinates [x, z] of node a
    xb: (2,) global endpoint coordinates [x, z] of node b

    Returns (forces_global, forces_local, T), with shapes (6,), (6,), (6, 6).
    Both action vectors are ordered as [X_a, Z_a, M_a, X_b, Z_b, M_b]
    in their respective coordinate systems, with forces_global = T.T @ forces_local.
    T maps global displacements to local displacements.
    These actions hold all endpoint displacements at zero.
    Equivalent applied nodal loads are -forces_global.
    """
    L, _, _ = geometry_line_2d(xa, xb)
    if not np.isfinite([fx, fz]).all():
        raise ValueError("fx and fz must be finite.")

    forces_local = np.array([-fx/2, -fz/2, fz*L/8,
                             -fx/2, -fz/2, -fz*L/8])
    T = transformation_matrix_beam(xa, xb)
    forces_global = T.T @ forces_local

    return forces_global, forces_local, T


def end_forces_beam(E, A, I, xa, xb, de, fixed_end_forces_local=None):
    """Return the end actions and transformation of a beam element.

    E:                      Young's modulus
    A:                      cross-sectional area
    I:                      second moment of area for bending in the local x-z plane
    xa:                     (2,) global endpoint coordinates [x, z] of node a
    xb:                     (2,) global endpoint coordinates [x, z] of node b
    de:                     (6,) global displacements [u_a, w_a, phi_a, u_b, w_b,
                            phi_b]
    fixed_end_forces_local: (6,) local fixed-end actions from member loading; None
                            means zero member loading

    Returns (forces_global, forces_local, T), with shapes (6,), (6,), (6, 6).
    Both action vectors are ordered as [X_a, Z_a, M_a, X_b, Z_b, M_b]
    in their respective coordinate systems, with forces_global = T.T @ forces_local.
    T maps global displacements to local displacements.
    These actions act on the isolated member, including member loading.
    """
    de = np.asarray(de, dtype=float)
    if de.shape != (6,) or not np.isfinite(de).all():
        raise ValueError("de must contain six finite element displacements.")
    if fixed_end_forces_local is None:
        fixed_end_forces_local = np.zeros(6)
    fixed_end_forces_local = np.asarray(fixed_end_forces_local, dtype=float)
    if fixed_end_forces_local.shape != (6,) or not np.isfinite(fixed_end_forces_local).all():
        raise ValueError("fixed_end_forces_local must contain six finite local actions.")

    _, Kloc, T = stiffness_matrix_beam(E, A, I, xa, xb)
    deloc = T @ de

    forces_local = Kloc @ deloc + fixed_end_forces_local
    forces_global = T.T @ forces_local

    return forces_global, forces_local, T
