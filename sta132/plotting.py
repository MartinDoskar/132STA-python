"""Simple visualisation functions for 132STA truss and frame exercises."""

import matplotlib.pyplot as plt
import numpy as np

from .beam import transformation_matrix_beam
from .geometry import geometry_line_2d


def plot_truss(xz, connectivity, d, scale=1.0):
    """Plot the original and magnified deformed shape of a 2D truss.

    xz:           (n_nodes, 2) coordinates [x, z] in metres
    connectivity: (n_elements, 2) integer node indices, STARTING AT ZERO
    d:            1D vector [u1, w1, u2, w2, ...] in metres
    scale:        magnification of plotted displacements

    Returns (fig, ax), so users can add annotations or save the figure.
    """
    xz = np.asarray(xz, dtype=float)
    connectivity = np.asarray(connectivity)
    d = np.asarray(d, dtype=float)

    if xz.ndim != 2 or xz.shape[1] != 2:
        raise ValueError("xz must have two columns: x and z.")
    if connectivity.ndim != 2 or connectivity.shape[1] != 2:
        raise ValueError("connectivity must contain pairs of node indices.")
    if not np.issubdtype(connectivity.dtype, np.integer):
        raise ValueError("connectivity must contain integer node indices, starting at 0.")
    if d.shape != (2 * len(xz),):
        raise ValueError("d must contain two displacements per node.")
    if np.any(connectivity < 0) or np.any(connectivity >= len(xz)):
        raise ValueError(f"connectivity indices must be between 0 and {len(xz) - 1}.")
    connectivity = connectivity.astype(int, copy=False)

    if not np.isfinite(xz).all() or not np.isfinite(d).all() or not np.isfinite(scale):
        raise ValueError("Coordinates, displacements, and scale must be finite.")

    moved = xz + scale * d.reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(8, 5))

    for e, (a, b) in enumerate(connectivity):
        # Original truss (dashed), deformed truss (solid).
        ax.plot(xz[[a, b], 0], xz[[a, b], 1], '--', color='0.60', lw=2,
                label='Original' if e == 0 else None)
        ax.plot(moved[[a, b], 0], moved[[a, b], 1], '-o', color='#246C9C', lw=2,
                markersize=5, label='Deformed' if e == 0 else None)

    for i, (x, z) in enumerate(xz):
        ax.text(x - 0.16, z - 0.16, str(i + 1), fontsize=10)

    ax.set_title(f'Truss deformation (magnified {scale:g}x)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m] (positive downwards)')
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_frame(xz, connectivity, d, scale=1.0):
    """Plot the original and interpolated deformed shape of a 2D frame.

    xz:           (n_nodes, 2) coordinates [x, z] in metres
    connectivity: (n_elements, 2) integer node indices, STARTING AT ZERO
    d:            1D vector [u1, w1, phi1, u2, w2, phi2, ...]; metres and radians
    scale:        magnification of plotted displacements

    Returns (fig, ax), so users can add annotations or save the figure.
    Uses linear axial and cubic Hermite transverse interpolation of nodal values.
    Member-load deformation with zero endpoint motion is omitted.
    """
    xz = np.asarray(xz, dtype=float)
    connectivity = np.asarray(connectivity)
    d = np.asarray(d, dtype=float)

    if xz.ndim != 2 or xz.shape[1] != 2:
        raise ValueError("xz must have two columns: x and z.")
    if connectivity.ndim != 2 or connectivity.shape[1] != 2:
        raise ValueError("connectivity must contain pairs of node indices.")
    if not np.issubdtype(connectivity.dtype, np.integer):
        raise ValueError("connectivity must contain integer node indices, starting at 0.")
    if d.shape != (3 * len(xz),):
        raise ValueError("d must contain three displacement components per node.")
    if np.any(connectivity < 0) or np.any(connectivity >= len(xz)):
        raise ValueError(f"connectivity indices must be between 0 and {len(xz) - 1}.")
    connectivity = connectivity.astype(int, copy=False)
    if not np.isfinite(xz).all() or not np.isfinite(d).all() or not np.isfinite(scale):
        raise ValueError("Coordinates, displacements, and scale must be finite.")

    nodal_displacements = d.reshape(-1, 3)
    xi = np.linspace(0., 1., 51)
    fig, ax = plt.subplots(figsize=(8, 5))

    for e, (a, b) in enumerate(connectivity):
        L, c, s = geometry_line_2d(xz[a], xz[b])
        T = transformation_matrix_beam(xz[a], xz[b])
        de = nodal_displacements[[a, b]].ravel()
        ua, wa, phia, ub, wb, phib = T @ de

        # Linear axial and cubic Hermite transverse displacement.
        u = (1 - xi) * ua + xi * ub
        H1 = 1 - 3 * xi**2 + 2 * xi**3
        H2 = L * (-xi + 2 * xi**2 - xi**3)
        H3 = 3 * xi**2 - 2 * xi**3
        H4 = L * (xi**2 - xi**3)
        w = H1 * wa + H2 * phia + H3 * wb + H4 * phib

        # Transform local displacements back to the global x-z plane.
        points = xz[a] + xi[:, None] * (xz[b] - xz[a])
        moved = points + scale * np.column_stack((c * u - s * w, s * u + c * w))
        ax.plot(xz[[a, b], 0], xz[[a, b], 1], '--', color='0.60', lw=2,
                label='Original' if e == 0 else None)
        ax.plot(moved[:, 0], moved[:, 1], '-', color='#246C9C', lw=2,
                label='Interpolated' if e == 0 else None)
        ax.plot(moved[[0, -1], 0], moved[[0, -1], 1], 'o', color='#246C9C', markersize=5)

    for i, (x, z) in enumerate(xz):
        ax.text(x - 0.16, z - 0.16, str(i + 1), fontsize=10)

    ax.set_title(f'Frame deformation (Hermite interpolation, magnified {scale:g}x)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m] (positive downwards)')
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, ax
