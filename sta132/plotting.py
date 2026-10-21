"""Simple visualisation functions for 132STA truss exercises."""

import matplotlib.pyplot as plt
import numpy as np


def plot_truss(xz, connectivity, d, scale=1.0):
    """Plot the original and magnified deformed shape of a 2D truss.

    xz:           (n_nodes, 2) coordinates [x, z] in metres
    connectivity: (n_elements, 2) node indices, STARTING AT ZERO
    d:            1D vector [u1, w1, u2, w2, ...] in metres
    scale:        magnification of plotted displacements

    Returns (fig, ax), so users can add annotations or save the figure.
    """
    xz = np.asarray(xz, dtype=float)
    connectivity = np.asarray(connectivity, dtype=int)
    d = np.asarray(d, dtype=float)

    if xz.ndim != 2 or xz.shape[1] != 2:
        raise ValueError("xz must have two columns: x and z.")
    if connectivity.ndim != 2 or connectivity.shape[1] != 2:
        raise ValueError("connectivity must contain pairs of node indices.")
    if d.shape != (2 * len(xz),):
        raise ValueError("d must contain two displacements per node.")
    if np.any(connectivity < 0) or np.any(connectivity >= len(xz)):
        raise ValueError("connectivity contains an invalid node index.")

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
