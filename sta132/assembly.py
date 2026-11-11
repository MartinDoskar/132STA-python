"""Code-number helpers for structural assembly in 132STA."""

import numpy as np


def code_numbers_from_nodes(node_numbers, dofs_per_node):
    """Return the full code numbers for an ordered list of nodes.

    node_numbers:  (n_nodes,) node numbers, STARTING AT ONE, in endpoint order
    dofs_per_node: number of displacement components per node; 3 for a frame

    Returns a 1D integer array of full code numbers, STARTING AT ONE.
    For nodes [3, 1] with three components per node, the result is
    [7, 8, 9, 1, 2, 3]. Subtract 1 when forming Python array indices.
    """
    if not isinstance(dofs_per_node, (int, np.integer)) or dofs_per_node <= 0:
        raise ValueError("dofs_per_node must be a positive integer.")

    node_numbers = np.asarray(node_numbers)
    if node_numbers.ndim != 1:
        raise ValueError("node_numbers must be a 1D list of node numbers.")
    if node_numbers.size == 0:
        return np.array([], dtype=int)
    if (not np.issubdtype(node_numbers.dtype, np.integer)
        or np.any(node_numbers < 1)):
        raise ValueError("node_numbers must be positive integers starting at 1.")

    node_numbers = node_numbers.astype(int, copy=False)
    node_offsets = dofs_per_node * (node_numbers - 1)
    local_code_numbers = np.arange(1, dofs_per_node + 1)

    return (node_offsets[:, None] + local_code_numbers).ravel()
