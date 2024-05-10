"""Routines for solving a linear system of equations."""

import numpy as np
from numpy.typing import NDArray


def gaussian_eliminate(aa: NDArray[np.float_], bb: NDArray[np.float_]) -> NDArray[np.float_] | None:
    """Solves a linear system of equations (Ax = b) by Gauss-elimination

    Args:
        aa: Matrix with the coefficients. Shape: (n, n).
        bb: Right hand side of the equation. Shape: (n,)

    Returns:
        Vector xx with the solution of the linear equation or None
        if the equations are linearly dependent.
    """
    nn = aa.shape[0]
    xx = np.zeros((nn,), dtype=np.float_)
    return xx
