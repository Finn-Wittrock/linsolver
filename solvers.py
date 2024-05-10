"""Routines for solving a linear system of equations."""

import numpy as np
from numpy.typing import NDArray


def gaussian_eliminate(aa: NDArray[np.float_], bb: NDArray[np.float_]) -> NDArray[np.float_] | None:
    """Solves a linear system of equations (Ax = b) by Gauss-elimination

    Args:
        aa: Matrix with the coefficients. Shape: (n, n). Note: aa is modified and contains the
            eliminated upper triangular matrix after the function call.
        bb: Right hand side of the equation. Shape: (n,). Note: bb is modified and contains the
            right hand side corresponding to the modified matrix aa after the function call.

    Returns:
        Vector xx with the solution of the linear equation or None if the equations are linearly
        dependent.
    """
    nn = aa.shape[0]
    for ii in range(nn - 1):
        # Partial pivot
        imax = np.argmax(np.abs(aa[ii:, ii])) + ii
        _swap_rows(aa, ii, imax)
        _swap_rows(bb, ii, imax)
        # Alternatively, you might use fancy indexing to enforce copy
        # aa[[ii, imax]] = aa[[imax, ii]]
        # bb[[ii, imax]] = bb[[imax, ii]]
        for jj in range(ii + 1, nn):
            coeff = -aa[jj, ii] / aa[ii, ii]
            aa[jj, ii:] += coeff * aa[ii, ii:]
            bb[jj] += coeff * bb[ii]

    xx = np.zeros((nn,), dtype=np.float_)
    for ii in range(nn - 1, -1, -1):
        # Note: dot product of two arrays of zero size is 0.0
        xx[ii] = (bb[ii] - aa[ii, ii + 1 :] @ xx[ii + 1 :]) / aa[ii, ii]
    return xx


def _swap_rows(mtx, irow1, irow2):
    """Swaps two rows of an array."""
    if irow1 == irow2:
        return
    tmp = mtx[irow1].copy()
    mtx[irow1] = mtx[irow2]
    mtx[irow2] = tmp
