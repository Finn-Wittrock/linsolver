"""Routines for solving a linear system of equations."""

import numpy as np
from numpy.typing import NDArray

# Tolerance value for dependency check
_DEPENDENCY_TOL = 1e-10


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
    for ii in range(nn):

        # Partial pivot
        imax = np.argmax(np.abs(aa[ii:, ii])) + ii
        _swap_rows(aa, ii, imax)
        _swap_rows(bb, ii, imax)
        # Alternatively, you might use fancy indexing to enforce copy
        # aa[[ii, imax]] = aa[[imax, ii]]
        # bb[[ii, imax]] = bb[[imax, ii]]

        # Dependency check
        # Note: the diagonal element in the last row (aa[nn - 1, nn - 1]) must be checked as well
        # in order to ensure the back substitution below to work. Therefore, the loop now runs
        # over all rows including the last one.
        if np.abs(aa[ii, ii]) < _DEPENDENCY_TOL:
            return None

        for jj in range(ii + 1, nn):
            coeff = -aa[jj, ii] / aa[ii, ii]
            aa[jj, ii:] += coeff * aa[ii, ii:]
            bb[jj] += coeff * bb[ii]

    xx = np.zeros((nn,), dtype=np.float_)
    for ii in range(nn - 1, -1, -1):
        # Note: dot product of two arrays of zero size is 0.0
        xx[ii] = (bb[ii] - aa[ii, ii + 1 :] @ xx[ii + 1 :]) / aa[ii, ii]
    return xx


def lu_decompose(aa: NDArray[np.float_]) -> tuple[NDArray[np.float_], NDArray[np.int_]] | None:
    """Decomposes a non-singular quadratic matrix into LU-form with partial pivoting.

    Args:
        aa: Matrix to decompose, contains LU-decomposed form on exit.

    Return:
        Tuple containing the LU-decomposed A matrix and the permutation vector of the row pivots.
        If matrix was linearly dependent, None is returned.
    """
    lu = aa.copy()
    nn = lu.shape[0]
    perm = np.arange(nn, dtype=np.int_)
    swapbuffer = np.arange(nn, dtype=np.float_)

    for ii in range(nn):
        # Partial pivot
        imax = np.argmax(np.abs(lu[ii:, ii])) + ii
        if imax != ii:
            swapbuffer[:] = lu[ii]
            lu[ii] = lu[imax]
            lu[imax] = swapbuffer
            perm[ii], perm[imax] = perm[imax], perm[ii]

        # Dependency check
        if np.abs(lu[ii, ii]) < _DEPENDENCY_TOL:
            return None

        for jj in range(ii + 1, nn):
            coeff = lu[jj, ii] / lu[ii, ii]
            lu[jj, ii + 1 :] -= coeff * lu[ii, ii + 1 :]
            lu[jj, ii] = coeff

    return lu, perm


def forward_substitute(lu: NDArray[np.float_], bb: NDArray[np.float_]) -> NDArray[np.float_]:
    """Solve L @ y = b for a lower triangle matrix via forward substitution.

    Args:
        lu: LU-decomposed matrix (as returned by lu_decompose()) containig lower triangle matrix L.
        bb: Right hand side of the equation.

    Returns:
        The solution vector y.
    """
    nn = lu.shape[0]
    solution = np.zeros((nn,), dtype=np.float_)
    for ii in range(nn):
        solution[ii] = bb[ii] - lu[ii, :ii] @ solution[:ii]
    return solution


def backward_substitute(lu: NDArray[np.float_], bb: NDArray[np.float_]) -> NDArray[np.float_]:
    """Solve U @ x = b for an upper triangle matrix via backward substitution.

    Args:
        lu: LU-decomposed matrix (as returned by lu_decompose()) containig upper triangle matrix U.
        bb: Right hand side of the equation.

    Returns:
        The solution vector x.
    """
    nn = lu.shape[0]
    solution = np.zeros((nn,), dtype=np.float_)
    for ii in range(nn - 1, -1, -1):
        solution[ii] = (bb[ii] - lu[ii, ii + 1 :] @ solution[ii + 1 :]) / lu[ii, ii]
    return solution


def _swap_rows(mtx, irow1, irow2):
    """Swaps two rows of an array."""
    if irow1 == irow2:
        return
    tmp = mtx[irow1].copy()
    mtx[irow1] = mtx[irow2]
    mtx[irow2] = tmp
