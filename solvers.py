"""Routines for solving a linear system of equations."""

import numpy as np
from numpy.typing import NDArray

# Tolerance value for dependency check
_DEPENDENCY_TOL = 1e-10


def solve(aa: NDArray[np.float_], bb: NDArray[np.float_]) -> NDArray[np.float_]:
    """Solves a linear system of equations (:math:`Ax = b`) by Gauss-elimination

    Args:
        aa: Matrix with the coefficients. Shape: (n, n).
        bb: Right hand side of the equation. Shape: (n,).

    Returns:
        Vector xx with the solution of the linear equation or None if the equations are linearly
        dependent.

    Raises:
        ValueError: if the matrix aa is linearly dependent.
    """
    decomp = lu_decompose(aa)
    lu, perm = decomp
    yy = forward_substitute(lu, bb[perm])
    xx = backward_substitute(lu, yy)
    return xx


def lu_decompose(aa: NDArray[np.float_]) -> tuple[NDArray[np.float_], NDArray[np.int_]]:
    """Decomposes a non-singular quadratic matrix into LU-form with partial pivoting.

    Args:
        aa: Matrix to decompose, contains LU-decomposed form on exit.

    Return:
        Tuple containing the LU-decomposed A matrix and the permutation vector of the row pivots.

    Raises:
        ValueError: if matrix was linearly dependent.
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
            raise ValueError("Linear dependency detected")

        for jj in range(ii + 1, nn):
            coeff = lu[jj, ii] / lu[ii, ii]
            lu[jj, ii + 1 :] -= coeff * lu[ii, ii + 1 :]
            lu[jj, ii] = coeff

    return lu, perm


def forward_substitute(lu: NDArray[np.float_], bb: NDArray[np.float_]) -> NDArray[np.float_]:
    """Solves :math:`L y = b` for a lower triangle matrix via forward substitution.

    Args:
        lu: LU-decomposed matrix (as returned by lu_decompose()) containig lower triangle matrix L.
        bb: Right hand side of the equation.

    Returns:
        The solution vector :math:`y`.
    """
    nn = lu.shape[0]
    solution = np.zeros((nn,), dtype=np.float_)
    for ii in range(nn):
        solution[ii] = bb[ii] - lu[ii, :ii] @ solution[:ii]
    return solution


def backward_substitute(lu: NDArray[np.float_], bb: NDArray[np.float_]) -> NDArray[np.float_]:
    """Solves :math:`U  x = b` for an upper triangle matrix via backward substitution.

    Args:
        lu: LU-decomposed matrix (as returned by lu_decompose()) containig upper triangle matrix U.
        bb: Right hand side of the equation.

    Returns:
        The solution vector :math:`x`.
    """
    nn = lu.shape[0]
    solution = np.zeros((nn,), dtype=np.float_)
    for ii in range(nn - 1, -1, -1):
        solution[ii] = (bb[ii] - lu[ii, ii + 1 :] @ solution[ii + 1 :]) / lu[ii, ii]
    return solution
