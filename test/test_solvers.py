#!/usr/bin/env python3
"""Contains routines to test the solvers module"""

import numpy as np
from numpy.typing import NDArray
import solvers

# Absolut tolerance when comparing reals
ABS_TOL = 1e-10

# Relative tolerance when comparing reals
REL_TOL = 1e-10


def test_elimination_3() -> None:
    """Tests elimination with 3 variables."""
    aa = np.array([[2.0, 4.0, 4.0], [5.0, 4.0, 2.0], [1.0, 2.0, -1.0]], dtype=np.float_)
    bb = np.array([1.0, 4.0, 2.0], dtype=np.float_)
    xx_expected = np.array([0.666666666666667, 0.416666666666667, -0.5], dtype=np.float_)
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    assert _check_result(xx_expected, xx_gauss)


def test_elimination_4() -> None:
    """Tests elimination with 4 variables."""
    aa = np.array(
        [[2.0, 7.0, 8.0, 3.0], [0.0, 7.0, 6.0, 9.0], [0.0, 8.0, 5.0, 2.0], [1.0, 5.0, 9.0, 4.0]],
        dtype=np.float_,
    )
    bb = np.array([8.0, 5.0, 8.0, 2.0], dtype=np.float_)
    xx_expected = np.array([2.03125, 1.53125, -0.8125, -0.09375], dtype=np.float_)
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    assert _check_result(xx_expected, xx_gauss)


def test_pivot_3() -> None:
    """Tests a solution with 3 variables, where pivot is necessary."""
    aa = np.array([[2.0, 4.0, 4.0], [1.0, 2.0, -1.0], [5.0, 4.0, 2.0]])
    bb = np.array([1.0, 2.0, 4.0])
    xx_expected = np.array([0.666666666666667, 0.416666666666667, -0.5])
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    assert _check_result(xx_expected, xx_gauss)


def test_lindep_3() -> None:
    """Tests a linearly dependent system wit three variables."""
    aa = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    bb = np.array([1.0, 2.0, 3.0])
    xx_expected = None
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    assert _check_result(xx_expected, xx_gauss)


def _check_result(expected: NDArray[np.float_] | None, obtained: NDArray[np.float_] | None) -> bool:
    """Checks whether expected and obtained results match."""
    result_ok = False
    if expected is None and obtained is None:
        result_ok = True
    elif expected is not None and obtained is not None:
        result_ok = np.allclose(obtained, expected, atol=ABS_TOL, rtol=REL_TOL)
    return result_ok
