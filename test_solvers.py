#!/usr/bin/env python3
"""Contains routines to test the solvers module"""

import numpy as np
from numpy.typing import NDArray
import solvers


def main() -> None:
    """Main testing function."""

    print("\nTest elimination")
    test_elimination_3()
    print("\nTest pivot")
    test_pivot_3()
    print("\nTest linear dependency")
    test_lindep_3()


def test_elimination_3() -> None:
    """Tests elimination with 3 variables."""
    aa = np.array([[2.0, 4.0, 4.0], [5.0, 4.0, 2.0], [1.0, 2.0, -1.0]], dtype=np.float_)
    bb = np.array([1.0, 4.0, 2.0], dtype=np.float_)
    xx_expected = np.array([0.666666666666667, 0.416666666666667, -0.5], dtype=np.float_)
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    _check_result(xx_expected, xx_gauss)


def test_pivot_3() -> None:
    """Tests a solution with 3 variables, where pivot is necessary."""
    aa = np.array([[2.0, 4.0, 4.0], [1.0, 2.0, -1.0], [5.0, 4.0, 2.0]])
    bb = np.array([1.0, 2.0, 4.0])
    xx_expected = np.array([0.666666666666667, 0.416666666666667, -0.5])
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    _check_result(xx_expected, xx_gauss)


def test_lindep_3() -> None:
    """Tests a linearly dependent system wit three variables."""
    aa = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    bb = np.array([1.0, 2.0, 3.0])
    xx_expected = None
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    _check_result(xx_expected, xx_gauss)


def _check_result(expected: NDArray[np.float_] | None, obtained: NDArray[np.float_] | None) -> None:
    """Checks results by printing expected and obtained one."""
    print(f"Expected: {expected}")
    print(f"Obtained: {obtained}")


if __name__ == "__main__":
    main()
