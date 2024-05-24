"""Contains routines to test the solvers module"""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import pytest
import solvers

# Absolut tolerance when comparing reals
ABS_TOL = 1e-10

# Relative tolerance when comparing reals
REL_TOL = 1e-10

# Where to find the test data (relative to project folder)
TEST_DATA_PATH = Path("test/data")

# Tests which should return a result
SOLVABLE_TESTS = ["elimination_3", "elimination_4", "pivot_3"]

# Tests which should signalize linear dependency
LINEARLY_DEPENDENT_TESTS = ["lindep_3"]


@pytest.mark.parametrize("testname", SOLVABLE_TESTS)
def test_solvable_system(testname: str) -> None:
    """Tests the result of a solvable system"""
    aa, bb = _get_input(testname + ".in")
    xx_expected = _get_expected_output(testname + ".out")
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    assert _check_result(xx_expected, xx_gauss)


@pytest.mark.parametrize("testname", LINEARLY_DEPENDENT_TESTS)
def test_linearly_dependent_system(testname: str) -> None:
    """Tests whether linear dependency is detected"""
    aa, bb = _get_input(testname + ".in")
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    assert _check_result(None, xx_gauss)


def test_lu_decomposition():
    """Tests the LU-decomposition with a 3x3 matrix."""
    aa = np.array([[0.0, 5.0, 22.0 / 3], [4.0, 2.0, 1.0], [2.0, 7.0, 9.0]], dtype=np.float_)
    lu_expected = np.array(
        [[4.0, 2.0, 1.0], [0.5, 6.0, 8.5], [0.0, 5.0 / 6.0, 0.25]], dtype=np.float_
    )
    perm_expected = np.array([1, 2, 0], dtype=np.int_)
    obtained = solvers.lu_decompose(aa)
    assert obtained is not None
    lu_obtained, perm_obtained = obtained
    assert _check_result(lu_expected, lu_obtained)
    assert np.all(perm_expected == perm_obtained)


def test_lindep_lu_decomposition():
    """Tests the LU-decomposition with a linearly dependent 3x3 matrix."""
    aa = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float_)
    obtained = solvers.lu_decompose(aa)
    assert obtained is None


def test_forward_substitution():
    """Tests the forward substitution routine"""
    lu = np.array(
        [[4.0, 2.0, 1.0], [0.5, 6.0, 8.5], [0.0, 0.833333333333333, 0.25]], dtype=np.float_
    )
    bb = np.array([3.0, -3.0, 6.0], dtype=np.float_)
    expected = np.array([3.0, -4.5, 9.75])
    obtained = solvers.forward_substitute(lu, bb)
    assert np.allclose(expected, obtained, atol=ABS_TOL, rtol=REL_TOL)


def test_backwards_substitution():
    """Tests the backward substitution routine"""
    lu = np.array(
        [[4.0, 2.0, 1.0], [0.5, 6.0, 8.5], [0.0, 0.833333333333333, 0.25]], dtype=np.float_
    )
    bb = np.array([3.0, -4.5, 9.75], dtype=np.float_)
    expected = np.array([19.0, -56.0, 39.0], dtype=np.float_)
    obtained = solvers.backward_substitute(lu, bb)
    assert np.allclose(expected, obtained, atol=ABS_TOL, rtol=REL_TOL)


def _get_input(infile: str) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Reads test input data from a file"""
    data = np.loadtxt(TEST_DATA_PATH / infile)
    nn = data.shape[1]
    return data[:nn, :], data[nn, :]


def _get_expected_output(outfile: str) -> NDArray[np.float_]:
    """Reads the expected test output from a file"""
    return np.loadtxt(TEST_DATA_PATH / outfile)


def _check_result(expected: NDArray[np.float_] | None, obtained: NDArray[np.float_] | None) -> bool:
    """Checks whether expected and obtained results match."""
    result_ok = False
    if expected is None and obtained is None:
        result_ok = True
    elif expected is not None and obtained is not None:
        result_ok = np.allclose(obtained, expected, atol=ABS_TOL, rtol=REL_TOL)
    return result_ok
