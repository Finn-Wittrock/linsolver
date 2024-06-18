
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import pytest
import solvers
DATA_PATH = Path()
def read_input(infile: str) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Reads input data from a file"""
    data = np.loadtxt(DATA_PATH / infile)
    nn = data.shape[1]
    return data[:nn, :], data[nn, :]

def write_result(outfile: str, outmatrix: NDArray[np.float_]):
    """Saves output data to a file"""
    np.savetxt(DATA_PATH / outfile, outmatrix)

aa, bb = read_input("linsolver.in")
solved = solvers.solve(aa, bb)
write_result("linsolver.out", solved)
               
                 