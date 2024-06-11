import numpy as np
from numpy.typing import NDArray
import solvers
import time

def mk_random_matrices(dim : int = 3) -> tuple[NDArray[np.float_], NDArray[np.float_]] :

    rdmmatrix = np.zeros((dim,dim))
    for i in range(dim):
        rdmmatrix[i] = np.random.random(dim)
    rdmvector = np.random.random(dim)
    return rdmmatrix, rdmvector

def solve_rdm_linequations(maxdim : int, amountequations : int) -> tuple[list[float], list[int]]:
    timetosolve = []
    equationdim = []
    equationcount = 0
    for dim in np.linspace(2,maxdim, amountequations):
        equationcount += 1
        dim = int(dim)
        rdmmatrix, rdmvector = mk_random_matrices(dim)
        timestart = time.time()
        solvers.solve(rdmmatrix, rdmvector)
        timeend = time.time()
        timetosolve.append(timeend - timestart)
        equationdim.append(timetosolve)
        print(f"Equation{equationcount} with dim = {dim} took {timeend -timestart} seconds to solve.")
    return timetosolve, equationdim






