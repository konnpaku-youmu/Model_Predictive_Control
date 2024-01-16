from given.config import Problem
from scipy.linalg import block_diag
from dataclasses import dataclass
import numpy as np
import numpy.linalg as la

#-----------------------------------------------------------
# Representation for quadratic functions and conversion to single shooting 
#-----------------------------------------------------------

@dataclass
class QuadraticFunction:
    """
    Represent 
    f(x) = 1/2 u'Hu + (G*x0)'u 
    by H and G
    """
    H: np.ndarray
    G: np.ndarray

    def __call__(self, u, x0):
        return 0.5 * u@self.H@u + self.g(x0)@u

    def g(self, x0):
        return self.G@x0

    def grad(self, u, x0):
        return self.H@u + self.g(x0)

    @property 
    def input_dim(self):
        return self.H.shape[1]

def convert_to_single_shooting(problem: Problem) -> QuadraticFunction:

    Qbar = block_diag(*(problem.Q for _ in range(problem.N)), problem.P)
    Rbar = block_diag(*(problem.R for _ in range(problem.N)))

    Bbar = get_Bbar(problem)
    Abar = get_Abar(problem)

    H = Rbar + Bbar.T@Qbar@Bbar 
    G = Bbar.T@Qbar@Abar
    return QuadraticFunction(H, G)

def get_Abar(problem: Problem):
    return np.vstack([la.matrix_power(problem.A, i) for i in range(problem.N+1)])

def get_Bbar(problem: Problem):
    """Compute \bar{B} from the solution document 
    """
    # Convenience function to generate a zero matrix of appropriate dimensions 
    # This is safer than saving the output as a variable and using it everywhere 
    # because that may lead to unexpected behavior: 
    #   we would be filling the matrix with multiple objects all pointing to the same data. 
    O = lambda: np.zeros((problem.ns, problem.nu))
    Bb = [[O() for _ in range(problem.N)]]
    for blkrow in range(problem.N):
        new_row = [
            la.matrix_power(problem.A, i)@problem.B for i in range(blkrow, -1, -1)] \
                + [O() for _ in range(problem.N - blkrow - 1)]
        Bb.append(new_row)
    Bbar = np.block(Bb)
    return Bbar
