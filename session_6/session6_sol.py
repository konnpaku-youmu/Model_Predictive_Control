from typing import Callable
from given.config import Problem

from scipy.linalg import block_diag
import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt 

from dataclasses import dataclass, field
import os
WORKING_DIR = os.path.split(__file__)[0]

from rcracers.simulator import simulate

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

#-----------------------------------------------------------
# Implementation of projected gradient 
#-----------------------------------------------------------

@dataclass
class OptimizerStats:
    costs: list = field(default_factory= list)
    optimal_value: float = np.nan 
    minimizer: np.ndarray = np.nan


@dataclass
class Box:
    u_min: np.ndarray
    u_max: np.ndarray 

    def Π(self, u):
        """Projection of point u onto self"""
        return np.maximum(self.u_min, np.minimum(self.u_max, u))


def projected_gradient(f, df, γ: float, U: Box, *, max_it: int = 200):
    """Projected gradient algorithm.

    Args:
        f (Callable): Cost function Rⁿ -> R
        df (_type_): Gradient of the cost Rⁿ -> Rⁿ
        γ (float): Step size
        U (Box): Box (constraints)
        max_it (int): Maximum number of iterations
    """

    results = OptimizerStats()

    u = np.zeros_like(U.u_min)
    results.costs.append(f(u))
    
    for _ in range(max_it):
        # Projected gradient step 
        u = U.Π(u - γ * df(u))
    
        # Update the results 
        results.costs.append(f(u))

    results.minimizer = u 
    results.optimal_value = f(u)
    
    return results


def compute_nesterov_const(f: QuadraticFunction):
    eigenvalues = np.abs(la.eigvalsh(f.H))
    λ_max = max(eigenvalues)
    λ_min = min(eigenvalues)
    κ = λ_max/λ_min
    sqrt_κ = np.sqrt(κ)
    return (sqrt_κ-1)/(sqrt_κ+1)

def projected_gradient_nesterov(f, df, γ: float, U: Box, α: float, *, max_it: int = 200):
    """Projected gradient algorithm.

    Args:
        f (Callable): Cost function Rⁿ -> R
        df (Callable): Gradient of the cost Rⁿ -> Rⁿ
        γ (float): Step size
        U (Box): Box (constraints)
        α (float): Nesterov acceleration coefficient 
        max_it (int): Maximum number of iterations
    """

    results = OptimizerStats()

    u = np.zeros_like(U.u_min)
    results.costs.append(f(u))
    ν = u 
    for _ in range(max_it):
        # Projected gradient step 
        u_next = U.Π(ν - γ * df(ν))
        ν = u_next + α*(u_next-u)  # Nesterov acceleration 

        # Update the results 
        u = u_next 
        results.costs.append(f(u))

    results.minimizer = u 
    results.optimal_value = f(u)
    
    return results


def projected_gradient_linesearch(f, df, γ0: float, U: Box, *, max_it: int = 200):
    """Projected gradient algorithm.

    Args:
        f (Callable): Cost function Rⁿ -> R
        df (_type_): Gradient of the cost Rⁿ -> Rⁿ
        γ0 (float): Initial step size 
        U (Box): Box (constraints)
        max_it (int): Maximum number of iterations
    """

    results = OptimizerStats()

    u = np.zeros_like(U.u_min)
    results.costs.append(f(u))
    γ = γ0 
    for _ in range(max_it):
        
        # Line search |----
        while True:
            u_next = U.Π(u - γ*df(u))
            if (f(u_next) <= f(u) + df(u)@(u_next-u) + 1/(2*γ)*np.sum((u_next-u)**2)):
                break
            else:
                γ /= 2
        # ---| Line search 

        # Update the results 
        u = u_next 
        results.costs.append(f(u))

    results.minimizer = u 
    results.optimal_value = f(u)
    
    return results


def ama(f: Callable, H_inv: np.ndarray, g: np.ndarray, L: np.ndarray, project: Callable, γ: float, *, max_it: int):
    """Run the AMA algorithm

    Args:
        f (Callable): Cost function 
        H_inv (np.ndarray): Inverse of hessian of the cost function 
        g (np.ndarray): linear term of the cost function 
        L (np.ndarray): Linear constraint map 
        project (Callable): Method to project the dual variables on their constraints 
        max_it (int): Maximum number of iterations 

    Returns:
        OptimizerStats
    """

    result = OptimizerStats()
    y = np.zeros(L.shape[0]) #initial dual vectors
    w = np.zeros(L.shape[0])
    

    for _ in range(max_it):

        # z-update
        z = -H_inv@(g+L.T@y)

        # w-update
        w = project(L@z+y/γ)
        
        # y-update
        y = y + γ*(L@z - w)

        result.costs.append(f(z))

    result.optimal_value = f(z)
    result.minimizer = z
    return result


def print_pen_and_paper(i):
    print(f"Exercise {i}: Pen and paper")


def plot_convergence_rate(results, **style):
    subopt =  np.abs((np.array(results.costs)[0:len(results.costs)//2] - results.optimal_value)/results.optimal_value)
    plt.semilogy(subopt, **style)
    plt.title("Convergence rate")
    plt.xlabel("Iterations")
    plt.ylabel("Relative suboptimality")

def simulate_results(problem: Problem, results: OptimizerStats) -> np.ndarray:
    print("-- Rollout the found solution")
    def policy(y, t): 
        return results.minimizer[t]
    
    def dynamics(x,u):
        return problem.A @ x + problem.B @ np.atleast_1d(u)
    
    print("-- Calling the simulation utility from rcracers")
    x = simulate(problem.x0, dynamics, len(results.minimizer), policy=policy)
    return x 


def set_the_stage():
    print("-- Instantiate problem")
    problem = Problem(N=30)
    # Convert the problem to single shooting.
    print("-- Convert cost function")
    cost_function: QuadraticFunction = convert_to_single_shooting(problem)

    # Build the constraint set
    print("-- Build constraint set")
    U = Box(problem.u_min*np.ones(cost_function.input_dim), 
            problem.u_max*np.ones(cost_function.input_dim))
    
    # Define the cost function 
    print("-- Defining the cost function")
    def f(u):
        return cost_function(u, problem.x0)
    
    # Define the gradient of the cost
    print("-- Defining the gradient")
    def df(u):
        return cost_function.grad(u, problem.x0)
    return problem, cost_function, f, df, U


#-----------------------------------------------------------
# Exercises 
#-----------------------------------------------------------

def exercise1():
    print_pen_and_paper(1)

def exercise2():
    print_pen_and_paper(2)

def exercise3():
    print("Exercise 3")
    
    problem, cost_function, f, df, U = set_the_stage()
    # Compute the Lipschitz constant of the gradient 
    Lf = la.norm(cost_function.H, ord=2) # Matrix 2-norm: i.e., max eigenvalue is the Lipschitz constant

    print("-- Running the solver")
    results = projected_gradient(f, df, 1./Lf, U)
    
    plt.figure()
    plot_convergence_rate(results)

    x = simulate_results(problem, results)
    plt.figure()
    plt.plot(x[:,0])
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.title("Predicted state trajectory under optimal solution (Proj. grad.)")
    plt.show()

def exercise4(): 
    print("Exercise 4")
    problem, cost_function, f, df, U = set_the_stage()

    α = compute_nesterov_const(cost_function)

    # Compute the Lipschitz constant of the gradient 
    Lf = la.norm(cost_function.H, ord=2) # Matrix 2-norm: i.e., max eigenvalue is the Lipschitz constant

    print("-- Running the solver")
    
    results_projgrad = projected_gradient(f, df, 1./Lf, U)
    results_nesterov = projected_gradient_nesterov(f, df, 1./Lf, U, α)
    
    plt.figure()
    plot_convergence_rate(results_projgrad, label="Proj. gradient")
    plot_convergence_rate(results_nesterov, label="Nesterov")
    plt.legend()

    x_nest = simulate_results(problem, results_nesterov)
    x_proj = simulate_results(problem, results_projgrad)
    plt.figure()
    plt.plot(x_nest[:,0], label="Nesterov")
    plt.plot(x_proj[:,0], label="Proj. grad.")
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.legend()
    plt.title("Predicted state trajectory under optimal solution")
    plt.show()

    
def exercise5():
    print("Exercise 5")
    problem, cost_function, f, df, U = set_the_stage()

    # Compute the Lipschitz constant of the gradient (only NECESSARY for proj. gradient and nesterov)
    Lf = la.norm(cost_function.H, ord=2) # Matrix 2-norm: i.e., max eigenvalue is the Lipschitz constant

    print("-- Running the solver")
    
    results_projgrad = projected_gradient(f, df, 1./Lf, U)
    α = compute_nesterov_const(cost_function)
    results_nesterov = projected_gradient_nesterov(f, df, 1./Lf, U, α)
    results_linesearch = projected_gradient_linesearch(f, df, 100./Lf, U)
    
    plt.figure()
    plot_convergence_rate(results_projgrad, label="Proj. Grad.")
    plot_convergence_rate(results_nesterov, label="Nesterov")
    plot_convergence_rate(results_linesearch, label="Line search")
    plt.legend()

    x_nest = simulate_results(problem, results_nesterov)
    x_proj = simulate_results(problem, results_projgrad)
    x_line = simulate_results(problem, results_linesearch)
    plt.figure()
    plt.plot(x_nest[:,0], label="Nesterov")
    plt.plot(x_proj[:,0], label="Proj. grad.")
    plt.plot(x_line[:,0], label="Line search")
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.legend()
    plt.title("Predicted state trajectory under optimal solution")
    plt.show()


def exercise6():
    print("Execise 6")
    problem, cost_function, f, _, _ = set_the_stage()
    
    Fk = np.array([[1., 0.],[0., 0.]])
    FN = np.array([[1., 0.]])
    F = block_diag(*[Fk for _ in range(problem.N)], FN)

    H_inv = la.inv(cost_function.H)  # Precompute inverse for solving linear systems 
    #  Numerically, it's better to precompute a Cholesky factorization, but 
    #  this implementation keeps the code closer to the mathematical formulations. 

    Gk = np.array([[0.],[1.]])
    G = block_diag(*[Gk for _ in range(problem.N)])
    G = np.vstack([G, np.zeros((1,G.shape[1]))])
    L = F@get_Bbar(problem)+G;
    
    γ = 2./la.norm(L@H_inv@L.T, ord=2)

    U = Box(problem.u_min*np.ones(cost_function.input_dim), 
            problem.u_max*np.ones(cost_function.input_dim))
    
    Abar = get_Abar(problem)
    X = Box(-np.inf*np.ones(problem.N+1), 
            np.array([-F[2*k,:]@Abar@problem.x0 for k in range(problem.N+1)])
    )

    def project(w):
        w = w.copy()
        w[0::2] = X.Π(w[0::2])  # Project the odd entries  (states)
        w[1::2] = U.Π(w[1::2])  # Project the even entries (inputs)
        return w
    
    result = ama(f, H_inv, cost_function.g(problem.x0), L, project, γ, max_it=2000)

    plt.figure()
    plot_convergence_rate(result)
    plt.title("Convergence AMA")

    x = simulate_results(problem, result)
    constraint_violation_x = np.max(x[:, 0])
    constraint_violation_u = np.max(np.abs(result.minimizer - U.Π(result.minimizer)))
    print(f"Max state constraint violation: {constraint_violation_x}")
    print(f"Max input constraint violation: {constraint_violation_u}")

    plt.figure()
    plt.plot(x[:,0], label="AMA")
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.legend()
    plt.title("Predicted state trajectory under optimal solution")
    plt.show()


if __name__ == "__main__":
    exercise5()
