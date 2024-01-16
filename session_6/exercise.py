from rcracers.simulator import simulate
from typing import Callable
from given.config import Problem

from scipy.linalg import block_diag
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
import os
WORKING_DIR = os.path.split(__file__)[0]


@dataclass
class QuadraticFunction:
    """
    Represent 
    f(x) = 1/2 u'Hu + (G*x0)'u 
    by H and G
    """
    H: np.ndarray
    G: np.ndarray
    x0: np.ndarray

    def __call__(self, u):
        return 0.5 * u@self.H@u + self.g()@u

    def g(self):
        return self.G@self.x0

    def grad(self, z):
        return self.H@z + self.G@self.x0

    @property
    def input_dim(self):
        return self.H.shape[1]


@dataclass
class OptimizerStats:
    costs: list = field(default_factory=list)
    optimal_value: float = np.nan
    minimizer: np.ndarray = np.nan


@dataclass
class Box:
    u_min: np.ndarray
    u_max: np.ndarray

    def Π(self, w):
        return np.minimum(self.u_max, np.maximum(w, self.u_min))


def get_Abar(problem: Problem):
    return np.vstack([la.matrix_power(problem.A, i) for i in range(problem.N+1)])


def get_Bbar(problem: Problem):
    """Compute \bar{B} from the solution document 
    """
    # Convenience function to generate a zero matrix of appropriate dimensions
    # This is safer than saving the output as a variable and using it everywhere
    # because that may lead to unexpected behavior:
    #   we would be filling the matrix with multiple objects all pointing to the same data.
    def O(): return np.zeros((problem.ns, problem.nu))
    Bb = [[O() for _ in range(problem.N)]]
    for blkrow in range(problem.N):
        new_row = [
            la.matrix_power(problem.A, i)@problem.B for i in range(blkrow, -1, -1)] \
            + [O() for _ in range(problem.N - blkrow - 1)]
        Bb.append(new_row)
    Bbar = np.block(Bb)
    return Bbar


def convert_to_single_shooting(prob: Problem) -> QuadraticFunction:

    Q_blk = np.kron(np.eye(prob.N+1), prob.Q)
    Q_blk[-prob.ns:, -prob.ns:] = prob.P
    R_blk = np.kron(np.eye(prob.N), prob.R)

    A_bar = get_Abar(prob)
    B_bar = get_Bbar(prob)

    H = B_bar.T @ Q_blk @ B_bar + R_blk
    G = B_bar.T @ Q_blk.T @ A_bar

    return QuadraticFunction(H, G, prob.x0)


def line_search(f: QuadraticFunction, γ: float, u_next, u):

    pass


def projected_gradient(f: QuadraticFunction, γ: float, U: Box, *, max_it: int = 200,
                       use_nesterov: bool = False, a: float = 0.5,
                       use_linesearch: bool = False):
    """Projected gradient algorithm.

    Args:
        f (Callable): Cost function Rⁿ -> R
        df (_type_): Gradient of the cost Rⁿ -> Rⁿ
        γ (float): Step size
        U (Box): Box (constraints)
        max_it (int): Maximum number of iterations
    """
    # initial u
    solver_res = OptimizerStats()

    u = np.zeros(f.input_dim)  # initial guess: 0
    if use_nesterov:
        v = u
    solver_res.costs.append(f(u))

    for _ in range(max_it):
        if use_linesearch:
            while True:
                if use_nesterov:
                    u_next = U.Π(v - γ * f.grad(v))
                    v = u + a*(u_next - u)
                else:
                    u_next = U.Π(u - γ * f.grad(u))
                if f(u_next) <= f(u) + f.grad(u).T@(u_next - u) + 1/(2*γ)*np.linalg.norm(u_next - u)**2:
                    break
                γ /= 2
        else:
            if use_nesterov:
                u_next = U.Π(v - γ * f.grad(v))
                v = u + a*(u_next - u)
            else:
                u_next = U.Π(u - γ * f.grad(u))

        u = u_next
        solver_res.costs.append(f(u_next))

    solver_res.minimizer = u
    solver_res.optimal_value = f(u)

    return solver_res


def simulate_results(problem: Problem, results: OptimizerStats) -> np.ndarray:
    print("-- Rollout the found solution")

    def policy(y, t):
        return results.minimizer[t]

    def dynamics(x, u):
        return problem.A @ x + problem.B @ np.atleast_1d(u)

    print("-- Calling the simulation utility from rcracers")
    x = simulate(problem.x0, dynamics, len(results.minimizer), policy=policy)
    return x


def plot_convergence_rate(results, **style):
    subopt = np.abs((np.array(results.costs)[0:len(
        results.costs)//2] - results.optimal_value)/results.optimal_value)
    plt.semilogy(subopt, **style)
    plt.title("Convergence rate")
    plt.xlabel("Iterations")
    plt.ylabel("Relative suboptimality")


def exercise3():
    problem = Problem()
    cost_func = convert_to_single_shooting(problem)
    U = Box(problem.u_min * np.ones(cost_func.input_dim),
            problem.u_max * np.ones(cost_func.input_dim))

    Lf = np.linalg.norm(cost_func.H, ord=2)  # Maximum eigenvalue

    results = projected_gradient(cost_func, 1.0/Lf, U)

    plt.figure()
    plot_convergence_rate(results)

    x = simulate_results(problem, results)
    plt.figure()
    plt.plot(x[:, 0], x[:, 1])
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Predicted state trajectory under optimal solution (Proj. grad.)")
    plt.show()


def exercise4():
    problem = Problem()
    cost_func = convert_to_single_shooting(problem)
    U = Box(problem.u_min * np.ones(cost_func.input_dim),
            problem.u_max * np.ones(cost_func.input_dim))

    eigv, _ = np.linalg.eigh(cost_func.H)
    Lf, μf = eigv[-1], eigv[0]
    κf = Lf / μf
    ακ = (np.sqrt(κf)-1) / (np.sqrt(κf)+1)

    res_plain = projected_gradient(cost_func, 1.0/Lf, U)
    res_nest = projected_gradient(
        cost_func, 1.0/Lf, U, use_nesterov=True, a=ακ)

    plt.figure()
    plot_convergence_rate(res_plain)
    plot_convergence_rate(res_nest)

    x = simulate_results(problem, res_nest)
    plt.figure()
    plt.plot(x[:, 0], x[:, 1])
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Predicted state trajectory under optimal solution (Proj. grad.)")
    plt.show()


def exercise5():
    problem = Problem()
    cost_func = convert_to_single_shooting(problem)
    U = Box(problem.u_min * np.ones(cost_func.input_dim),
            problem.u_max * np.ones(cost_func.input_dim))

    eigv, _ = np.linalg.eigh(cost_func.H)
    Lf, μf = eigv[-1], eigv[0]
    κf = Lf / μf
    ακ = (np.sqrt(κf)-1) / (np.sqrt(κf)+1)

    res_plain = projected_gradient(cost_func, 1.0/Lf, U)
    res_nest = projected_gradient(cost_func, 1.0/Lf, U, use_nesterov=True, a=ακ)
    res_ls = projected_gradient(cost_func, 100.0/Lf, U, use_nesterov=True, a=ακ, use_linesearch=True)

    plt.figure()
    plot_convergence_rate(res_plain)
    plot_convergence_rate(res_nest)
    plot_convergence_rate(res_ls)

    x = simulate_results(problem, res_ls)
    plt.figure()
    plt.plot(x[:, 0], x[:, 1])
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Predicted state trajectory under optimal solution (Proj. grad.)")
    plt.show()


if __name__ == "__main__":
    exercise5()
