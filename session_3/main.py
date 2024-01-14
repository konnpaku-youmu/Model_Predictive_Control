from mpc_controller import MPCCvxpy
from typing import Callable, List
from rcracers.utils.geometry import Polyhedron, Rectangle, plot_polytope, Ellipsoid, plot_ellipsoid
from rcracers.utils.lqr import LqrSolution, dlqr
from rcracers.simulator import simulate

from given.problem import Problem
from given.log import ControllerLog

import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})

# Solution code


@dataclass
class InvSetResults:
    n_iter: int = 0
    iterations: List[Polyhedron] = field(default_factory=list)
    success: bool = False

    def increment(self):
        self.n_iter += 1

    @property
    def solution(self) -> Polyhedron:
        return self.iterations[-1]


def build_feasible_state_set(problem: Problem):
    H = np.vstack([np.eye(problem.n_state),
                  -np.eye(problem.n_state)])

    h = np.array([problem.p_max, problem.v_max, -
                 problem.p_min, -problem.v_min])

    return Polyhedron.from_inequalities(H, h)


def build_feasible_input_set(problem: Problem):
    # find the LQR solution of the problem
    lqr_solution = dlqr(problem.A, problem.B, problem.Q, problem.R)

    Hu = np.vstack([np.eye(problem.n_input),
                    -np.eye(problem.n_input)]) @ lqr_solution.K
    hu = np.array([problem.u_max, -problem.u_min])

    return Polyhedron.from_inequalities(Hu, hu)


def intersect(Hs, hs):
    return np.concatenate(Hs), np.concatenate(hs)


def compute_pre(set: Polyhedron, Acl: np.ndarray):
    return set.from_inequalities(set.H@Acl, set.h)


def invariant_iter(Ω_0: Polyhedron, pre: Callable, max_iter: int):
    result = InvSetResults()
    result.iterations.append(Ω_0)

    Ω = Ω_0
    while result.n_iter <= max_iter:
        Ω_next = pre(Ω).intersect(Ω).canonicalize()
        result.iterations.append(Ω_next)
        if Ω_next == Ω:
            result.success = True
            break
        Ω = Ω_next
        result.increment()

    return result


def compute_invariant_set(problem: Problem, max_iter: int = 200, weak_brakes: bool = False):
    Xx = build_feasible_state_set(problem)
    Xu = build_feasible_input_set(problem)
    X_init = Xx.intersect(Xu)
    Ω_0 = X_init

    # find the LQR solution of the problem
    lqr_solution = dlqr(problem.A, problem.B, problem.Q, problem.R)
    K = lqr_solution.K
    Acl = problem.A + problem.B@K

    def pre(Ω):
        return compute_pre(Ω, Acl)

    result = invariant_iter(Ω_0, pre, max_iter)

    return result


def plot_invariant_sets(results: InvSetResults, ax: plt.Axes, plot_first: bool = False):

    if plot_first:
        plot_polytope(results.iterations[0], fill=False, ec="black",
                      linestyle="--", label="$X \cap \{x \mid K x \in U\}$", ax=ax)
    for it in results.iterations:
        plot_polytope(it, alpha=0.1, fc="tab:blue", ax=ax)

    plot_polytope(results.solution, fill=False, color="tab:blue",
                  label="$\Omega_{\infty}$", ax=ax)

    ax.legend()


def plot_system_trajectory(problem: Problem, policy: MPCCvxpy, logs: ControllerLog, x_sim):
    default_style = dict(marker=".", color="b")
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(x_sim[:, 0], x_sim[:, 1], **default_style)

    # Plot the predicted states for every time step
    for x_pred in logs.state_prediction:
        plt.plot(x_pred[:, 0], x_pred[:, 1], alpha=0.4,
                 linestyle="--", **default_style)

    failures = np.logical_not(logs.solver_success)
    if np.any(failures):
        plt.scatter(
            *x_sim[:-1][failures, :].T, color="tab:red", marker="x", label="Infeasible"
        )
        plt.legend()

    # Plot the constraints for easier interpretation
    const_style = dict(color="black", linestyle="--", linewidth=1)
    plt.axvline(problem.p_max, **const_style)
    plt.axhline(problem.v_max, **const_style)
    plt.axhline(problem.v_min, **const_style)
    plt.xlabel("Position")
    plt.ylabel("Velocity")

    plt.title(
        r"\textbf{Closed-loop trajectory of the vehicle. (%s)}" % policy.name)

    ax2 = plt.subplot(1, 2, 2)
    inputs = [u[0] for u in logs.input_prediction]
    plt.plot(inputs, marker=".", color="g")
    plt.xlabel("Time step")
    plt.ylabel("Control action")
    plt.title(r"\textbf{Control actions. (%s)}" % policy.name)

    return ax1, ax2


def run_mpc_simulation(problem: Problem, x0: np.ndarray, Xf, logs: ControllerLog, weak_brakes: bool = False):

    if weak_brakes:
        print(" Weakening the brakes!")
        problem.u_min = -10

    lqr_solution = dlqr(problem.A, problem.B, problem.Q, problem.R)

    # Define the control policy
    policy = MPCCvxpy(problem, Xf, lqr_solution)
    
    # Run the simulation
    print(" Running closed-loop simulation.")
    
    x_sim = simulate(x0, problem.f, n_steps=60, policy=policy, log=logs)

    return x_sim, policy


def exercise2():
    problem = Problem()
    H, h = build_feasible_state_set(problem)
    X = Polyhedron.from_inequalities(H, h)
    plot_polytope(X)
    plt.show()


def exercise4():
    problem = Problem(N=15)
    invariant_iters = compute_invariant_set(problem, weak_brakes=True)
    run_mpc_simulation(problem, invariant_iters, weak_brakes=True)


def compute_ellipsoid(weak_brakes: bool = False):
    problem = Problem(N=7)
    if weak_brakes:
        problem.u_min = -10

    lqr_solution = dlqr(problem.A, problem.B, problem.Q, problem.R)

    invariant_iters = compute_invariant_set(problem)
    Xf = invariant_iters.solution

    H, h = Xf.H, Xf.h

    alpha_max = np.min(
        [gi**2 / (hi.T @ np.linalg.inv(lqr_solution.P) @ hi) for hi, gi in zip(H, h)])

    E = Ellipsoid(lqr_solution.P / alpha_max)

    x0 = np.array([-15.0, 0])
    logs = ControllerLog()
    x_sim, policy = run_mpc_simulation(problem, x0, E, logs)

    # Plot the state trajectory
    plt.figure(figsize=[15, 6])
    ax1, _ = plot_system_trajectory(problem, policy, logs, x_sim)

    plot_invariant_sets(invariant_iters, ax1, plot_first=True)

    plot_ellipsoid(E, label=r"$lev_{\alpha}V_{\infty}$", alpha=0.75, fill=False, linewidth=1.5, color="tab:green", ax=ax1)

    ax1.legend(fontsize=16)
    # plt.title(r"\textbf{Polyhedral \& ellipsoidal invariant set}")
    plt.show()


if __name__ == "__main__":
    compute_ellipsoid()  # assignment 3.2
