from log import ControllerLog
from rcracers.simulator import simulate
from typing import Tuple
from functools import reduce
import numpy as np
import cvxpy as cp
from rcracers.utils import quadprog
import matplotlib.pyplot as plt
from problem import Problem

import sys
import os

sys.path.append(os.path.split(__file__)[0])  # Allow relative imports

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16
})


"""Setup of session 1 """


def get_dynamics_continuous() -> Tuple[np.ndarray]:
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0], [-1]])
    return A, B


def get_dynamics_discrete(ts: float) -> Tuple[np.ndarray]:
    A, B = get_dynamics_continuous()
    Ad = np.eye(2) + A * ts
    Bd = B * ts
    return Ad, Bd


"""End of setup"""

"""Exercise 3: SDP with cvxpy"""


def setup_session1() -> Tuple[np.ndarray]:
    T_s = 0.5
    A, B = get_dynamics_discrete(T_s)

    C = np.array([[1, -2 / 3]])
    Q = np.matmul(C.T, C) + 1e-3 * np.eye(2, 2)
    R = np.array([[0.1]])

    return A, B, Q, R


def ex3():
    A, B, Q, R = setup_session1()
    K = np.array([[1.0, 2.0]])
    Abar = A + B @ K
    Qbar = Q + K.T @ R @ K

    ns = A.shape[1]
    x0 = np.array([[10.0], [10.0]])  # Condition initiale

    P = cp.Variable((ns, ns), PSD=True)

    cost = x0.T @ P @ x0

    constraints = [(Abar.T @ P @ Abar - P + Qbar) << 0]

    optim = cp.Problem(cp.Minimize(cost), constraints)
    sol = optim.solve()


""" Helper functions """


def unpack_states(sol: quadprog.QuadProgSolution, problem: Problem) -> np.ndarray:
    n_state = problem.n_state
    N = problem.N
    return sol.x_opt[: n_state * (N + 1)].reshape((-1, n_state))


def unpack_inputs(sol: quadprog.QuadProgSolution, problem: Problem) -> np.ndarray:
    n_u = problem.n_input
    n_state = problem.n_state
    N = problem.N

    return sol.x_opt[n_state * (N + 1):].reshape((-1, n_u))


"""End of helper functions"""


""" Abstract class for MPC controller """


class MPC:
    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.ocp_solver = self._build()

    def _build(self):
        ...

    def solve(self, x0) -> quadprog.QuadProgSolution:
        ...

    def __call__(self, y, log) -> np.ndarray:
        if np.isnan(y).any():
            log("solver_success", False)
            log("state_prediction", np.nan)
            log("input_prediction", np.nan)
            return np.nan * np.ones(self.problem.n_input)

        sol = self.solve(y)

        log("solver_success", sol.solver_success)
        log("state_prediction", unpack_states(sol, self.problem))
        log("input_prediction", unpack_inputs(sol, self.problem))

        return unpack_inputs(sol, self.problem)[0]


class MPCCvxpy(MPC):
    name: str = "cvxpy"

    def _build(self) -> cp.Problem:
        ns = self.problem.n_state
        nu = self.problem.n_input
        N = self.problem.N
        x = [cp.Variable((ns,), name=f"x_{i}") for i in range(N + 1)]
        u = [cp.Variable((nu,), name=f"u_{i}") for i in range(N)]
        Ts = self.problem.Ts

        x0 = cp.Parameter((ns,), name="x0")

        A = self.problem.A
        B = self.problem.B

        # Constraintes d'états
        x_max = np.array([self.problem.p_max, self.problem.v_max])
        x_min = np.array([self.problem.p_min, self.problem.v_min])

        # Constraintes d'entrées
        u_max = np.array([self.problem.u_max])
        u_min = np.array([self.problem.u_min])

        Q, R = self.problem.Q, self.problem.R

        # Coût
        cost = cp.sum(
            [cp.quad_form(xk, Q) + cp.quad_form(uk, R) for xk, uk in zip(x, u)]
        )
        cost += cp.quad_form(x[-1], Q)

        constraints = (
            [uk >= u_min for uk in u]
            + [uk <= u_max for uk in u]
            + [xk >= x_min for xk in x]
            + [xk <= x_max for xk in x]
            # Comment out the following line to recreate the infeasible case for 2.3
            + [xk[0] + 0.5*Ts**2*u_min * \
                (N**2-N)+xk[1]*Ts*N <= self.problem.p_max for xk in x]
            + [xk1 == A @ xk + B @ uk for xk1, xk, uk in zip(x[1:], x, u)]
            + [x[0] == x0]
        )

        solver = cp.Problem(cp.Minimize(cost), constraints)

        return solver

    def solve(self, x) -> quadprog.QuadProgSolution:
        solver: cp.Problem = self.ocp_solver

        # Get the symbolic parameter for the initial state
        solver.param_dict["x0"].value = x

        # Call the solver
        optimal_cost = solver.solve()

        if solver.status == "unbounded":
            raise RuntimeError(
                "The optimal control problem was detected to be unbounded. This should not occur and signifies an error in your formulation."
            )

        if solver.status == "infeasible":
            print("  The problem is infeasible!")
            success = False
            optimizer = np.nan * np.ones(
                sum(v.size for v in solver.variables())
            )  # Dummy input.
            value = np.inf  # Infeasible => Infinite cost.

        else:

            # Extract the first control action
            try:
                optimizer = np.concatenate(
                    [solver.var_dict[f"x_{i}"].value for i in range(
                        self.problem.N + 1)]
                    + [solver.var_dict[f"u_{i}"].value for i in range(self.problem.N)]
                )
                success = True  # Everything went well.
                # Get the optimal cost
                value = float(optimal_cost)
            except ValueError:
                print("  The problem is infeasible!")
                success = False
                optimizer = np.nan * np.ones(
                    sum(v.size for v in solver.variables())
                )  # Dummy input.
                value = np.inf  # Infeasible => Infinite cost

        return quadprog.QuadProgSolution(optimizer, value, success)


def run_mpc_simulation(weak_brakes: bool = False, horizon: int = 5):
    print("-"*50 + "Assignment 2.3" + "-"*50)
    # Get the problem data
    problem = Problem(N=horizon)
    if weak_brakes:  # Exercise 7!
        print(" Weakening the brakes!")
        problem.u_min = -10

    # Define the control policy
    policy = MPCCvxpy(problem)

    # Initial state
    x0 = np.array([-100.0, 0])

    # Run the simulation
    print(" Running closed-loop simulation.")
    logs = ControllerLog()
    x_sim = simulate(x0, problem.f, n_steps=60, policy=policy, log=logs)

    # Plot the state trajectory
    default_style = dict(marker=".", color="b")
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 2, 1)
    plt.plot(x_sim[:, 0], x_sim[:, 1], **default_style)

    # Plot the predicted states for every time step
    for x_pred in logs.state_prediction:
        plt.plot(x_pred[:, 0], x_pred[:, 1], alpha=0.4,
                 linestyle="--", **default_style)

    failures = np.logical_not(logs.solver_success)
    plt.scatter(
        *x_sim[:-1][failures, :].T, color="tab:red", marker="x", label="Infeasible"
    )

    # Plot the constraints for easier interpretation
    const_style = dict(color="black", linestyle="--", linewidth=1)
    plt.axvline(problem.p_max, **const_style)
    plt.axhline(problem.v_max, **const_style)
    plt.axhline(problem.v_min, **const_style)
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    if np.any(failures):
        plt.legend(fontsize=16)
    plt.title(
        r"\textbf{Closed-loop trajectory of the vehicle. (%s)}" % policy.name)

    plt.subplot(1, 2, 2)
    inputs = [u[0] for u in logs.input_prediction]
    plt.plot(inputs, marker=".", color="g")
    plt.xlabel("Time step")
    plt.ylabel("Control action")
    plt.title(r"\textbf{Control actions. (%s)}" % policy.name)

    print("*"*120)

    plt.show()


def feasibility_test(weak_breaks: bool = False):
    print("-"*50 + "Assignment 2.4" + "-"*50)

    N = [2, 5, 10]

    fig, axs = plt.subplots(1, 3, sharey=True)

    axs[0].set_ylabel("Velocity")

    for i, horizon in enumerate(N):

        labels = {"inf": "Infeasible", "f": "Feasible"}

        problem = Problem(N=horizon)

        if weak_breaks:
            print("Weak break")
            problem.u_min = -5

        x0_min = np.array([-10.0, 0])
        x0_max = np.array([1.0, 25.0])

        for p0 in np.linspace(x0_min[0], x0_max[0], num=11):
            for v0 in np.linspace(x0_min[1], x0_max[1], num=25):
                policy = MPCCvxpy(problem)
                logs = ControllerLog()

                x0 = np.array([p0, v0])
                x_sim = simulate(x0=x0, dynamics=problem.f,
                                 n_steps=20, policy=policy, log=logs)

                if not logs.solver_success[0]:
                    axs[i].scatter(
                        *x_sim[:-1][0, :].T, color="tab:red", marker="x", label=labels["inf"]
                    )
                    labels["inf"] = "_nolegend_"
                if logs.solver_success[0]:
                    axs[i].scatter(
                        *x_sim[:-1][0, :].T, color="tab:blue", marker="o", facecolors='none', label=labels["f"]
                    )
                    labels["f"] = "_nolegend_"

        const_style = dict(color="black", linestyle="--", linewidth=1.5)
        axs[i].grid()
        axs[i].axvline(problem.p_max, **const_style)
        axs[i].axhline(problem.v_max, **const_style)
        axs[i].axhline(problem.v_min, **const_style)
        axs[i].legend(fontsize=16)
        axs[i].set_xlabel("Position")
        axs[i].set_title(r"\textbf{N = %d}" % horizon)

    print("*"*120)

    plt.show()


if __name__ == "__main__":
    run_mpc_simulation(weak_brakes=True)  # assignment 2.3
    feasibility_test(weak_breaks=True)  # assignment 2.4
