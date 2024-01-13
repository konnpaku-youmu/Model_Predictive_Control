from typing import Optional
import casadi as cs
from given.problem import Problem
from rcracers.utils.geometry import Polyhedron
from rcracers.utils.lqr import LqrSolution
from rcracers.utils import quadprog
import cvxpy as cp

import numpy as np

# -----------------------------------------------------------
# Helper functions (See also solution code of session 2.)
# -----------------------------------------------------------


def unpack_states(sol: quadprog.QuadProgSolution, problem: Problem) -> np.ndarray:
    n_state = problem.n_state
    N = problem.N
    return sol.x_opt[: n_state * (N + 1)].reshape((-1, n_state))


def unpack_inputs(sol: quadprog.QuadProgSolution, problem: Problem) -> np.ndarray:
    n_u = problem.n_input
    n_state = problem.n_state
    N = problem.N

    return sol.x_opt[n_state * (N + 1):].reshape((-1, n_u))


class MPC:
    def __init__(self, problem: Problem, Xf: Optional[Polyhedron],
                 lqr_solution: Optional[LqrSolution]) -> None:
        self.problem = problem
        self.Xf = Xf
        self.term_controller = lqr_solution
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


        constraints = (
            [uk >= u_min for uk in u]
            + [uk <= u_max for uk in u]
            + [xk >= x_min for xk in x]
            + [xk <= x_max for xk in x]
            + [xk[0] + 0.5*Ts**2*u_min *
                (N**2-N)+xk[1]*Ts*N <= self.problem.p_max for xk in x]
            + [xk1 == A @ xk + B @ uk for xk1, xk, uk in zip(x[1:], x, u)]
            + [x[0] == x0]
        )

        # Terminal conditions
        if self.term_controller is not None:
            cost += cp.quad_form(x[-1], self.term_controller.P)
        else:
            cost += cp.quad_form(x[-1], Q)

        if self.Xf is not None:
            constraints += [self.Xf.H @ x[-1] <= self.Xf.h]

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
