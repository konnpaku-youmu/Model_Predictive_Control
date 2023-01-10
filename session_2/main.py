from typing import Tuple
import numpy as np
import cvxpy as cp
from rcracers.utils import quadprog
import matplotlib.pyplot as plt
from problem import Problem

import sys
import os
sys.path.append(os.path.split(__file__)[0])  # Allow relative imports


"""Setup of session 1 """


def get_dynamics_continuous() -> Tuple[np.ndarray]:
    A = np.array(
        [[0., 1.],
         [0., 0.]]
    )
    B = np.array(
        [[0],
         [-1]]
    )
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

    C = np.array([[1, -2/3]])
    Q = np.matmul(C.T, C) + 1e-3 * np.eye(2, 2)
    R = np.array([[0.1]])

    return A, B, Q, R


def ex3():
    A, B, Q, R = setup_session1()
    K = np.array([[1., 2.]])
    Abar = A + B@K
    Qbar = Q + K.T@R@K

    ns = A.shape[1]
    x0 = np.array([[10.], [10.]])  # Condition initiale

    P = cp.Variable((ns, ns), PSD=True)

    cost = x0.T@P@x0

    constraints = [(Abar.T@P@Abar - P + Qbar) << 0]

    optim = cp.Problem(cp.Minimize(cost), constraints)
    sol = optim.solve()


""" Helper functions """


def unpack_states(sol: quadprog.QuadProgSolution, problem: Problem) -> np.ndarray:
    n_state = problem.n_state
    N = problem.N
    return sol.x_opt[: n_state*(N+1)].reshape((-1, n_state))


def unpack_inputs(sol: quadprog.QuadProgSolution, problem: Problem) -> np.ndarray:
    n_u = problem.n_input
    n_state = problem.n_state
    N = problem.N

    return sol.x_opt[n_state*(N+1):].reshape((-1, n_u))


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


class MPCcvxpy(MPC):

    def _build(self):
        ns = self.problem.n_state
        nu = self.problem.n_input
        N = self.problem.N
        x = [cp.Variable((ns, ), name=f"x_{i}") for i in range(N+1)]
        u = [cp.Variable((nu, ), name=f"u_{i}") for i in range(N)]

        x0 = cp.Parameter((ns, ), name="x0")

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
        cost = cp.sum([cp.quad_form(xk, Q) + cp.quad_form(uk, R)
                      for xk, uk in zip(x, u)])
        cost += cp.quad_form(x[-1], Q)

        constraints =   [uk >= u_min for uk in u] + \
                        [uk <= u_max for uk in u] + \
                        [xk >= x_min for xk in x] + \
                        [xk <= x_max for xk in x] + \
                        [xk1 == A@xk + B@uk for xk1, xk, uk in zip(x[1:], x, u)] + \
                        [x[0] == x0]

        solver = cp.Problem(cp.Minimize(cost), constraints)

        return solver

    def solve(self, x) -> quadprog.QuadProgSolution:
        
        self.ocp_solver.param_dict["x0"].value = x
        
        optim_cost = self.ocp_solver.solve()