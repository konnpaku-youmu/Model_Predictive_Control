import casadi as ca
import numpy as np
from scipy import linalg
from numpy.linalg import *
import rcracers

from typing import Tuple, Callable

import matplotlib.pyplot as plt

import LinearSystem

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})


class AutoCruising(LinearSystem.LinearSystem):

    def set_opti_gain(self, gains) -> None:
        self.gains = gains

    def control_law(self, x, t) -> np.ndarray:
        return self.gains[0] @ x

    def pred(self, x, t) -> np.ndarray:
        return self.gains[t] @ x


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


def ricatti_recursion(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P_f: np.ndarray, N: int):
    P = [P_f]
    K = []

    for _ in range(N):
        K_k = -inv(R + B.T @ P[-1] @ B) @ B.T @ P[-1] @ A
        P_k = Q + A.T @ P[-1] @ A + A.T @ P[-1] @ B @ K_k
        K.append(K_k)
        P.append(P_k)

    return P[::-1], K[::-1]


def run_and_plot_traj(A, B, Q, R, P_f, x0):
    plt.figure(1)

    # Horizon
    N_lst = [4, 6, 10]

    for idx, N in enumerate(N_lst):
        # Calculer le gain optimal
        _, gains = ricatti_recursion(A, B, Q, R, P_f, N)

        # Simulation

        sys = AutoCruising(A, B)  # Définir le système
        n_steps = 30

        sys.set_opti_gain(gains)
        sys.simulate(x0, sys.control_law, n_steps)

        plt.subplot(1, 3, idx+1)
        plt.title(r"$N = {}$".format(N))
        sys.plot_traj()

        # Calculer le prédiction de la trajectoire
        for t in range(n_steps):
            x_pred = sys.prediction(sys.x[:, :, t], sys.pred, N)

            plt.plot(x_pred[0, 0, :], x_pred[1, 0, :],
                     'o', linestyle=':', color='#E07360')
            plt.xlabel(r"$x$")
            plt.ylabel(r"$v_x$")

    N = 10

    P_inf = linalg.solve_discrete_are(A, B, Q, R)
    K_inf = -inv(R + B.T@P_inf@B)@B.T@P_inf@A

    sys.set_opti_gain([K_inf] * N)
    sys.simulate(x0, sys.control_law, n_steps)

    plt.figure(2)
    sys.plot_traj()
    for t in range(n_steps):
        x_pred = sys.prediction(sys.x[:, :, t], sys.pred, N)

        plt.plot(x_pred[0, 0, :], x_pred[1, 0, :],
                 'o', linestyle=':', color='#E07360')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$v_x$")
        plt.title(r"Infinite Horizon Controller")

    plt.show()


def compare_term_cost(A, B, Q, R, P_f, x0):
    N_lst = range(1, 10)
    V_N = []

    for N in N_lst:
        P_n, _ = ricatti_recursion(A, B, Q, R, P_f, N)
        P_N = P_n[0]
        V_N.append(np.squeeze(x0.T@P_N@x0))

    P_inf = linalg.solve_discrete_are(A, B, Q, R)
    V_inf = x0.T@P_inf@x0

    plt.plot(N_lst, V_N)
    plt.hlines(V_inf, 1, 10)
    plt.show()


def main():
    # Intervalle d'échantillonnage
    T_s = 0.5
    A, B = get_dynamics_discrete(T_s)

    C = np.array([[1], [-2/3]])
    Q = np.matmul(C, C.T) + 1e-3 * np.eye(2, 2)
    R = np.array([0.1])
    P_f = Q

    x0 = np.array([[10.], [10.]])  # Condition initiale

    # run_and_plot_traj(A, B, Q, R, P_f, x0)
    compare_term_cost(A, B, Q, R, P_f, x0)


if __name__ == "__main__":
    main()
