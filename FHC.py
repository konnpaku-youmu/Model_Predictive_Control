import casadi as ca
import numpy as np
from numpy.linalg import *
import rcracers

from typing import Tuple, Callable

import matplotlib.pyplot as plt

import LinearSystem


class AutoCruising(LinearSystem.LinearSystem):

    def set_opti_gain(self, gains) -> None:
        self.gains = gains

    def control_law(self, x, t) -> np.ndarray:
        return self.gains[0] @ x


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


def main():
    # Intervalle d'échantillonnage
    T_s = 0.5
    A, B = get_dynamics_discrete(T_s)

    C = np.array([[1], [-2/3]])
    Q = np.matmul(C, C.T) + 1e-3 * np.eye(2, 2)
    R = np.array([0.1])
    P_f = Q

    # Horizon
    N = 20
    # Calculer le gain optimal
    _, gains = ricatti_recursion(A, B, Q, R, P_f, N)

    # Simulation
    x0 = np.array([[1.0], [1.0]]) # Condition initiale

    sys = AutoCruising(A, B, x0) # Définir le système
    n_steps = 10

    sys.set_opti_gain(gains)
    sys.simulate(sys.control_law, n_steps)
    sys.plot_traj()

    plt.show()


if __name__ == "__main__":
    main()
