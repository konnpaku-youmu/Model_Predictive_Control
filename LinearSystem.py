from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt


class LinearSystem:
    def __init__(self, A, B, x0) -> None:
        self.A = A
        self.B = B
        self.x = np.ndarray(shape=(2, 1, 1), dtype=float, buffer=x0)

    def set_output_eq(self, C, D) -> None:
        self.C = C
        self.D = D

    def f(self, u) -> None:
        x_next = self.A @ self.x[:, :, -1] + self.B @ u
        self.x = np.dstack((self.x, x_next))

    def plot_traj(self) -> None:
        plt.plot(self.x[0, 0, :], self.x[1, 0, :],
                 'x', linestyle=':', color='#685BF5')

    def simulate(self, control_law: Callable, steps: int) -> None:
        for t in range(1, steps):
            u_t = control_law(self.x[:, :, -1], t)
            self.f(u_t)
