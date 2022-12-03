from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt


class LinearSystem:
    def __init__(self, A, B) -> None:
        self.A = A
        self.B = B

    def set_output_eq(self, C, D) -> None:
        self.C = C
        self.D = D

    def f(self, x, u) -> np.ndarray:
        x_next = self.A @ x + self.B @ u
        return x_next
    
    def simulate(self, x0: np.ndarray, control_law: Callable, steps: int) -> None:
        self.x = np.expand_dims(x0, axis=2)
        for t in range(1, steps):
            u_t = control_law(self.x[:, :, -1], t)
            # Calculer l'état suivant
            x_next = self.f(self.x[:, :, -1], u_t)
            self.x = np.dstack((self.x, x_next))

    def prediction(self, xt: np.ndarray, pred_law: Callable, horizon: int) -> Tuple[np.ndarray]:
        x_pred = np.expand_dims(xt, axis=2)#
        for t in range(1, horizon):
            u_p = pred_law(x_pred[:, :, -1], t)
            x_p = self.f(x_pred[:, :, -1], u_p)
            x_pred = np.dstack((x_pred, x_p))
        
        return x_pred

    def plot_traj(self) -> None:
        plt.plot(self.x[0, 0, :], self.x[1, 0, :],
                 'x', linestyle='--', color='#685BF5', label = "Trajectory")
        plt.legend()
    
    def plot_cost(self, P_N) -> None:
        pass

    def plot_pred(self, horizon: int) -> None:
        pass
