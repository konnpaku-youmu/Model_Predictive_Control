from dataclasses import dataclass
import numpy as np 

# 
# Use the problem definition from session 2
# 

@dataclass
class Problem:
    """Convenience class representing the problem data for session 2/3."""

    Ts: float = 0.3
    Q: np.ndarray = np.diag([10, 1])
    R: np.ndarray = np.diag([0.01])
    p_min: float = -120 # Minimal position
    p_max: float = 1.0  # Maximal position 
    v_min: float = -50  # Minimal velocity
    v_max: float = 25.0  # Maximal velocity
    u_min: float = -20.0
    u_max: float = 10.0
    N: int = 5

    A: np.ndarray = None
    B: np.ndarray = None

    def __post_init__(self):
        self.A = np.array([[1.0, self.Ts], [0, 1.0]])
        self.B = np.array([[0], [self.Ts]])

    def f(self, x, u):
        return self.A @ x + self.B @ u

    @property
    def n_state(self):
        return self.A.shape[0]

    @property
    def n_input(self):
        return self.B.shape[1]