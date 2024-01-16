from dataclasses import dataclass, field
import numpy as np 

@dataclass
class Problem: 
    ts: float = 0.3  # Sample time 
    u_min: float = -10. 
    u_max: float = 10. 
    Q: np.ndarray = np.diag([10., 1.])
    R: np.ndarray = np.diag([1.])
    N: int = 20
    x0: np.ndarray = np.array([-100., 0.])
    # The following fields are initialized in __post_init__ 
    P: np.ndarray = field(init=False)
    A: np.ndarray = field(init=False)
    B: np.ndarray = field(init=False)
    
    
    def __post_init__(self):
        self.A = np.array([[1, self.ts],[0, 1]])
        self.B = np.array([[0],[self.ts]])
        self.P = 5 * self.Q 

    @property
    def ns(self):
        """State dimension"""
        return self.A.shape[0]

    @property 
    def nu(self): 
        return self.B.shape[1]