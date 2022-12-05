import numpy as np 
from typing import Tuple , Callable

try: 
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not yet installed. Install it using `pip install matplotlib`.")
    exit()


def get_dynamics_continuous() -> Tuple[np.ndarray]: 
    """Get the continuous-time dynamics represented as 
    
    ..math::
        \dot{x} = A x + B u 
    
    """
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
    """Get the dynamics of the cars in discrete-time:

    ..math::
        x_{k+1} = A x_k + B u_k  
    
    Args: 
        ts [float]: sample time [s]
    """
    A, B = get_dynamics_continuous()
    Ad = np.eye(2) + A * ts 
    Bd = B * ts 
    return Ad, Bd


def riccati_recursion(A: np.ndarray, B: np.ndarray, R: np.ndarray, Q: np.ndarray, Pf: np.ndarray, N: int):
    """Solve the finite-horizon LQR problem through recursion

    Args:
        A: System A-matrix 
        B: System B-matrix 
        R: weights on the input (positive definite)
        Q: weights on the states (positive semidefinite)
        Pf: Initial value for the Hessian of the cost-to-go (positive definite)
        N: Control horizon  
    """
    import numpy.linalg as la  # Import linear algebra to solve linear system

    P = [Pf] 
    K = [] 
    for _ in range(N):
        Kk = -la.solve(R + B.T@P[-1]@B, B.T@P[-1]@A)
        K.append(Kk)
        Pk = Q + A.T@P[-1]@(A + B@K[-1])
        P.append(Pk)

    return P[::-1], K[::-1]  # Reverse the order for easier indexing later. 


def simulate(x0: np.ndarray, f: Callable, policy: Callable, steps: int) -> Tuple[np.ndarray, bool]:
    """Generic simulation loop.
    
    Simulate the discrete-time dynamics f: (x, u) -> x
    using policy `policy`: (x, t) -> u 
    for `steps` steps ahead and return the sequence of states.

    Returns 
        x: sequence of states 
        instability_occurred: whether or not the state grew to a large norm, indicating instability 
    """
    instability_occured = False  # Keep a flag that indicates whenever we detected instability. 
    x = [x0]
    for t in range(steps):
        xt = x[-1]
        ut = policy(xt, t)
        xnext = f(xt, ut)
        x.append(xnext)
        if np.linalg.norm(xnext) > 100 and not instability_occured:  
            # If the state become very large, we flag instability. 
            # (This is a heuristic of course, but for this example, it suffices.)
            instability_occured = True 
    
    return np.array(x), instability_occured


def plot_ex4(x0: np.ndarray, A: np.ndarray, B: np.ndarray, gains: list, sim_time=10): 
    """Plot the simulated states, when applying the given state feedback gains.
    
    Args: 
        x0: Initial state 
        A: A-matrix of the system
        B: B-matrix of the system
        gains: List of state-feedback gains
    """

    def f(x,u):
        """Dynamics"""
        return A@x + B@u

    def κ(x,t):
        """Control policy (receding horizon)"""
        return gains[0]@x
    horizon = len(gains)
    x_closed_loop, cl_unstable = simulate(x0, f, κ, sim_time)

    if cl_unstable: 
        print("The state grew quite large under the closed-loop policy, which indicates instability!")
        print("Note that in this case (everything is linear), we can also test exactly for stability! (How?) Implement this as an exercise!")
    plt.figure()
    plt.plot(x_closed_loop[:, 0], x_closed_loop[:, 1], marker=".", color="k", linewidth=2)

    # Plot the predictions
    def κ_pred(x,t): 
        """Control policy (receding horizon)"""
        return gains[t]@x

    for xt in x_closed_loop:
        x_pred, _ = simulate(xt, f, κ_pred, horizon)
        plt.plot(x_pred[:, 0], x_pred[:, 1], color="tab:red", linestyle="--", marker=".", alpha=0.5)
    plt.annotate("$x_0$", x0)

    plt.title(f"State trajectory (real: black | predicted: red) for N = {horizon}")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.show()


def setup():
    ts = 0.5 
    C = np.array([[1, -2./3]])
    Q = C.T@C + 1e-3 * np.eye(2)
    R = np.array([[0.1]])

    A, B = get_dynamics_discrete(ts)
    
    return A, B, Q, R



#-----------------------------------------------------------
# Solutions to exercises
#-----------------------------------------------------------

def exercise3():
    print("Exercise 3: See `riccati_recursion`")

def exercise4():
    print("Exercise 4.")
    # Set the given parameters 
    print(" Setting parameter values.")
    A, B, Q, R = setup()

    Pf = Q 
    x0 = 10 * np.ones(2)
    N = 5

    for N in [4, 6, 10, 20]:

        print(f" Running recursion for N = {N}.")
        _, gains = riccati_recursion(A, B, R, Q, Pf, N)
        
        plot_ex4(x0, A, B, gains, sim_time=30)





def main():

    exercise3()
    exercise4()




if __name__ == "__main__": 
    main()
