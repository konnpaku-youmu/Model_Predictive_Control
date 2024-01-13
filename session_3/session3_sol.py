from typing import Callable, List
from rcracers.utils.geometry import Polyhedron, Rectangle, plot_polytope
from rcracers.utils.lqr import LqrSolution, dlqr
from rcracers.simulator import simulate

from given.problem import Problem
from given.log import ControllerLog

import numpy as np 
from dataclasses import dataclass, field 
import matplotlib.pyplot as plt

# Solution code 
from mpc_controller import MPCCvxpy

Matrix = np.ndarray

def compute_ctrl_pre(p: Polyhedron, U: Polyhedron, A: Matrix, B: Matrix) -> Polyhedron:
    """Compute the set of states x for which there exists an input u ∈ U
    such that 
    
    ``Ax + Bu ∈ p``

    project(
        {z :  [H[A  B]]     [ h ]  }
        {  :  [  0 Hu ]z <= [ hu] },
    [0,1]   
    )

    """
    nx = A.shape[0]  # State dim 
    nu = B.shape[1]
    cu = U.H.shape[0]  # Complexity of the input constraints 

    H_augmented = np.vstack(
        [ 
          np.hstack([p.H, np.zeros((p.H.shape[0], nu))]), # This constraint is theoretically unnecessary, but it helps the numerics a lot. 
          p.H @ np.hstack([A, B]), 
          np.hstack([np.zeros((cu, nx)), U.H])
        ]
    )
    h_augmented = np.concatenate([
        p.h,
        p.h, 
        U.h
    ])
    return p.from_inequalities(H_augmented, h_augmented).coordinate_projection((0,1))


def compute_pre(p: Polyhedron, A: Matrix) -> Polyhedron:
    """Compute the set of states x such that Ax ∈ p.
    
    Given that p is described by Hx ≤ h:

    Ax ∈ p ⇔ HAx ≤ h 

    Args:
        p (Polyhedron): Set to compute the pre-set of.
        A (Matrix): Closed-loop dynamics matrix 
    """
    return p.from_inequalities(p.H@A, p.h)


@dataclass
class InvSetResults:
    iterates: List[Polyhedron] = field(default_factory=list) # Iterates of the algorithm 
    n_its: int = 0                                           # Number of iterations performed
    success: bool = False 

    @property 
    def solution(self) -> Polyhedron:
        """Get the final solution, i.e., the last computed iterate."""
        return self.iterates[-1]


def _invariant_set_iteration(X_init, pre: Callable[[Polyhedron], Polyhedron], *, max_it: int = 20):
    """Abstract iterative computation of invariant sets, for a given pre-set computation."""

    result = InvSetResults()
    result.iterates.append(X_init)

    Ω = X_init 
    while result.n_its <= max_it:
        if result.n_its %5 == 0:
            print(f"--- Iteration {result.n_its} - Complexity: {result.solution.nb_inequalities} inequalities")
        Ωnext = pre(Ω).intersect(Ω).canonicalize()
        result.iterates.append(Ωnext)
        if Ωnext == Ω:
            result.success = True 
            return result
        Ω = Ωnext  # Update iterate 
        result.n_its += 1   # Increment loop counter  
    return result


def positive_invariant_set(problem: Problem, K: Matrix, *, max_it: int=20):
    """Use the proposed iterative scheme to compute a positive invariant set for the system in the problem 
    and the control policy ``u = K x``. 

    Args:
        problem (Problem): Problem description
        K (Matrix): State feedback gain shape (ns x nu)
        max_it (int, optional): Maximum number of iterations (for safety). Defaults to 8.
    """
    # Set ingredients 
    Xx = build_state_set(problem)  # Original state constraints 
    Xu = build_input_set(problem, K) # States x for which Kx satisfies the input constraints 
    X = Xx.intersect(Xu)   # Combined state set 

    Acl = problem.A + problem.B@K
    def pre(Ω):
        return compute_pre(Ω, Acl)

    # Call iterative method 
    result = _invariant_set_iteration(X, pre, max_it=max_it)
    return result


def control_invariant_set(problem: Problem, *, max_it: int=50) -> InvSetResults: 

    X = build_state_set(problem)
    U = Rectangle(np.array([problem.u_min]), np.array([problem.u_max]))

    def pre(Ω):
        return compute_ctrl_pre(Ω, U, problem.A, problem.B)
    
    result = _invariant_set_iteration(X, pre, max_it=max_it)
    return result



def build_input_set(problem: Problem, K: Matrix) -> Polyhedron:
    """Build a polyhedral set that describes all states that yield a feasible control action 
    when applying the controller ``u = Kx``.  That is, 
    
    {x | K x ∈ U }

    If U is described by H u ≤ h, then this is 

    {x | HK x ≤ h}    

    Args:
        problem (Problem): Problem description
        K (Matrix): State feedback gain 

    Returns:
        Polyhedron: Set of states such that the controller is feasible 
    """
    U = Rectangle(np.array([problem.u_min]), np.array([problem.u_max]))
    H_in = U.H@K
    h_in = U.h    

    return Polyhedron.from_inequalities(H_in, h_in)


def build_state_set(problem: Problem) -> Polyhedron:
    """Build the state set (exercise 2).

    Args:
        problem (Problem): Problem instance 

    Returns:
        Polyhedron: Set of states that are within the provided bounds. This set is a box. 
    """

    H = np.vstack([
        np.eye(2),
        -np.eye(2)])
    h = np.array([problem.p_max, problem.v_max, -problem.p_min, -problem.v_min])

    return Polyhedron.from_inequalities(H, h)


def plot_invariant_sets(problem: Problem, results: InvSetResults, plot_first: bool = False):
    
    plt.figure()
    ax = plt.subplot()
    plt.title("Invariant set computations")
    plt.xlabel("Position")
    plt.ylabel("Velocity")

    # plot_polytope(ax, invariant_set_computations.original_x, ec="black", label="$X$")
    plot_polytope(build_state_set(problem), fill=False, ec="black", label="$X$", ax=ax)
    if plot_first: 
        plot_polytope(results.iterates[0], fill=False, ec="black", linestyle="--", label="$X \cap \{x \mid K x \in U\}$", ax=ax)
    for it in results.iterates:
        plot_polytope(it, alpha=0.1, fc="tab:blue",ax=ax)
    
    plot_polytope(results.solution, fill = False, color="tab:blue", label="$\Omega_{\infty}$", ax=ax)
    plt.legend()
    plt.show()


def run_mpc_simulation(problem: Problem, Xf: Polyhedron, N: int, lqr_solution: LqrSolution = None, x0: np.ndarray=np.array([-100, 0])):
    """Helper function to run an MPC controller with a given terminal set."""
    print("--Building MPC solver")
    problem.N = N
    controller = MPCCvxpy(problem, Xf, lqr_solution)

    print("--Define dynamics etc. for simulation")

    def dynamics(x, u): 
        return problem.A@x + problem.B@u 

    print("--Run simulation")

    log = ControllerLog()
    x = simulate(x0, dynamics=dynamics, policy=controller, n_steps=60, log=log)

    infeasible = np.logical_not(log.solver_success)
    infeasible = np.append(infeasible, False)  # Cover the last visited state, where the solver has not been called yet. 
    print("--Plot results")
    
    plot_polytope(Xf, color="tab:green", alpha=0.5, label="Terminal set $X_f$")
    plt.plot(x[:,0], x[:,1], marker=".", label="Closed-loop state trajectory")
    if np.any(infeasible):
        # Indicate the infeasible time steps.
        plt.scatter(x[infeasible,0], x[infeasible,1], marker="x", color="tab:red", label="Infeasible time steps")
    
    for x_pred in log.state_prediction:
        plt.plot(x_pred[:, 0], x_pred[:, 1], color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Plot the constraints for easier interpretation 
    const_style = dict(color="black", linestyle="--")
    plt.axvline(problem.p_max, **const_style)
    plt.axhline(problem.v_max, **const_style)
    plt.axhline(problem.v_min, **const_style)
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(f"Closed-loop trajectory ({controller.name} + invariant set {'+ terminal cost' if lqr_solution is not None else ''}, N={N})")
    plt.legend()
    plt.show()


#-----------------------------------------------------------
# EXERCISE SOLUTIONS
#-----------------------------------------------------------

def exercise2():
    print("Exercise 2")
    problem = Problem()
    print("-- Build the state set")
    poly = build_state_set(problem)
    print("-- Plotting the set")
    plt.figure()
    ax = plt.subplot()
    plot_polytope(poly, ax=ax)
    plt.title("Constraint polyhedron")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.show()


def exercise4():
    print("Exercise 4")
    problem = Problem() 
    print("--Compute LQR solution")
    lqr_solution = dlqr(problem.A, problem.B, problem.Q, problem.R)

    print("--Computing invariant sets")
    invariant_set_computations = positive_invariant_set(problem, lqr_solution.K)
    plot_invariant_sets(problem, invariant_set_computations, plot_first=True)

def exercise5():
    print("Exercise 5")
    problem = Problem()

    print("--Computing LQR solution")
    lqr_solution = dlqr(problem.A, problem.B, problem.Q, problem.R)
    print("--Computing positive invariant set")
    Xf = positive_invariant_set(problem, lqr_solution.K).solution 

    for N in [5, 10, 20]:
        run_mpc_simulation(problem, lqr_solution=lqr_solution, Xf=Xf, N=N)

def exercise6():
    print("Exercise 6")
    problem = Problem() 
    print("--Compute LQR solution")
    lqr_solution = dlqr(problem.A, problem.B, problem.Q, problem.R)

    print("--Computing invariant sets")
    Xf = positive_invariant_set(problem, lqr_solution.K).solution
    N = 1
    run_mpc_simulation(problem, lqr_solution=lqr_solution, Xf=Xf, N=N, x0=np.array([-3., 0]))

def exercise7_8():
    print("Exercise 7")
    problem = Problem(u_min=-10)
    print("--Compute control invariant set")
    results = control_invariant_set(problem)
    plot_invariant_sets(problem, results)
    
    print("Exercise 8")
    Xf = results.solution
    
    for N in [1, 5, 10]:
        run_mpc_simulation(problem, lqr_solution=None, Xf=Xf, N=N)

if __name__ == "__main__":
    # print("Exercise 1 -- pen and paper.")
    # exercise2()
    # print("Exercise 3 -- pen and paper.")
    # exercise4()
    # exercise5()
    exercise6()
    # exercise7_8()