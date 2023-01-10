import matplotlib.pyplot as plt 
from problem import Problem

import sys
import os
sys.path.append(os.path.split(__file__)[0])  # Allow relative imports

from rcracers.utils import quadprog

import cvxpy as cp
import numpy as np

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

def get_states(sol: quadprog.QuadProgSolution, problem: Problem) -> np.ndarray:
    """Given a solution of a QP solver, return the predicted state sequence.

    Args:
        sol (quadprog.QuadProgSolution): QP solution
        problem (Problem): problem

    Returns:
        np.ndarray: state sequence shape: (N, nx)
    """
    ns = problem.n_state
    N = problem.N
    return sol.x_opt[: ns * (N + 1)].reshape((-1, ns))


def get_inputs(sol: quadprog.QuadProgSolution, problem: Problem) -> np.ndarray:
    """Given a solution of a QP solver, return the predicted input sequence.

    Args:
        sol (qp_solver.QuadProgSolution): QP solution
        problem (Problem): problem

    Returns:
        np.ndarray: state sequence shape: (N, nu)
    """
    ns = problem.n_state
    N = problem.N
    nu = problem.n_input
    return sol.x_opt[ns * (N + 1) :].reshape((-1, nu))

class MPC:
    """Abstract baseclass for an MPC controller. 
    """
    def __init__(self, problem: Problem):
        self.problem = problem 
        print(" Building MPC problem")
        self.qp = self._build()
    
    def _build(self):
        """Build the optimization problem."""
        ...
    
    def solve(self, x) -> quadprog.QuadProgSolution:
        """Call the optimization problem for a given initial state"""
        ...

    def __call__(self, y, log) -> np.ndarray: 
        """Call the controller for a given measurement. 

        The controller assumes perfect state measurements.
        Solve the optimal control problem, write some stats to the log and return the control action. 
        """

        # If the state is nan, something already went wrong. 
        # There is no point in calling the solver in this case. 
        if np.isnan(y).any():
            log("solver_success", False)
            log("state_prediction", np.nan)
            log("input_prediction", np.nan)
            return np.nan * np.ones(self.problem.n_input)

        # Solve the MPC problem for the given state 
        solution = self.solve(y)
        
        log("solver_success", solution.solver_success)
        log("state_prediction", get_states(solution, self.problem))
        log("input_prediction", get_inputs(solution, self.problem))
        
        return get_inputs(solution, self.problem)[0]


class MPCCvxpy(MPC):
    name:str="cvxpy"

    def _build(self) -> cp.Problem:
        
        # Make symbolic variables for the states and inputs 
        x = [cp.Variable((self.problem.n_state,), name=f"x_{i}") for i in range(self.problem.N+1)]
        u = [cp.Variable((self.problem.n_input,), name=f"u_{i}") for i in range(self.problem.N)]
        
        # Symbolic variable for the parameter (initial state)
        x_init = cp.Parameter((self.problem.n_state,), name="x_init")

        # Equality constraints 
        # -- dynamics
        A = self.problem.A
        B = self.problem.B
        
        # Inequality constraints -- simple bounds on the variables 
        #  -- state constraints 
        xmax = np.array([self.problem.p_max, self.problem.v_max])
        xmin = np.array([self.problem.p_min, self.problem.v_min])

        #  -- Input constraints 
        umax = np.array([self.problem.u_max])
        umin = np.array([self.problem.u_min])

        # Cost 
        Q, R = self.problem.Q, self.problem.R 
        
        # Sum of stage costs 
        cost = cp.sum([cp.quad_form(xt, Q) + cp.quad_form(ut, R) for (xt, ut) in zip(x,u)])
        cost = cost + cp.quad_form(x[-1], Q)  # Add terminal cost

        constraints = [ uk <= umax for uk in u ] + \
                      [ uk >= umin for uk in u ] + \
                      [ xk <= xmax for xk in x ] + \
                      [ xk >= xmin for xk in x ] + \
                      [ x[0] == x_init] + \
                      [ xk1 == A@xk + B@uk for xk1, xk, uk in zip(x[1:], x, u)] 

        solver = cp.Problem(cp.Minimize(cost), constraints)

        return solver

    def solve(self, x) -> quadprog.QuadProgSolution: 
        solver: cp.Problem = self.qp
        
        # Get the symbolic parameter for the initial state 
        solver.param_dict["x_init"].value = x 
        
        # Call the solver 
        optimal_cost = solver.solve()

        if solver.status == "unbounded":
            raise RuntimeError("The optimal control problem was detected to be unbounded. This should not occur and signifies an error in your formulation.")

        if solver.status == "infeasible":
            print("  The problem is infeasible!")
            success = False 
            optimizer = np.nan * np.ones(sum(v.size for v in solver.variables())) # Dummy input. 
            value = np.inf  # Infeasible => Infinite cost. 

        else: 
            success = True # Everything went well. 
            # Extract the first control action
            optimizer = np.concatenate([solver.var_dict[f"x_{i}"].value for i in range(self.problem.N + 1)]\
                                       + [solver.var_dict[f"u_{i}"].value for i in range(self.problem.N)])
    
            # Get the optimal cost 
            value = float(optimal_cost)
        
        return quadprog.QuadProgSolution(optimizer, value, success)


def run_mpc_simulation(weak_brakes: bool = False):
    from rcracers.simulator import simulate 
    from log import ControllerLog

    # Get the problem data
    problem = Problem()
    if weak_brakes: # Exercise 7! 
        print(" Weakening the brakes!")
        problem.u_min = -10

    # Define the simulator dynamics (we assume no model mismatch)
    def f(x, u):
        return problem.A @ x + problem.B @ u

    # Define the control policy
    policy = MPCCvxpy(problem)

    # Initial state 
    x0 = np.array([-100.0, 0.0])

    # Run the simulation
    print(" Running closed-loop simulation.")
    logs = ControllerLog()
    x_sim = simulate(x0, f, n_steps=60, policy=policy, log=logs)

    # Plot the state trajectory 
    default_style = dict(marker=".", color="black")
    plt.plot(x_sim[:,0], x_sim[:,1], **default_style)
    
    # Plot the predicted states for every time step
    for x_pred in logs.state_prediction:
        plt.plot(x_pred[:,0], x_pred[:,1], alpha=0.4, linestyle="--", **default_style)

    failures = np.logical_not(logs.solver_success)
    plt.scatter(*x_sim[:-1][failures, :].T, color="tab:red", marker="x", label="Infeasible")

    # Plot the constraints for easier interpretation 
    const_style = dict(color="black", linestyle="--")
    plt.axvline(problem.p_max, **const_style)
    plt.axhline(problem.v_max, **const_style)
    plt.axhline(problem.v_min, **const_style)
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    if np.any(failures):
        plt.legend()
    plt.title(f"Closed-loop trajectory of the vehicle. ({policy.name})")

    plt.figure()
    inputs = [u[0] for u in logs.input_prediction]
    plt.plot(inputs, **default_style)
    plt.xlabel("Time step")
    plt.ylabel("Control action")
    plt.title(f"Control actions. ({policy.name})")

    plt.show()

# -----------------------------------------------------------
# Solutions to the exercises
# -----------------------------------------------------------
def setup_part1():
    ts = 0.5 
    C = np.array([[1, -2./3]])
    Q = C.T@C + 1e-3 * np.eye(2)
    R = np.array([[0.1]])

    A = np.array(
        [[1., ts],
         [0., 1.]]
        )
    B = np.array(
        [[0],
         [-ts]]
    )

    return A, B, Q, R


def cost_est_sim(A, B, K, Q, R, x0, nsteps = 50):
    cost = 0 
    x  = x0 
    KRK = K.T@R@K 
    QKRK = Q + KRK 
    for _ in range(nsteps):
        cost += x@QKRK@x
        x = (A + B@K)@x 
    return cost 

def exercise3():
    print("Exercise 3.")
    import cvxpy as cp 
    
    A, B, Q, R = setup_part1()
    K = np.array([[1., 2.]])
    
    # Set up the numerical values 
    Abar = A + B@K
    Qbar = Q + K.T@R@K
    ns = A.shape[0]  # State dimensions
    x = 10*np.ones(ns)   # Initial state 
    # cvxpy.org/api_reference/cvxpy.expressions.html#cvxpy.expressions.leaf.Leaf
    P = cp.Variable((ns, ns), PSD=True)

    cost = x@P@x  # Set the cost. Remember, P is the variable here! 
    # Hint: semidefinite constraints are implemented using `<<' and `>>'.
    # `<=' and `>=' are element-wise!
    constraints = [ Abar.T@P@Abar - P + Qbar << 0 ]  # Set the constraints 
    optimizer = cp.Problem(cp.Minimize(cost), constraints) # build optimizer 
    optimizer.solve()

    print(f"Cost predicted by SDP: {cost.value}")
    for steps_sim in np.arange(4, 22, 2):
        estimated_cost = cost_est_sim(A, B, K, Q, R, x, steps_sim)
        print(f"Cost from {steps_sim:3d}-step simulation: {estimated_cost:10.5f} |  rel. error = {float((estimated_cost - cost.value)/cost.value):3.4e}")


def exercise5(): 
    print("Exercise 5. See class `MPCCvxpy`")

def exercise6():
    print("Exercise 6.")
    run_mpc_simulation()

def exercise7():
    print("Exercise 7.")
    run_mpc_simulation(weak_brakes=True)

if __name__ == "__main__":
    exercise3()
    # exercise5()
    # exercise6()
    # exercise7()