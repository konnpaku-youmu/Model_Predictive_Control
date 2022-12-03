from typing import Callable, Tuple
import sys 
import casadi as cs 
import os
WORKING_DIR = os.path.split(__file__)[0]
sys.path.append(os.path.join(WORKING_DIR, os.pardir))
from parameters import VehicleParameters
from animation import AnimateParking
from plotting import *

from rcracers.simulator.dynamics import KinematicBicycle
from rcracers.simulator import simulate

import numpy as np
import matplotlib.pyplot as plt 

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})

PARK_DIMS = np.array((0.25, 0.12)) # w x h of the parking spot. Just for visualization purposes. 

#-----------------------------------------------------------
# INTEGRATION
#-----------------------------------------------------------

def forward_euler(f, ts) -> Callable:
    def fw_eul(x,u):
        return x + f(x,u) * ts
    return fw_eul

def runge_kutta4(f, ts) -> Callable:
    def rk4_dyn(x,u):
        s_1 = f(x, u)
        s_2 = f(x + (ts / 2) * s_1, u)
        s_3 = f(x + (ts / 2) * s_2, u)
        s_4 = f(x + ts * s_3, u)
        x_next = x + (ts / 6) * (s_1 + 2*s_2 + 2*s_3 + s_4)
        return x_next 
    
    return rk4_dyn

def exact_integration(f, ts) -> Callable:
    """Ground truth for the integration
    
    Integrate the given dynamics using scipy.integrate.odeint, which is very accurate in 
    comparison to the methods we implement in this settings, allowing it to serve as a 
    reference to compare against.

    Args:
        f (dynamics): The dynamics to integrate (x,u) -> xdot
        ts (_type_): Sampling time 

    Returns:
        Callable: Discrete-time dynamics (x, u) -> x+ 
    """
    from scipy.integrate import odeint  # Load scipy integrator as a ground truth
    def dt_dyn(x, u):
        f_wrap = lambda x, t: np.array(f(x, u)).reshape([x.size])
        y = odeint(f_wrap, x.reshape([x.size]), [0, ts])
        return y[-1].reshape((x.size,))
    return dt_dyn

def build_test_policy():
    # Define a policy to test the system
    acceleration = 1 # Fix a constant longitudinal acceleration 
    policy = lambda y, t: np.array([acceleration, 0.1 * np.sin(t)])
    return policy

#-----------------------------------------------------------
# MPC CONTROLLER
#-----------------------------------------------------------


class MPCController:

    def __init__(self, N: int, ts: float, *, params: VehicleParameters):
        """Constructor.

        Args:
            N (int): Prediction horizon
            ts (float): sampling time [s]
        """
        self.N = N
        self.ts = ts 
        nlp_dict, self.bounds = self.build_ocp(params)
        
        opts = {"ipopt": {"print_level": 1}, "print_time": False}
        self.ipopt_solver = cs.nlpsol("solver", "ipopt", nlp_dict, opts) 
        
    def solve(self, x) -> dict:
        return self.ipopt_solver(p=x, **self.bounds)
        
    def build_ocp(self, params: VehicleParameters) -> Tuple[dict, dict]:
        """
        TODO IMPLEMENT

        Given the prediction horizon, build a nonlinear program that represents the parametric optimization problem described above, with the initial state x as a parameter. 
        Use a single shooting formulation, i.e., do not define a new decision variable for the states, but rather write them as functions of the initial state and the control variables. Also return the lower bound and upper bound on the decision variables and constraint functions:

        Args: 
            horizon [int]: decision horizon  
        Returns: 
            solver [dict]: the nonlinear program as a dictionary: 
                {"f": [cs.SX] cost (as a function of the decision variables, built as an expression, e.g., x + y, where x and y are CasADi SX.sym objects),
                "g": [cs.Expression] nonlinear constraint function as 
                an expression of the variables and the parameters. 
                These constraints represent the bounds on the state. 
                "x": [cs.SX] decision_vars (all control actions over the prediction horizon (concatenated into a long vector)), 
                "p": [cs.SX] parameters (initial state vector)} 
            bounds [dict]: the bounds on the constraints 
                {"lbx": [np.ndarray] Lower bounds on the decision variables, 
                "ubx": [np.ndarray] Upper bounds on the decision variables, 
                "lbg": [np.ndarray] Lower bounds on the nonlinear constraint g, 
                "ubg": [np.ndarray] Upper bounds on the nonlinear constraint g 
                }
        """
        # Create a parameter for the initial state. 
        nx = 4
        nu = 2
        x0 = cs.SX.sym("x0", (nx,1))
        x = x0
        u = [cs.SX.sym(f"u_{t}", (2, 1)) for t in range(self.N)]

        # define the input constraints
        lb_inputs = np.array([params.min_drive, -params.max_steer])
        ub_inputs = np.array([params.max_drive, params.max_steer])

        # define the state constraints
        lb_states = np.array([params.min_pos_x, params.min_pos_y, params.min_vel, params.min_heading])
        ub_states = np.array([params.max_pos_x, params.max_pos_y, params.max_vel, params.max_heading])

        Q = np.diag(np.array([1, 3, 0.1, 0.01]))
        Q_N = 5*Q
        R = np.diag(np.array([1, 0.01]))

        model = KinematicBicycle(params=params, symbolic=True)
        # f = forward_euler(model, ts=self.ts)
        f = runge_kutta4(model, ts=self.ts)

        cost = 0

        lbx = []
        ubx = []
        g = []
        lbg = []
        ubg = []

        for i in range(self.N):
            cost += x.T@Q@x + u[i].T@R@u[i]
            x = f(x, u[i])
            lbx.append(lb_inputs)
            ubx.append(ub_inputs)
            g.append(x)
            lbg.append(lb_states)
            ubg.append(ub_states)
            
        cost += x.T@Q_N@x

        nlp = dict(f = cost, x = cs.vertcat(*u), g = cs.vertcat(*g), p = x0)
        bounds = dict(lbx = cs.vertcat(*lbx), ubx = cs.vertcat(*ubx), lbg = cs.vertcat(*lbg), ubg = cs.vertcat(*ubg))

        return nlp, bounds

    def reshape_input(self, sol):
        return np.reshape(sol["x"], ((-1, 2)))

    def __call__(self, y):
        """Solve the OCP for initial state y.

        Args:
            y (np.ndarray): Measured state 
        """
        solution = self.solve(y)
        u = self.reshape_input(solution)
        return u[0]


#-----------------------------------------------------------
# UTILITIES
#-----------------------------------------------------------

def build_test_policy():
    # Define a policy to test the system
    acceleration = 1 # Fix a constant longitudinal acceleration 
    policy = lambda y, t: np.array([acceleration, 0.1 * np.sin(t)])
    return policy

def compare_open_loop(ts: float, x0: np.ndarray, steps: int): 
    """Compare the open-loop predictions using different discretization schemes.

    Args:
        ts (float): Sampling time (s) 
        x0 (np.ndarray): Initial state
        steps (int): Number of steps to predict
    """
    params = VehicleParameters()
    kin_bicycle = KinematicBicycle(params)
    rk4_discrete_time = runge_kutta4(kin_bicycle, ts)
    fe = forward_euler(kin_bicycle, ts)
    gt_discrete_time = exact_integration(kin_bicycle, ts)
    
    test_policy = build_test_policy()

    # Plot the results
    _, axes = plt.subplots(constrained_layout = True)
    axes.set_xlabel("$p_{x}$")
    axes.set_ylabel("$p_{y}$")
    axes.set_title(f"Position trajectories Ts = {ts}")
    results = dict()
    for name, dynamics in {"Forward Euler": fe, "RK 4": rk4_discrete_time, "Ground truth": gt_discrete_time}.items():
        states = simulate(x0, dynamics, steps, policy=test_policy)
        axes.plot(states[:,0], states[:,1], label=name, linestyle="--")
        results[name] = states

    axes.legend()
    
    # Plot the errors
    plt.figure()
    plt.xlabel("Time step $k$")
    plt.ylabel("$\| x_k - \hat{x}_k\|$")
    for name, dynamics in {"Forward Euler": fe, "RK 4": rk4_discrete_time}.items():
        error = np.linalg.norm(results["Ground truth"] - results[name], axis=1)
        plt.semilogy(error, label=name)
    
    plt.legend()
    plt.title(f"Open loop prediction errors ($T_s = {ts}s$)")
    plt.show()


def rel_error(val, ref):
    """Compute the relative errors between `val` and `ref`, taking the ∞-norm along axis 1. 
    """
    return np.linalg.norm(
        val - ref, axis=1, ord=np.inf,
    )/0.5*(1e-12 + np.linalg.norm(val, axis=1, ord=np.inf) + np.linalg.norm(ref, axis=1, ord=np.inf))


#-----------------------------------------------------------
# EXERCISES
#-----------------------------------------------------------

def print_incomplete(i: int, msg: str):
    print(f"Exercise {i} incomplete.\n\n{msg}")

def exercise1(): 
    print("Exercise 1. See implementation of runge_kutta4.")
    sample_times = (0.05, 0.1, 0.5)

    x0 = np.zeros(4)
    steps = 100

    for ts in sample_times:
        compare_open_loop(ts, x0, steps)
    
def exercise2():
    print("Exercise 2 -- Pen and paper.")

def exercise3():
    print("Exercise 3")
    N=50
    ts = 0.05
    x0 = np.array([0.6, -0.25, 0, 0])

    print("--Set up the MPC controller")
    controller = MPCController(N=N, ts=ts, params=VehicleParameters())
    
    print(f"--Solve the OCP for x0 = {x0}")
    solution = controller.solve(x0)
    controls = controller.reshape_input(solution)

    def open_loop_policy(t):
        return controls[t]

    # Build the assumed model 
    bicycle = KinematicBicycle(VehicleParameters())
    dynamics_assumed = forward_euler(bicycle, ts)

    print(f"--Simulate under the assumed model")
    x_open_loop_model = simulate(x0, dynamics_assumed, n_steps=N, policy=open_loop_policy)

    # With more accurate predictions: 
    print(f"--Simulate using more precise integration")
    dynamics_accurate = exact_integration(bicycle, ts)
    x_open_loop_exact = simulate(x0, dynamics_accurate, n_steps=N, policy=open_loop_policy)

    print(f"--Plotting the results")

    print(f"---Plot Controls")
    plot_input_sequence(controls, VehicleParameters())
    plt.show()
    print(f"---Plot trajectory under the predictions")
    plot_state_trajectory(x_open_loop_model, color="tab:blue", label="Predicted")
    print("---Plot the trajectory under the more accurate model")
    plot_state_trajectory(x_open_loop_exact, color="tab:red", label="Real")
    plt.title("Trajectory (integration error)")
    plt.show()

    print(f"---Plot trajectory under the predictions")
    plt.figure()
    plt.plot(rel_error(x_open_loop_model, x_open_loop_exact) * 100)
    plt.xlabel("Time step")
    plt.ylabel("$\| x - x_{pred} \| / \| x \| \\times 100$")
    plt.title("Relative prediction error (integration error) [%]")
    plt.show()

    anim = AnimateParking()
    anim.setup(x_open_loop_exact, ts)
    anim.add_car_trajectory(x_open_loop_model, color=(150, 10, 50))
    anim.trace(x_open_loop_exact)
    anim.run()

def exercise4():
    print("Exercise 3")
    N=50
    ts = 0.05
    x0 = np.array([0.6, -0.25, 0, 0])

    print("--Set up the MPC controller")
    controller = MPCController(N=N, ts=ts, params=VehicleParameters())
    
    print(f"--Solve the OCP for x0 = {x0}")
    solution = controller.solve(x0)
    controls = controller.reshape_input(solution)

    def open_loop_policy(t):
        return controls[t]

    # Build the assumed model 
    bicycle = KinematicBicycle(VehicleParameters())
    dynamics_assumed = forward_euler(bicycle, ts)

    print(f"--Simulate under the assumed model")
    x_open_loop_model = simulate(x0, dynamics_assumed, n_steps=N, policy=open_loop_policy)

    # With more accurate predictions: 
    print(f"--Simulate using more precise integration")
    dynamics_accurate = exact_integration(bicycle, ts)
    x_open_loop_exact = simulate(x0, dynamics_accurate, n_steps=N, policy=open_loop_policy)

    print(f"--Plotting the results")

    print(f"---Plot Controls")
    plot_input_sequence(controls, VehicleParameters())
    plt.show()
    print(f"---Plot trajectory under the predictions")
    plot_state_trajectory(x_open_loop_model, color="tab:blue", label="Predicted")
    print("---Plot the trajectory under the more accurate model")
    plot_state_trajectory(x_open_loop_exact, color="tab:red", label="Real")
    plt.title("Trajectory (integration error)")
    plt.show()

    print(f"---Plot trajectory under the predictions")
    plt.figure()
    plt.plot(rel_error(x_open_loop_model, x_open_loop_exact) * 100)
    plt.xlabel("Time step")
    plt.ylabel("$\| x - x_{pred} \| / \| x \| \\times 100$")
    plt.title("Relative prediction error (integration error) [%]")
    plt.show()

    anim = AnimateParking()
    anim.setup(x_open_loop_exact, ts)
    anim.add_car_trajectory(x_open_loop_model, color=(150, 10, 50))
    anim.trace(x_open_loop_exact)
    anim.run()
    

def exercise5():
    print("Exercise 5")
    N=50
    ts = 0.05
    x0 = np.array([0.6, -0.25, 0, 0])

    print("--Set up the MPC controller")
    controller = MPCController(N=N, ts=ts, params=VehicleParameters())
    
    print(f"--Solve the OCP for x0 = {x0}")
    solution = controller.solve(x0)
    controls = controller.reshape_input(solution)

    def open_loop_policy(t):
        return controls[t]

    # Build the assumed model 
    bicycle = KinematicBicycle(VehicleParameters())
    dynamics_assumed = forward_euler(bicycle, ts)

    print(f"--Simulate under the assumed model")
    x_open_loop_model = simulate(x0, dynamics_assumed, n_steps=N, policy=controller)

    # With more accurate predictions: 
    print(f"--Simulate using more precise integration")
    dynamics_accurate = exact_integration(bicycle, ts)
    x_open_loop_exact = simulate(x0, dynamics_accurate, n_steps=N, policy=controller)

    print(f"--Plotting the results")

    print(f"---Plot Controls")
    plot_input_sequence(controls, VehicleParameters())
    plt.show()
    print(f"---Plot trajectory under the predictions")
    plot_state_trajectory(x_open_loop_model, color="tab:blue", label="Predicted")
    print("---Plot the trajectory under the more accurate model")
    plot_state_trajectory(x_open_loop_exact, color="tab:red", label="Real")
    plt.title("Trajectory (integration error)")
    plt.show()

    print(f"---Plot trajectory under the predictions")
    plt.figure()
    plt.plot(rel_error(x_open_loop_model, x_open_loop_exact) * 100)
    plt.xlabel("Time step")
    plt.ylabel("$\| x - x_{pred} \| / \| x \| \\times 100$")
    plt.title("Relative prediction error (integration error) [%]")
    plt.show()

    anim = AnimateParking()
    anim.setup(x_open_loop_exact, ts)
    anim.add_car_trajectory(x_open_loop_model, color=(150, 10, 50))
    anim.trace(x_open_loop_exact)
    anim.run()

if __name__ == "__main__":
    # exercise1()
    # exercise3()
    # exercise4()
    exercise5()
