from typing import Callable, Tuple
import sys 
import casadi as cs 
import os
WORKING_DIR = os.path.split(__file__)[0]
sys.path.append(os.path.join(WORKING_DIR, os.pardir))
from parameters import VehicleParameters
from animation import AnimateParking
from plotting import plot_state_trajectory, plot_input_sequence
from matplotlib.patches import Rectangle
from rcracers.simulator.dynamics import KinematicBicycle
import numpy as np
import matplotlib.pyplot as plt 
from rcracers.simulator import simulate

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
        s1 = f(x,u) 
        s2 = f(x + 0.5*ts*s1, u)
        s3 = f(x + 0.5*ts*s2, u)
        s4 = f(x + ts * s3, u) 
        return x + ts/6. * (s1 + 2 * s2 + 2* s3 + s4)
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
        x0 = cs.SX.sym("x0", (4,1))
        x = x0
        
        # Create a decision variable for the controls 
        u = [cs.SX.sym(f"u_{t}", (2,1)) for t in range(self.N)]
        self.u = u 

        # Create the weights for the states 
        Q = cs.diagcat(1, 3, 0.1, 0.01)
        QT = 10 * Q 
        # controls weights matrix
        R = cs.diagcat(1., 1e-2)

        # Initialize the cost 
        cost = 0

        # -- State bounds
        # TODO fill in the state and input bounds 
        states_lb = np.array([params.min_pos_x, params.min_pos_y, params.min_heading, params.min_vel])
        states_ub = np.array([params.max_pos_x, params.max_pos_y, params.max_heading, params.max_vel])
        
        # -- Input bounds 
        inputs_lb = np.array([params.min_drive, -params.max_steer])
        inputs_ub = np.array([params.max_drive, params.max_steer])

        lbx = []
        ubx = []
        g = [] 
        lbg = []
        ubg = []

        # TODO: Implement the dynamical model
        
        ode = KinematicBicycle(params, symbolic=True)
        f = forward_euler(ode, self.ts)

        # Build the cost function 
        for i in range(self.N):
            cost += x.T @ Q @ x + u[i].T @ R @ u[i]
            x = f(x, u[i])
            lbx.append(inputs_lb)
            ubx.append(inputs_ub)
            g.append(x)   # Bound the state 
            lbg.append(states_lb)
            ubg.append(states_ub)
        
        cost += x.T @ QT @ x

        variables = cs.vertcat(*u)
        nlp = {"f": cost,
            "x": variables,
            "g": cs.vertcat(*g),
            "p": x0}
        bounds = {"lbx": cs.vertcat(*lbx), 
                "ubx": cs.vertcat(*ubx), 
                "lbg": cs.vertcat(*lbg), 
                "ubg": cs.vertcat(*ubg), 
                }
    
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


def plot_input_sequence(u_sequence, params: VehicleParameters): 
    plt.subplot(2,2, (1,3))
    plt.title("Control actions")
    plt.plot(u_sequence[:,0], u_sequence[:,1], marker=".") 
    bounds = Rectangle(np.array((params.min_drive, -params.max_steer)), params.max_drive-params.min_drive, 2*params.max_steer, fill=False)
    plt.gca().add_patch(bounds)
    plt.xlabel("$a$")
    plt.ylabel("$\delta$");
    plt.subplot(2,2,2)
    plt.title("Steering angle")
    plt.plot(u_sequence[:,1].squeeze(), marker=".") 
    style=dict(linestyle="--", color="black") 
    plt.axhline(params.max_steer, **style)
    plt.axhline(-params.max_steer, **style)
    plt.ylabel("$\delta$");
    plt.subplot(2,2,4)
    plt.title("Acceleration")
    plt.plot(u_sequence[:,0].squeeze(), marker=".") 
    plt.axhline(params.min_drive, **style)
    plt.axhline(-params.max_drive, **style)
    plt.ylabel("$a$");
    plt.xlabel("$t$")
    plt.tight_layout()


def plot_state_trajectory(x_sequence, title: str = "Trajectory", ax = None, color="tab:blue", label: str=""):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    car_params = VehicleParameters()
    parking_area = Rectangle(-0.5*PARK_DIMS, *PARK_DIMS, ec="tab:green", fill=False)
    ax.add_patch(parking_area)
    extra_arg = dict() 
    for i, xt in enumerate(x_sequence):
        if i==len(x_sequence)-1:
            extra_arg["label"]= label
        if i%2 == 0:  # Only plot a subset 
            alpha = min(0.1 + i / len(x_sequence), 1)
            anchor = xt[:2] - 0.5 * np.array([car_params.length, car_params.width])
            car = Rectangle(anchor, car_params.length, car_params.width, 
                angle=xt[2]/np.pi*180.,
                rotation_point="center",
                alpha=alpha, 
                ec="black",
                fc=color,
                **extra_arg
            )
            ax.add_patch(car)
    plt.legend()
    ax.plot(x_sequence[:,0], x_sequence[:,1], marker=".", color="black")
    ax.set_xlabel("$p_x$ [m]")
    ax.set_ylabel("$p_y$ [m]")
    ax.set_aspect("equal")

def plot_states_separately(x_sequence):
    plt.subplot(4,1,1)
    plt.title("Position x")
    plt.plot(x_sequence[:,0].squeeze(), marker=".") 
    plt.ylabel("$p_x$");
    plt.subplot(4,1,2)
    plt.title("Position y")
    plt.plot(x_sequence[:,1].squeeze(), marker=".") 
    plt.ylabel("$y$")
    plt.subplot(4,1,3)
    plt.title("Angle")
    plt.plot(x_sequence[:,2].squeeze(), marker=".")
    plt.ylabel("$\psi$")
    plt.subplot(4,1,4)
    plt.title("Velocity")
    plt.plot(x_sequence[:,3].squeeze(), marker=".") 
    plt.ylabel("$v$")
    plt.xlabel("$t$")
    plt.tight_layout()
    

def rel_error(val, ref):
    """Compute the relative errors between `val` and `ref`, taking the ∞-norm along axis 1. 
    """
    return np.linalg.norm(
        val - ref, axis=1, ord=np.inf,
    )/0.5*(1e-12 + np.linalg.norm(val, axis=1, ord=np.inf) + np.linalg.norm(ref, axis=1, ord=np.inf))


#-----------------------------------------------------------
# EXERCISES
#-----------------------------------------------------------


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


def exercise4():
    print("Exercise 4")
    N=50
    ts = 0.05
    x0 = np.array([0.6, -0.25, 0, 0])
    print("--Set up the MPC controller")
    controller = MPCController(N=N, ts=ts, params=VehicleParameters())
    
    print("--Solve for the optimal closed-loop control sequence")
    solution = controller.solve(x0)
    controls = controller.reshape_input(solution)

    def open_loop_policy(t): 
        return controls[t]

    # Build the assumed model
    bicycle = KinematicBicycle(VehicleParameters())
    dynamics_assumed = forward_euler(bicycle, ts)
    x_open_loop_model = simulate(x0, dynamics_assumed, n_steps=N, policy=open_loop_policy)

    print("--Build dynamics with parameter mismatch")
    params = VehicleParameters()
    params.friction *= 0.8

    bicycle_true = KinematicBicycle(params)
    dynamics_accurate = exact_integration(bicycle_true, ts)
    print("--Simulate with true model")
    x_open_loop_exact = simulate(x0, dynamics_accurate, n_steps=N, policy=open_loop_policy)

    print("--Plot comparisons")
    plt.figure()
    print("Using the MPC prediction model")
    plot_state_trajectory(x_open_loop_model, color="tab:blue", label="Predicted")
    print("Using the accurate model")
    plot_state_trajectory(x_open_loop_exact, color="tab:red", label="Real")
    plt.title("Trajectory (parameter error)")
    plt.show()

    plt.figure()
    plt.plot(rel_error(x_open_loop_model, x_open_loop_exact)*100)
    plt.xlabel("Time step")
    plt.ylabel("$\| x - x_{pred} \| / \| x \| \\times 100$")
    plt.title("Relative prediction error (parameter error) [%]")
    plt.show()

    print("---Extra: run an animation")
    
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
    
    nstep = 100

    # Build the assumed model
    bicycle = KinematicBicycle(VehicleParameters())
    dynamics_assumed = forward_euler(bicycle, ts)
    print("--Set up the MPC controller")
    controller = MPCController(N=N, ts=ts, params=VehicleParameters())
    
    print("--Simulate the controller in closed-loop (the __call__ method is invoked every time step)")
    x_closed_loop_model = simulate(x0, dynamics_assumed, n_steps=nstep, policy=controller)

    print("--Build system with the true dynamics (including model mismatch!)")
    params = VehicleParameters()
    params.friction *= 0.8
    bicycle_true = KinematicBicycle(params)
    dynamics_accurate = exact_integration(bicycle_true, ts)
    x_closed_loop_exact = simulate(x0, dynamics_accurate, n_steps=nstep, policy=controller)

    print("--Plot comparisons")
    plt.figure()
    print("Using the MPC prediction model")
    plot_state_trajectory(x_closed_loop_model, color="tab:blue", label="Predicted")
    print("Using the accurate model")
    plot_state_trajectory(x_closed_loop_exact, color="tab:red", label="Real")
    plt.title("Trajectory (parameter error)")
    plt.show()

    plt.figure()
    plt.plot(rel_error(x_closed_loop_exact, x_closed_loop_model) * 100)
    plt.xlabel("Time step")
    plt.ylabel("$\| x - x_{pred} \| / \| x \| \\times 100$")
    plt.title("Relative prediction error (parameter error) [%]")
    plt.show()

    print("---Extra: run an animation")
    anim = AnimateParking()
    anim.setup(x_closed_loop_exact, ts)
    anim.add_car_trajectory(x_closed_loop_model, color=(150, 10, 50))
    anim.trace(x_closed_loop_exact)
    anim.run()


if __name__ == "__main__":
    exercise1()
    exercise2()
    exercise3()
    exercise4()
    exercise5()
