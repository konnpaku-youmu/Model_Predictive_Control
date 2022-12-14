import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
from rcracers.simulator import simulate
from rcracers.simulator.dynamics import KinematicBicycle
from plotting import *
from animation import AnimateParking
from parameters import VehicleParameters
from typing import Callable, Tuple
import sys
import casadi as cs
import os
WORKING_DIR = os.path.split(__file__)[0]
sys.path.append(os.path.join(WORKING_DIR, os.pardir))


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})

# w x h of the parking spot. Just for visualization purposes.
PARK_DIMS = np.array((0.25, 0.12))

π = np.pi


class MPCController:

    def __init__(self, N: int, ts: np.float64, params: VehicleParameters, model, x_obs: np.ndarray) -> None:
        self.N = N      # Horizon
        self.ts = ts    # Sampling time

        self.nlp, self.bounds = self.build_ocp(
            nx=4, nu=2, model=model, params=params, x_obs=x_obs)

        opts = {"ipopt": {"print_level": 1}, "print_time": False}
        self.ipopt_solver = cs.nlpsol("solver", "ipopt", self.nlp, opts)

    def build_ocp(self, nx, nu, model, params: VehicleParameters, x_obs: np.ndarray) -> Tuple[dict, dict]:
        x0 = cs.SX.sym("x0", (nx, 1))
        x = x0

        # create a vector for storing the control input
        u = [cs.SX.sym(f"u_{t}", (2, 1)) for t in range(self.N)]

        # anti-collision radius: r^2
        n_c = 3
        c, r = create_cover_circles(params.length, params.width, n_c)
        c_p, r_p = create_cover_circles(params.length, params.width, n_c)
        r2 = (r + r_p)**2

        T_obs = x2T(x_obs, True)
        circ_obs = [T_obs@center for center in c_p]

        # définir les contraintes d'état
        lb_states = np.array(
            [params.min_pos_x, params.min_pos_y, params.min_heading, params.min_vel])
        ub_states = np.array(
            [params.max_pos_x, params.max_pos_y, params.max_heading, params.max_vel])

        # définir les contraintes anti-collision
        lb_anti_colli = np.ones((n_c*n_c, 1)) * r2
        ub_anti_colli = np.ones((n_c*n_c, 1)) * np.inf

        # définir les contraintes d'entrée
        lb_inputs = np.array([params.min_drive, -params.max_steer])
        ub_inputs = np.array([params.max_drive, params.max_steer])

        # définir les matrice Q, Q_N et R
        Q = np.diag([1., 6., 0.2, 0.05])
        Q_N = 100*Q
        R = np.diag([1, 0.01])

        f = fwd_euler(model, self.ts)

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

            T = x2T(x, True)
            circ_vehicle = [T@center for center in c]

            constraints = [
                cs.norm_2(c_v - c_o)**2 for c_v in circ_vehicle for c_o in circ_obs]
            colli_constraints = cs.vertcat(*constraints)

            g.append(colli_constraints)
            lbg.append(lb_anti_colli)
            ubg.append(ub_anti_colli)

        cost += x.T@Q_N@x

        nlp = dict(f=cost, x=cs.vertcat(*u), g=cs.vertcat(*g), p=x0)

        bounds = dict(lbx=cs.vertcat(*lbx), ubx=cs.vertcat(*ubx),
                      lbg=cs.vertcat(*lbg), ubg=cs.vertcat(*ubg))

        return nlp, bounds

    def solve(self, x) -> dict:
        return self.ipopt_solver(p=x, **self.bounds)

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


def fwd_euler(f, ts) -> Callable:
    def fw_eul(x, u):
        return x + f(x, u) * ts
    return fw_eul


def runge_kutta4(f, ts) -> Callable:
    def rk4_dyn(x, u):
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
        def f_wrap(x, t): return np.array(f(x, u)).reshape([x.size])
        y = odeint(f_wrap, x.reshape([x.size]), [0, ts])
        return y[-1].reshape((x.size,))
    return dt_dyn


def x2T(x: np.array, symbolic: bool):

    ops = cs if symbolic else np

    φ = x[2]

    T = cs.SX.zeros(3, 3) if symbolic else np.zeros((3, 3))

    T[0, 0] = ops.cos(φ)
    T[0, 1] = -ops.sin(φ)
    T[1, 0] = ops.sin(φ)
    T[1, 1] = ops.cos(φ)
    T[0, 2] = x[0]
    T[1, 2] = x[1]

    return T


def create_cover_circles(l: np.float64, w: np.float64, n_c: int):
    d = l / (2*n_c)
    r = np.sqrt(d**2 + (w**2) / 4)

    center = []

    for k in range(n_c):
        center.append(np.array([(2*k + 1) * d - l/2, 0, 1]))  # Homogeneous

    return center, r


def plot_cover_circle(x: np.float64, l: np.float64, w: np.float64, center: list[np.ndarray], radius: np.float64) -> None:
    ax = plt.gca()

    for c in center:
        circ = patch.Circle(c, radius, fill=True,
                            linewidth=1.5, color="#445491", fc="#738ADE", alpha=0.5)
        ax.add_patch(circ)

    vehicle_rect = patch.Rectangle(
        [x[0]-l/2, x[1]-w/2], l, w, angle=np.rad2deg(x[2]), rotation_point="center",
        fill=False, linewidth=2.5, color="#DB8276")
    ax.add_patch(vehicle_rect)
    ax.set_aspect("equal")

    # plt.title(r"Covering circle: with vehicle pose")
    # plt.xlabel(r"$x$")
    # plt.xlim(-2, 6)
    # plt.ylabel(r"$y$")
    # plt.ylim(-1, 5)
    plt.show()


def test_circle() -> None:
    l = 4
    w = 2
    n_c = 3
    center, radius = create_cover_circles(l, w, n_c)

    x = np.array([2, 2, π/4, 0])
    T = x2T(x, False)

    center_T = []
    for c in center:
        center_T.append(T@c)

    plot_cover_circle(x, l, w, center_T, radius)


def main():
    horizon = 30
    steps = 100
    ts = 0.08
    params = VehicleParameters()

    x_obs = np.array([0.25, 0, 0., 0.])
    x0 = np.array([0.3, -0.1, 0., 0.])

    controller = MPCController(N=horizon, ts=ts, params=params, model=KinematicBicycle(
        params=params, symbolic=True), x_obs=x_obs)

    solution = controller.solve(x0)
    controls = controller.reshape_input(solution)

    def open_loop_policy(t):
        return controls[t]

    # Build the assumed model
    bicycle = KinematicBicycle(VehicleParameters())
    dynamics_assumed = fwd_euler(bicycle, ts)

    # print(f"--Simulate under the assumed model")
    # x_open_loop_model = simulate(
    #     x0, dynamics_assumed, n_steps=100, policy=controller)

    # With more accurate predictions:
    print(f"--Simulate using more precise integration")
    dynamics_accurate = exact_integration(bicycle, ts)
    x_open_loop_exact = simulate(
        x0, dynamics_accurate, n_steps=100, policy=controller)

    print(f"--Plotting the results")

    print(f"---Plot Controls")
    plot_input_sequence(controls, VehicleParameters())
    plt.show()
    print(f"---Plot trajectory under the predictions")

    # plot_state_trajectory(
    #     x_open_loop_model, color="tab:blue", label="Predicted")
    # print("---Plot the trajectory under the more accurate model")
    plot_state_trajectory(x_open_loop_exact, color="tab:red", label="Real", park_dims=PARK_DIMS)
    plot_state_trajectory(np.vstack([x_obs]*100), color="#86EBA0", label="Obstacle")

    plt.title("Trajectory (integration error)")
    plt.show()

    # anim = AnimateParking()
    # anim.setup(x_open_loop_exact, ts)
    # anim.add_car_trajectory(x_open_loop_model, color=(150, 10, 50))
    # anim.add_car_trajectory(np.array([[0.25, 0, 0., 0.]]), color=(50, 100, 50))
    # anim.trace(x_open_loop_exact)
    # anim.run()

if __name__ == "__main__":
    main()
