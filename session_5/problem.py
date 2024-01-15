from dataclasses import dataclass
from typing import List, Tuple
import warnings
import numpy as np
import numpy.random as npr
import casadi as cs

from rcracers.simulator.core import rk4, BaseControllerLog, list_field
from rcracers.control.signatures import LOGGER, POLICY
from rcracers.simulator.core import simulate


@dataclass
class Config:
    """Experiment configuration."""

    Ts: float = 0.25  # time step
    x0: np.ndarray = np.array([0.5, 0.05, 0.0])  # initial state of simulator
    x0_est: np.ndarray = np.array([1, 0, 4])  # initial state for estimator
    sig_w: float = 0.002  # model noise
    sig_v: float = 0.25  # measurement noise
    sig_p: float = 0.5  # state confidence
    seed: int = 1024  # rng seed
    rg: npr.Generator = None

    def __post_init__(self):
        self._rg = npr.default_rng(self.seed)


@dataclass
class ObserverLog(BaseControllerLog):
    """Observer Log."""

    x: List[np.ndarray] = list_field()  # state estimate
    y: List[np.ndarray] = list_field()  # measurement

    def finish(self):
        self.x = np.array(self.x)
        self.y = np.array(self.y)


def default_config() -> Config:
    """Generate exercise configuration."""
    return Config()


def get_system_equations(
    *,
    symbolic: bool = False,
    Ts: float = 0.25,
    noise: float = None,
    rg: npr.Generator = None,
):
    """Generate the system equations for the batch chemical reactor

    :param symbolic: support symbolic variables, defaults to False
    :param Ts: time step, defaults to 0.01
    :param noise: model noise standard deviation (σw, σv), defaults to None (i.e. no noise)
    :param rg: random number generator, defaults to None
    """
    NX, NY = 3, 1
    if rg is None:
        rg = npr.default_rng()

    @rk4(Ts=Ts)
    def dynamics(x):
        k1, km1, k2, km2 = 0.5, 0.05, 0.2, 0.01
        ca, cb, cc = x[0], x[1], x[2]
        r1, r2 = k1 * ca - km1 * cb * cc, k2 * cb**2 - km2 * cc
        res = (-r1, r1 - 2 * r2, r1 + r2)
        if symbolic:
            return cs.vertcat(*res)
        return np.array(res)

    def measure(x):
        return 32.84 * (x[0] + x[1] + x[2])

    if symbolic:
        x = cs.SX.sym("x", NX)
        if noise:
            w = cs.SX.sym("w", NX)
            f = cs.Function("f", [x, w], [dynamics(x) + w])
            h = cs.Function("h", [x], [measure(x)])
        else:
            f = cs.Function("f", [x], [dynamics(x)])
            h = cs.Function("h", [x], [measure(x)])
        return f, h

    if noise:
        f = lambda x: np.maximum(
            dynamics(x) + noise[0] * rg.normal(size=(NX,)), np.zeros(NX)
        )
        h = lambda x: measure(x) + noise[1] * rg.normal(size=(NY,))
    else:
        f = lambda x: np.maximum(dynamics(x), np.zeros(NX))
        h = measure
    return f, h


def system_info(f: cs.Function, h: cs.Function):
    """Get system dimensions from provided dynamics.

    :param f: dynamics
    :param h: measurement
    :return: nx, nw, ny
    """
    if f.n_in() != 2:
        raise ValueError(f"Expected f to have 2 arguments got {f.n_in()}.")
    if h.n_in() != 1:
        raise ValueError(f"Expected h to have 1 argument got {h.n_in()}.")

    return f.size1_in(0), f.size1_in(1), h.size1_out(0)


def get_linear_dynamics(f: cs.Function, h: cs.Function) -> Tuple[cs.DM]:
    """Get linearized dynamics.

    :param f: state dynamics
    :param h: measurement model
    :return: dfdx, dfdw, dhdx as `cs.Function` instances

    Example:
    >>> fs, hs = get_system_equations(symbolic=True, noise=True)
    >>> dfdx, dfdw, dhdx = get_linear_dynamics(fs, hs)
    """
    nx, nw, _ = system_info(f, h)
    x, w = cs.SX.sym("x", nx), cs.SX.sym("w", nw)

    dfdx = cs.Function("dfdx", [x, w], [cs.jacobian(f(x, w), x)])
    dfdw = cs.Function("dfdx", [x, w], [cs.jacobian(f(x, w), w)])
    dhdx = cs.Function("dhdx", [x], [cs.jacobian(h(x), x)])

    return dfdx, dfdw, dhdx


def build_mhe(
    loss: cs.Function,
    f: cs.Function,
    h: cs.Function,
    horizon: int,
    *,
    lbx: np.ndarray = -np.inf,
    ubx: np.ndarray = np.inf,
    use_prior: bool = False,
):
    """Build the MHE problem

    :param loss: loss function (w, v) -> float
    :param f: dynamics (x, w) -> x+
    :param h: measurement model x -> y
    :param horizon: measurement window
    :param lbx: lower bound on state, defaults to -np.inf
    :param ubx: upper bound on state, defaults to np.inf
    :param use_prior: use prior cost, defaults to False
    :return: solver

    Example:
    >>> # mhe without prior
    >>> f, h = get_system_equations(symbolic=True, noise=True)
    >>> loss = lambda w, v: w.T @ w + v.T @ v
    >>> solver = build_mhe(loss, f, h, 10, lbx=0.0, ubx=10.0, use_prior=False)
    >>> x, w = solver(y=np.zeros((10, 1)))

    Example:
    >>> # mhe with prior
    >>> f, h = get_system_equations(symbolic=True, noise=True)
    >>> loss = lambda w, v: w.T @ w + v.T @ v
    >>> solver = build_mhe(loss, f, h, 10, lbx=0.0, ubx=10.0, use_prior=True)
    >>> x, w = solver(P=np.eye(3), x0=np.zeros(3), y=np.zeros((10, 1)))
    """
    # prepare
    nx, nw, ny = system_info(f, h)
    w, v = cs.SX.sym("w", nw), cs.SX.sym("v", ny)
    if not isinstance(loss, cs.Function):
        loss = cs.Function("l", [w, v], [loss(w, v)])

    # process bounds
    if isinstance(lbx, float):
        lbx = np.full((horizon, nx), lbx)
    if isinstance(ubx, float):
        ubx = np.full((horizon, nx), ubx)
    if lbx.ndim == 1:
        lbx = np.repeat(lbx[np.newaxis, :], horizon, axis=0)
    if ubx.ndim == 1:
        ubx = np.repeat(ubx[np.newaxis, :], horizon, axis=0)
    lbx, ubx = lbx[: horizon + 1, :], ubx[: horizon + 1, :]

    # get the variables
    w = [cs.SX.sym(f"w_{t}", nw) for t in range(horizon)]
    x = [cs.SX.sym(f"x_{t}", nx) for t in range(horizon + 1)]
    variables = cs.vertcat(*x, *w)

    # setup the parameters
    y = [cs.SX.sym(f"y_{t}", ny) for t in range(horizon)]
    parameters = cs.vertcat(*y)
    if use_prior:
        hessian = cs.SX.sym("P", nx**2)
        x0 = cs.SX.sym("x0_prior", nx)
        parameters = cs.vertcat(hessian, x0, parameters)

    # setup cost
    cost = sum([loss(w[t], y[t] - h(x[t])) for t in range(horizon)])
    if use_prior:
        error = x[0] - x0
        cost += error.T @ cs.solve(cs.reshape(hessian, nx, nx), error)

    # setup constraints
    g = [x[t + 1] - f(x[t], w[t]) for t in range(horizon)]
    constraints = cs.vertcat(*g)

    # setup bounds
    lbx = cs.vertcat(np.reshape(lbx, (-1)), np.full((nw * (horizon + 1)), -np.inf))
    ubx = cs.vertcat(np.reshape(ubx, (-1)), np.full((nw * (horizon + 1)), np.inf))
    lbg = np.zeros((nx * horizon))
    ubg = np.zeros((nx * horizon))

    # gather data
    nlp = {"f": cost, "x": variables, "g": constraints, "p": parameters}
    bounds = {"lbx": lbx, "ubx": ubx, "lbg": lbg, "ubg": ubg}
    opts = {"ipopt": {"print_level": 1}, "print_time": False}
    solver = cs.nlpsol("solver", "ipopt", nlp, opts)

    # build function handler
    if use_prior:

        def evaluator(P: np.ndarray, x0: np.ndarray, y: List[np.ndarray]):
            p = cs.vertcat(cs.reshape(P, nx**2, 1), x0, *y)
            sol = solver(p=p, **bounds)
            x = np.reshape(sol["x"][: nx * (horizon + 1)], (horizon + 1, nx))
            w = np.reshape(sol["x"][nx * (horizon + 1) :], (horizon, nw))
            return x, w

    else:

        def evaluator(y: List[np.ndarray]):
            p = cs.vertcat(*y)
            sol = solver(p=p, **bounds)
            x = np.reshape(sol["x"][: nx * (horizon + 1)], (horizon + 1, nx))
            w = np.reshape(sol["x"][nx * (horizon + 1) :], (horizon, nw))
            return x, w

    return evaluator