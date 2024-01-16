import numpy as np
import numpy.random as npr
import casadi as cs

import matplotlib.pyplot as plt

from problem import (
    get_system_equations,
    get_linear_dynamics,
    build_mhe,
    default_config,
    simulate,
    ObserverLog,
    LOGGER,
)


class EKF:
    def __init__(self, f: cs.Function, h: cs.Function, x0_est, σ_w, σ_v, σ_p, zero_clipping=False) -> None:
        self.f = f
        self.h = h
        self.dfdx, self.dfdw, self.dhdx = get_linear_dynamics(f, h)

        C0 = self.dhdx(x0_est)

        # Integer invariants
        self.n = self.f.size1_out(0)
        self.p = self.h.size1_out(0)

        self.Q = np.eye(self.n) * σ_w**2
        self.R = np.eye(self.p) * σ_v**2
        self.Pk = np.eye(self.n) * σ_p**2

        Pk, R = self.Pk, self.R
        self.Lk = Pk @ C0.T @ np.linalg.inv(C0@Pk@C0.T + R)

        self.x_hat = x0_est
        self.x_bar = x0_est

        self.clip_zero = zero_clipping

    def __call__(self, y: np.ndarray, log: LOGGER):
        """Process a measurement
            Implement EKF using the linearization produced by
                `get_linear_dynamics`.

        :param y: the measurement
        :param log: the logger for output
        """
        Ck = self.dhdx(self.x_bar)

        Pk_m = self.Pk
        self.Lk = Pk_m @ Ck.T @ np.linalg.inv(Ck@Pk_m@Ck.T + self.R)
        self.Pk = Pk_m - self.Lk @ Ck @ Pk_m
        self.x_hat = self.x_bar + self.Lk @ (y - self.h(self.x_bar))

        if self.clip_zero:
            self.x_hat = np.maximum(self.x_hat, 0.0)

        Ak, Gk = self.dfdx(self.x_hat, 0), self.dfdw(self.x_hat, 0),
        self.x_bar = self.f(self.x_hat, 0)
        self.Pk = Ak @ self.Pk @ Ak.T + Gk @ self.Q @ Gk.T

        # log the state estimate and the measurement for plotting
        log("y", y)
        log("x", np.squeeze(self.x_hat))


class MHE:
    def __init__(self, f: cs.Function, h: cs.Function, horizon, σ_w, σ_v, lbx=-np.inf, ubx=np.inf, use_prior=False) -> None:
        """Create an instance of this `MHE`.
        Pass required arguments and build the MHE problem using
            `build_mhe`. You can use the output of `get_system_equations`.
        """
        self.f = f
        self.h = h
        self.N = horizon
        self.lbx, self.ubx = lbx, ubx
        self.use_prior = use_prior

        # Integer invariants
        self.n = self.f.size1_out(0)
        self.p = self.h.size1_out(0)

        self.Q = np.eye(self.n) * σ_w**2
        self.R = np.eye(self.p) * σ_v**2

        self.solver = self.build(horizon)

        self.y_ = []
    
    def loss(self, w, v):
        return w.T @ np.linalg.inv(self.Q) @ w + v.T @ np.linalg.inv(self.R) @ v

    def build(self, horizon):
        return build_mhe(self.loss, self.f, self.h, horizon,
                         lbx=self.lbx, ubx=self.ubx, use_prior=self.use_prior)

    def __call__(self, y: np.ndarray, log: LOGGER):
        """Process a measurement
            Implement MHE using the solver produced by `build_mhe`.

        :param y: the measurement
        :param log: the logger for output
        """
        self.y_.append(y)
        if len(self.y_) > self.N:
            self.y_.pop(0)

        solver = self.solver
        if len(self.y_) < self.N:
            # rebuild MHE with smaller horizon
            solver = self.build(len(self.y_))

        x, w = solver(self.y_)

        # log the state estimate and the measurement for plotting
        log("y", y)
        log("x", x[-1])


def part_1():
    """Implementation for Exercise 1."""
    print("\nExecuting Exercise 1\n" + "-" * 80)
    # problem setup
    cfg = default_config()

    fs, hs = get_system_equations(symbolic=True, noise=(
        0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)

    # setup the extended kalman filter
    ekf = EKF(fs, hs, x0_est=cfg.x0_est, σ_w=cfg.sig_w,
              σ_v=cfg.sig_v, σ_p=cfg.sig_p, zero_clipping=True)

    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    n_steps = 400
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    x = simulate(cfg.x0, f, n_steps, policy=ekf, measure=h, log=log)
    t = np.arange(0, n_steps + 1) * cfg.Ts

    # plot output in `x` and `log.x`
    show_result(t, x, log.x)


def part_2():
    """Implementation for Exercise 2."""
    print("\nExecuting Exercise 2\n" + "-" * 80)
    # problem setup
    cfg = default_config()

    # setup the extended kalman filter
    fs, hs = get_system_equations(symbolic=True, noise=True, Ts=cfg.Ts, rg=cfg.rg)

    N = 25
    mhe = MHE(fs, hs, horizon=N, σ_w=cfg.sig_w,
              σ_v=cfg.sig_v, lbx=0.0, ubx=10.0)

    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    n_steps = 400
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    x = simulate(cfg.x0, f, n_steps, policy=mhe, measure=h, log=log)
    t = np.arange(0, n_steps + 1) * cfg.Ts

    # plot output in `x` and `log.x`
    show_result(t, x, log.x)


def part_3():
    """Implementation for Homework."""
    print("\nExecuting Homework\n" + "-" * 80)
    # problem setup
    cfg = default_config()

    # setup the extended kalman filter
    mhe = MHE()

    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    x = simulate(cfg.x0, f, n_steps=400, policy=mhe, measure=h, log=log)

    # plot output in `x` and `log.x`

    print(f'{x.shape=}')
    print(f'{log.x.shape=}')


def show_result(t: np.ndarray, x: np.ndarray, x_: np.ndarray):
    _, ax = plt.subplots(1, 1)
    c = ["C0", "C1", "C2"]
    h = []
    for i, c in enumerate(c):
        h += ax.plot(t, x_[..., i], "--", color=c)
        h += ax.plot(t, x[..., i], "-", color=c)
    ax.set_xlim(t[0], t[-1])
    if np.max(x_) >= 10.0:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.legend(
        h,
        [
            "$A_{\mathrm{est}}$",
            "$A$",
            "$B_{\mathrm{est}}$",
            "B",
            "$C_{\mathrm{est}}$",
            "C",
        ],
        loc="lower left",
        mode="expand",
        ncol=6,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        borderaxespad=0,
    )
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.show()


if __name__ == "__main__":
    # part_1()
    part_2()
    # part_3()
