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
    def __init__(self, f, h) -> None:
        
        self.dfdx, self.dfdw, self.dhdx = get_linear_dynamics(f, h)
        self.P_k = cs.SX(3, 3)
        self.R = np.diag([1, 1, 1])


    def __call__(self, y: np.ndarray, log: LOGGER):

        # log the state estimate and the measurement for plotting
        log("y", y)
        log("x", np.zeros(3))
        A_k = self.dfdx(y, 0)
        B_k = self.dfdw(y, 0)
        C_k = self.dhdx(y, 0)

        self.P_k = self.P_k - self.P_k@C_k.T@cs.inv(C_k@self.P_k@C_k.T + self.R)@C_k@self.P_k
        L_k = self.P_k@C_k.T@cs.inv(C_k@self.P_k@C_k.T + self.R)
        x = y + L_k@()


class MHE:
    def __init__(self) -> None:
        """Create an instance of this `MHE`.
        TODO: Pass required arguments and build the MHE problem using
            `build_mhe`. You can use the output of `get_system_equations`.
        """
        print("UNIMPLEMENTED: MHE.")
        print("Try using:  `build_mhe`:")
        print(build_mhe.__doc__)

    def __call__(self, y: np.ndarray, log: LOGGER):
        """Process a measurement
            TODO: Implement MHE using the solver produced by `build_mhe`.

        :param y: the measurement
        :param log: the logger for output
        """
        # log the state estimate and the measurement for plotting
        log("y", y)
        log("x", np.zeros(3))


def part_1():
    """Implementation for Exercise 1."""
    print("\nExecuting Exercise 1\n" + "-" * 80)
    # problem setup
    cfg = default_config()

    # setup the extended kalman filter
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    ekf = EKF(f, h)

    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    x = simulate(cfg.x0, f, n_steps=100, policy=ekf, measure=h, log=log)

    # plot output in `x` and `log.x`
    print(f'{x.shape=}')
    print(f'{log.x.shape=}')


def part_2():
    """Implementation for Exercise 2."""
    print("\nExecuting Exercise 2\n" + "-" * 80)
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


def part_3():
    """Implementation for Homework."""
    print("\nExecuting Homework\n" + "-" * 80)
    # problem setup
    cfg = default_config()

    # setup the extended kalman filter
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    mhe = MHE(f, h, cfg)

    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    x = simulate(cfg.x0, f, n_steps=400, policy=mhe, measure=h, log=log)

    # plot output in `x` and `log.x`
    print(f'{x.shape=}')
    print(f'{log.x.shape=}')


if __name__ == "__main__":
    part_1()
    part_2()
    part_3()
