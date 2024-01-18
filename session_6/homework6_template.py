
from typing import Callable
from given.homework import problem
import numpy as np

import os
WORKING_DIR = os.path.split(__file__)[0]


def lqr_factor_step(N: int, nl: problem.NewtonLagrangeQP) -> problem.NewtonLagrangeFactors:
    Q, S, R, q, r, A, B, c = nl.Qk, nl.Sk, nl.Rk, nl.qk, nl.rk, nl.Ak, nl.Bk, nl.ck

    K, s, P, e= np.zeros([R.shape[0], R.shape[1], S.shape[2]]), np.zeros([q.shape[0] + 1, q.shape[1], q.shape[2]]) , np.zeros([Q.shape[0] + 1, Q.shape[1], Q.shape[2]]), np.zeros_like(R)
    P[N], s[N] = nl.QN, nl.qN
    for k in range(N-1, -1, -1):
        R_bar = R[k] + B[k].T@P[k+1]@B[k]
        S_bar = S[k] + B[k].T@P[k+1]@A[k]
        y = P[k+1]@c[k] + s[k+1]
        R_bar_inv = np.linalg.inv(R_bar)
        K[k] = -R_bar_inv@S_bar
        e[k] = -R_bar_inv@(B[k].T@y + r[k])
        s[k] = S_bar.T@e[k] + A[k].T@y + q[k]
        P[k] = Q[k] + A[k].T@P[k+1]@A[k] + S_bar.T@K[k]
    return problem.NewtonLagrangeFactors(K, s, P, e)


def symmetric(P):
    return 0.5 * (P.T + P)


def lqr_solve_step(
    prob: problem.Problem,
    nl: problem.NewtonLagrangeQP,
    fac: problem.NewtonLagrangeFactors
) -> problem.NewtonLagrangeUpdate:
    Δx, Δu, p = np.zeros([prob.N+1, prob.ns, 1]), np.zeros([prob.N+1, prob.nu, 1]), np.zeros_like(fac.s)
    for k in range(prob.N):
        Δu[k] = fac.K[k]@Δx[k] + fac.e[k]
        Δx[k+1] = nl.Ak[k]@Δx[k] + nl.Bk[k]@Δu[k] + nl.ck[k]
        p[k+1] = fac.P[k+1]@Δx[k+1] + fac.s[k+1]
    return problem.NewtonLagrangeUpdate(np.squeeze(Δx), np.squeeze(Δu), np.squeeze(p))


def armijo_condition(merit: problem.FullCostFunction, x_plus, u_plus, x, u, dx, du, c, σ, α):
    φ, g, dJdx, dJdu = merit.phi, merit.h, merit.dJdx, merit.dJdu
    return φ(c, x_plus, u_plus) <= φ(c, x, u) + σ * α * (dJdx(x, u) @ dx + dJdu(x, u)@du - c * g(x, u))


def armijo_linesearch(zk: problem.NLIterate, update: problem.NewtonLagrangeUpdate, merit: problem.FullCostFunction, *, σ=1e-4) -> problem.NLIterate:
    # Begin TODO----------------------------------------------------------
    raise NotImplementedError(
        "Line search has not been implemented yet! This is the subject of assignment 6.4. In the meantime, pass ``False`` for this option in the Newton-Lagrange solver.")
    # End TODO -----------------------------------------------------------
    return problem.NLIterate(xplus, uplus, pplus)


def update_iterate(zk: problem.NLIterate, update: problem.NewtonLagrangeUpdate, *, linesearch: bool, merit_function: problem.FullCostFunction = None) -> problem.NLIterate:
    """Take the current iterate zk and the Newton-Lagrange update and return a new iterate. 

    If linesearch is True, then also perform a linesearch procedure. 

    Args:
        zk (problem.NLIterate): Current iterate 
        update (problem.NewtonLagrangeUpdate): Newton-Lagrange step 
        linesearch (bool): Perform line search or not? 
        merit_function (problem.FullCostFunction, optional): The merit function used for linesearch. Defaults to None.

    Raises:
        ValueError: If no merit function was passed, but linesearch was requested. 

    Returns:
        problem.NLIterate: Next Newton-Lagrange iterate.
    """

    if linesearch:
        if merit_function is None:
            raise ValueError(
                "No merit function was passed but line search was requested")
        return armijo_linesearch(zk, update, merit_function)
    # Begin TODO----------------------------------------------------------



    # End TODO -----------------------------------------------------------
    # Hint: The initial state must remain fixed. Only update from time index 1!
    xnext = np.array([zk.x[0]] + [zk.x[k] + update.dx[k] for k in range(1, zk.x.shape[0])])
    unext = np.array([zk.u[k] + update.du[k] for k in range(0, zk.u.shape[0])])
    pnext = update.p[1:, :]
    return problem.NLIterate(
        x=xnext,
        u=unext,
        p=pnext
    )


def is_posdef(M):
    return np.min(np.linalg.eigvalsh(M)) > 0


def regularize(qp: problem.NewtonLagrangeQP):
    """Regularize the problem.

    If the given QP (obtained as a linearization of the problem) is nonconvex, 
    add an increasing multiple of the identity to the Hessian 
    until it is positive definite. 

    Side effects: the passed qp is modified by the regularization!

    Args:
        qp (problem.NewtonLagrangeQP): Linearization of the optimal control problem
    """
    # Begin TODO----------------------------------------------------------
    raise NotImplementedError("Implement regularization!")
    # End TODO -----------------------------------------------------------


def newton_lagrange(p: problem.Problem,
                    initial_guess=problem.NLIterate, cfg: problem.NewtonLagrangeCfg = None, *,
                    log_callback: Callable = lambda *args, **kwargs: ...
                    ) -> problem.NewtonLagrangeStats:
    """Newton Lagrange method for nonlinear OCPs
    Args:
        p (problem.Problem): The problem description 
        initial_guess (NLIterate, optional): Initial guess. Defaults to problem.NewtonLagrangeIterate.
        cfg (problem.NewtonLagrangeCfg, optional): Settings. Defaults to None.
        log_callback (Callable): A function that takes the iteration count and the current iterate. Useful for logging purposes. 

    Returns:
        Solver stats  
    """
    stats = problem.NewtonLagrangeStats(0, initial_guess)
    # Set the default config if None was passed
    if cfg is None:
        cfg = problem.NewtonLagrangeCfg()

    # Get the merit function ingredients in case line search was requested
    if cfg.linesearch:
        full_cost = problem.build_cost_and_constraint(p)
    else:
        full_cost = None  # We don't need it in this case

    QP_sym = problem.construct_newton_lagrange_qp(p)
    zk = initial_guess

    for it in range(cfg.max_iter):
        qp_it = QP_sym(zk)

        if cfg.regularize:
            regularize(qp_it)

        factor = lqr_factor_step(p.N, qp_it)

        update = lqr_solve_step(p, qp_it, factor)

        zk = update_iterate(
            zk, update, linesearch=cfg.linesearch, merit_function=full_cost)

        stats.n_its = it
        stats.solution = zk
        # Call the logger.
        log_callback(stats)

        # Sloppy heuristics as termination criteria.
        # In a real application, it's better to check the violation of the KKT conditions.
        # e.g., terminate based on the norm of the gradients of the Lagrangian.
        if np.linalg.norm(update.du.squeeze(), ord=np.inf)/np.linalg.norm(zk.u) < 1e-4:
            stats.exit_message = "Converged"
            stats.success = True
            return stats

        elif np.any(np.linalg.norm(update.du) > 1e4):
            stats.exit_message = "Diverged"
            return stats

    stats.exit_message = "Maximum number of iterations exceeded"
    return stats


def exercise1():
    print("Assignment 6.1.")
    p = problem.Problem()
    qp = problem.construct_newton_lagrange_qp(p)


def fw_euler(f, Ts):
    return lambda x, u, t: x + Ts*f(x, u)


def test_linear_system():

    p = problem.ToyProblem(cont_dynamics=problem.LinearSystem(), N=100)
    u0 = np.zeros((p.N, p.nu))
    x0 = np.zeros((p.N+1, p.ns))
    x0[0] = p.x0
    initial_guess = problem.NLIterate(x0, u0, np.zeros_like(x0))

    logger = problem.Logger(p, initial_guess)
    result = newton_lagrange(p, initial_guess, log_callback=logger)
    assert result.success, "Newton Lagrange did not converge on a linear system! Something is wrong!"
    assert result.n_its < 2, "Newton Lagrange took more than 2 iterations!"


def exercise2():
    print("Assignment 6.2.")
    from rcracers.simulator.core import simulate

    # Build the problem
    p = problem.ToyProblem()

    # Select initial guess by running an open-loop simulation
    # Begin TODO----------------------------------------------------------
    u0 = np.zeros((p.N, p.nu))
    x0 = np.zeros((p.N+1, p.ns))
    x0[0] = p.x0
    initial_guess = problem.NLIterate(x0, u0, np.zeros_like(x0))
    # End TODO -----------------------------------------------------------
    logger = problem.Logger(p, initial_guess)
    stats = newton_lagrange(p, initial_guess, log_callback=logger)
    from given.homework import animate
    animate.animate_iterates(
        logger.iterates, os.path.join(WORKING_DIR, "Assignment6-2"))


def exercise34(linesearch: bool):
    print("Assignment 6.3 and 6.4.")
    from rcracers.simulator.core import simulate
    f = problem.ToyDynamics(False)

    # Build the problem
    p = problem.ToyProblem()

    # Select initial guess by running an open-loop simulation
    # Begin TODO----------------------------------------------------------
    u0 = np.zeros((p.N, p.nu))
    x0 = np.zeros((p.N+1, p.ns))
    x0[0] = p.x0

    def policy(u):
        return np.zeros_like(u)
    
    x_sim = simulate(p.x0, f, p.N, policy=policy)

    initial_guess = problem.NLIterate(x0, u0, np.zeros_like(x0))
    # End TODO -----------------------------------------------------------

    logger = problem.Logger(p, initial_guess)
    cfg = problem.NewtonLagrangeCfg(linesearch=linesearch, max_iter=100)
    final_iterate = newton_lagrange(
        p, initial_guess, log_callback=logger, cfg=cfg)
    print(final_iterate.exit_message)
    from given.homework import animate
    animate.animate_iterates(
        logger.iterates, os.path.join(WORKING_DIR, "Assignment6-4"))


def exercise56(regularize=False):
    # Build the problem
    p = problem.ParkingProblem()

    # Select initial guess by running an open-loop simulation
    # Begin TODO----------------------------------------------------------
    u0 = np.zeros((p.N, p.nu))
    x0 = np.zeros((p.N+1, p.ns))
    x0[0] = p.x0
    initial_guess = problem.NLIterate(x0, u0, np.zeros_like(x0))
    # End TODO -----------------------------------------------------------

    logger = problem.Logger(p, initial_guess)
    cfg = problem.NewtonLagrangeCfg(
        linesearch=True, max_iter=50, regularize=regularize)
    final_iterate = newton_lagrange(
        p, initial_guess, log_callback=logger, cfg=cfg)
    print(final_iterate.exit_message)

    from given.homework import animate
    animate.animate_iterates(logger.iterates, os.path.join(
        WORKING_DIR, f"Assignment6-4-reg{regularize}"))
    animate.animate_positions(logger.iterates, os.path.join(
        WORKING_DIR, f"parking_regularize-{regularize}"))


if __name__ == "__main__":
    # test_linear_system()
    exercise2()
    # exercise34(False)
    # exercise34(True)
    # exercise56(regularize=False)
    # exercise56(regularize=True)
