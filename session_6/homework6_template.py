
from typing import Callable
from given.homework import problem
import numpy as np 

import os 
WORKING_DIR = os.path.split(__file__)[0]

def lqr_factor_step(N: int, nl: problem.NewtonLagrangeQP) -> problem.NewtonLagrangeFactors:
    #Begin TODO----------------------------------------------------------
    

    #End TODO -----------------------------------------------------------
    return problem.NewtonLagrangeFactors(K, s, P, e)

def symmetric(P):
    return 0.5 * (P.T + P)

def lqr_solve_step(
    prob: problem.Problem,
    nl: problem.NewtonLagrangeQP,
    fac: problem.NewtonLagrangeFactors
) -> problem.NewtonLagrangeUpdate: 
    #Begin TODO----------------------------------------------------------
    raise NotImplementedError("Implement the LQR solve step!")
    #End TODO -----------------------------------------------------------
    return problem.NewtonLagrangeUpdate(dx, du, p)


def armijo_condition(merit: problem.FullCostFunction, x_plus, u_plus, x, u, dx, du, c, σ, α):
    φ, g, dJdx, dJdu = merit.phi, merit.h, merit.dJdx, merit.dJdu
    return φ(c, x_plus, u_plus) <= φ(c, x, u) + σ * α * (dJdx(x, u) @ dx + dJdu(x,u)@du - c * g(x,u))


def armijo_linesearch(zk: problem.NLIterate, update: problem.NewtonLagrangeUpdate, merit: problem.FullCostFunction, *, σ=1e-4) -> problem.NLIterate:
    #Begin TODO----------------------------------------------------------
    raise NotImplementedError("Line search has not been implemented yet! This is the subject of assignment 6.4. In the meantime, pass ``False`` for this option in the Newton-Lagrange solver.")
    #End TODO -----------------------------------------------------------
    return problem.NLIterate(xplus, uplus, pplus)


def update_iterate(zk: problem.NLIterate, update: problem.NewtonLagrangeUpdate, *, linesearch: bool, merit_function: problem.FullCostFunction=None) -> problem.NLIterate:
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
            raise ValueError("No merit function was passed but line search was requested")
        return armijo_linesearch(zk, update, merit_function)
    #Begin TODO----------------------------------------------------------
    raise NotImplementedError("Implement regular Newton-Lagrange update in update_iterate!")
    #End TODO -----------------------------------------------------------
    # Hint: The initial state must remain fixed. Only update from time index 1!
    xnext = ... 
    unext = ... 
    pnext = ... 
    return problem.NLIterate(
        x = xnext,
        u = unext, 
        p = pnext 
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
    #Begin TODO----------------------------------------------------------
    raise NotImplementedError("Implement regularization!")
    #End TODO -----------------------------------------------------------


def newton_lagrange(p: problem.Problem,
         initial_guess = problem.NLIterate, cfg: problem.NewtonLagrangeCfg = None, *,
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
        full_cost = None # We don't need it in this case 
    
    QP_sym = problem.construct_newton_lagrange_qp(p)
    zk = initial_guess

    for it in range(cfg.max_iter):
        qp_it = QP_sym(zk)

        if cfg.regularize:
            regularize(qp_it)

        factor = lqr_factor_step(p.N, qp_it)

        update = lqr_solve_step(p, qp_it, factor)

        zk = update_iterate(zk, update, linesearch=cfg.linesearch, merit_function=full_cost)

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
    return lambda x,u,t: x + Ts*f(x,u)

def test_linear_system():

    p = problem.ToyProblem(cont_dynamics = problem.LinearSystem(), N=100)
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
    #Begin TODO----------------------------------------------------------
    raise NotImplementedError("Construct the required initial guess!")
    initial_guess = ... 
    #End TODO -----------------------------------------------------------
    logger = problem.Logger(p, initial_guess)
    stats = newton_lagrange(p, initial_guess, log_callback=logger)
    from given.homework import animate
    animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, "Assignment6-2"))


def exercise34(linesearch:bool):
    print("Assignment 6.3 and 6.4.")
    from rcracers.simulator.core import simulate
    f = problem.ToyDynamics(False)

    # Build the problem 
    p = problem.ToyProblem()

    # Select initial guess by running an open-loop simulation
    #Begin TODO----------------------------------------------------------
    raise NotImplementedError("Select the initial guess from an open-loop simulation")
    initial_guess = ... 
    #End TODO -----------------------------------------------------------
    
    logger = problem.Logger(p, initial_guess)
    cfg = problem.NewtonLagrangeCfg(linesearch=linesearch, max_iter=100)
    final_iterate = newton_lagrange(p, initial_guess, log_callback=logger, cfg=cfg)
    print(final_iterate.exit_message)
    from given.homework import animate
    animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, "Assignment6-4"))


def exercise56(regularize=False):
    # Build the problem 
    p = problem.ParkingProblem()

    # Select initial guess by running an open-loop simulation
    #Begin TODO----------------------------------------------------------
    raise NotImplementedError("Construct the initial guess for your solver from simulation.")
    #End TODO -----------------------------------------------------------

    logger = problem.Logger(p, initial_guess)
    cfg = problem.NewtonLagrangeCfg(linesearch=True, max_iter=50, regularize=regularize)
    final_iterate = newton_lagrange(p, initial_guess, log_callback=logger, cfg=cfg)
    print(final_iterate.exit_message)

    from given.homework import animate
    animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, f"Assignment6-4-reg{regularize}"))
    animate.animate_positions(logger.iterates, os.path.join(WORKING_DIR, f"parking_regularize-{regularize}"))


if __name__ == "__main__":
    test_linear_system()
    # exercise2()
    # exercise34(False)
    # exercise34(True)
    # exercise56(regularize=False)
    # exercise56(regularize=True)