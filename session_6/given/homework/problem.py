from dataclasses import dataclass
from rcracers.simulator.dynamics import KinematicBicycle
import numpy as np 
import casadi as cs 
from typing import Callable, Union

Vector = Union[cs.SX, np.ndarray] 
Linearization = Union[cs.Function, np.ndarray]

@dataclass
class ToyDynamics:
    """Dynamics of the toy problem.""" 
    symbolic: bool = True 

    def __call__(self, x, u):
        dx1 = 10 * (x[1] - x[0])
        dx2 = x[0] * (u - x[2]) - x[1]
        dx3 = x[0]*x[1] - 3 * x[2]

        cat = cs.vertcat if self.symbolic else (lambda *x: np.array([float(xi) for xi in x]))
        return cat(dx1, dx2, dx3)


@dataclass
class LinearSystem:  
    """Dynamics of a Linear test system"""
    A: np.ndarray = np.array([[1, 0.5, 0.01], 
                              [1, 0.,  0],
                              [0., 1., 0.]]
                            )
    B: np.ndarray = np.array([[1], [0.5], [0.1]])

    def __call__(self, x, u):
        return self.A@x + self.B@u


class Problem:
    """Problem description for a nonlinear OCP"""
    cont_dynamics: Callable  # Continuous-time dynamics 
    Q: np.ndarray  # Weights for the state in the stage cost 
    QT: np.ndarray  # Weights for the terminal cost 
    R: np.ndarray   # Weights for the inputs in the stage cost 
    Ts: int  # Sampling time 
    N: int   # Prediction horizon 
    x0: np.ndarray  # Initial state 

    def f(self, x,u):
        """Discrete-time dynamics."""
        return x + self.Ts*self.cont_dynamics(x,u) 

    def l(self, x, u):
        """Stage cost."""
        return 0.5* (x.T@self.Q@x + u.T@self.R@u)

    def vf(self, x):
        """Terminal cost"""
        return 0.5* (x.T@self.QT@x)

    def h(self, p, x, u):
        """Hamiltonian."""
        return self.l(x,u) + p.T@self.f(x,u)

    @property 
    def ns(self) -> int:
        """State dimension"""
        return self.Q.shape[0]

    @property 
    def nu(self) -> int:
        """Input dimension"""
        return self.R.shape[0]


#-----------------------------------------------------------
# TEST PROBLEMS 
#-----------------------------------------------------------

@dataclass
class ParkingProblem(Problem):
    """Description of the parking problem
    """
    cont_dynamics: KinematicBicycle = KinematicBicycle(symbolic=True)
    #Default tuning, not the best choices!
    Q: np.ndarray = cs.diagcat(1, 10, 0.1, 0.01)
    QT: np.ndarray = 10 * cs.diagcat(1, 3, 0.1, 0.01)
    R: np.ndarray = cs.diagcat(1., 1e-2)
    Ts: int = 0.05
    N: int = 50 # Horizon
    x0: np.ndarray = np.array([1., -1, 0, 0]) # Initial state 


@dataclass
class ToyProblem(Problem):
    """Toy problem"""
    cont_dynamics: ToyDynamics = ToyDynamics(symbolic=True)
    Q: np.ndarray = np.eye(3)
    QT: np.ndarray = 5 * np.eye(3)
    R: np.ndarray = 0.01 * np.eye(1)
    Ts: int = 0.05
    N: int = 20 # Horizon 
    x0: np.ndarray = np.array([-10, 15., 10.]) # Initial state 



#-----------------------------------------------------------
# Data containers 
#-----------------------------------------------------------
# Below are a few data containers that are used throughout 
# the template file. The fieldnames correspond closely to the 
# notation used in the slides, so it should be relatively straightforward to 
# interpret what each of the following structures represent. 
# 

@dataclass
class NLIterate:
    """Iterate of the Newton-Lagrange method"""
    x: np.ndarray   # State sequence  (N + 1) x ns
    u: np.ndarray   # Input sequence   N x nu 
    p: np.ndarray   # Lagrange multipliers / costates  N x ns 

    @property
    def z(self):
        """Concatenate the variables into a single vector."""
        return np.concatenate((self.x.ravel(), self.u.ravel()))


@dataclass 
class NewtonLagrangeQP:
    """QP representing the linearization of the problem at a given point.
    
    The ``Linearization``-type can either be a Casadi symbolic or a 
    numerical array. Thus, this class can both represent the 
    linearization as a function, as well as the numerical evaluation
    of the linearization in a given point.  
    """
    Qk: Linearization
    Rk: Linearization
    Sk: Linearization
    qN: Linearization
    qk: Linearization
    rk: Linearization
    QN: Linearization
    Ak: Linearization
    Bk: Linearization
    ck: Linearization
    f: Linearization

    def __call__(self, zk: NLIterate) -> "NewtonLagrangeQP":
        """Evaluate self in a given iterate ``zk``.

        Args:
            zk (NLIterate): The iterate in which to evaluate the linearization. 

        Returns:
            NewtonLagrangeQP: Return a linearization in which all the elements are evaluated and stored as numerical values. 
        """
        x = zk.x 
        u = zk.u 
        p = zk.p
        N = u.shape[0]
        Qk = np.empty((N,) + self.Qk.size_out(0))
        Rk = np.empty((N,) + self.Rk.size_out(0))
        Sk = np.empty((N,) + self.Sk.size_out(0))
        qk = np.empty((N,) + self.qk.size_out(0))
        rk = np.empty((N,) + self.rk.size_out(0))
        Ak = np.empty((N,) + self.Ak.size_out(0))
        Bk = np.empty((N,) + self.Bk.size_out(0))
        ck = np.empty((N,) + self.ck.size_out(0))
        f  = np.nan 

        for k in range(len(u)):
            Ak[k] = self.Ak(x[k], u[k])
            Bk[k] = self.Bk(x[k], u[k])
            ck[k] = self.ck(x[k], u[k], x[k+1])
            Rk[k] = self.Rk(x[k], u[k], p[k])
            Qk[k] = self.Qk(x[k], u[k], p[k])
            Sk[k] = self.Sk(x[k], u[k], p[k])
            qk[k] = self.qk(x[k], u[k])
            rk[k] = self.rk(x[k], u[k])
        QN = self.QN(x[-1])
        qN = self.qN(x[-1])
        return NewtonLagrangeQP(Qk, Rk, Sk, qN, qk, rk, QN, Ak, Bk, ck, f)


@dataclass
class NewtonLagrangeFactors:
    """Output of the LQR factorization step"""
    K: np.ndarray
    s: np.ndarray
    P: np.ndarray
    e: np.ndarray


@dataclass
class NewtonLagrangeUpdate:
    """Update of the Newton-Lagrange method"""
    dx: np.ndarray  # Update on x (Δx in the slides)
    du: np.ndarray  # Update on u (Δu in the slides)
    p: np.ndarray   # Update on the costates 


@dataclass
class NewtonLagrangeCfg:
    """Settings for the Newton Lagrange method. 
    """
    max_iter: int = 50 
    linesearch: bool = False
    regularize: bool = False


@dataclass 
class FullCostFunction:
    """Summary of the full cost, including its gradient w.r.t. x and u and the merit function.
    """
    JN: cs.Function
    dJdx: cs.Function 
    dJdu: cs.Function 
    phi: cs.Function
    h: cs.Function 


@dataclass
class NewtonLagrangeStats:
    """Statistics on the Newton-Lagrange method. This is the output of the method."""
    n_its: int = 0 
    solution: NLIterate = None 
    exit_message: str = None
    success: bool = False

#-----------------------------------------------------------
# Utility functions
#-----------------------------------------------------------
# Below are a few utility functions that will prove useful 
# for the homework assignment. 

def to_vec(x: np.ndarray) -> np.ndarray:
    """Flatten an array into a vector, by concatenating the rows (C-style)

    Args:
        x (np.ndarray): Matrix 

    Returns:
        np.ndarray: vectorized version
    """
    return np.reshape(x, (-1,1), order="C")



def construct_newton_lagrange_qp(prob: Problem) -> NewtonLagrangeQP:
    """Construct the ingredients for the Newton Lagrange method. 

    Given a problem, construct the linearizations defined in slide 10-16. 
    These are stored in the ``NewtonLagrange`` dataclass (see above).

    Args:
        prob (Problem): Problem specs 

    Returns:
        NewtonLagrange: Newton-Lagrange QP 
    """
    # Symbolic variables for x and u
    xk = cs.SX.sym("xk", prob.ns)    # States
    xk1 = cs.SX.sym("xk+", prob.ns)  # x_{k+1}: successor states 
    uk = cs.SX.sym("xk", prob.nu)    # Controls
    pk = cs.SX.sym("pk", prob.ns)    # Lagrange multipliers (costates)

    H  = prob.h(pk, xk, uk)          # Hamiltonian 

    HHu, gradHu = cs.hessian(H,uk)   # The Hessian returns the gradient as a by-product.
    Qk = cs.Function('Qk',[xk, uk, pk], [cs.hessian(H,xk)[0]])
    Rk = cs.Function('Rk',[xk, uk, pk], [HHu])
    Sk = cs.Function('Sk',[xk, uk, pk], [cs.jacobian(gradHu,xk)])
    
    Vf = prob.vf(xk)
    l  = prob.l(xk, uk)

    HVf, gradVf = cs.hessian(Vf, xk)
    qN = cs.Function('qN',[xk], [gradVf])
    qk = cs.Function('qk',[xk, uk], [cs.gradient(l,xk)])
    rk = cs.Function('rk',[xk, uk], [cs.gradient(l,uk)])
    QN = cs.Function('QN',[xk], [HVf])

    f  = prob.f(xk, uk)
    Ak = cs.Function('Ak',[xk, uk], [cs.jacobian(f,xk)])
    Bk = cs.Function('Bk',[xk, uk], [cs.jacobian(f,uk)])
    ck = cs.Function('ck',[xk, uk, xk1], [f - xk1])
    f  = cs.Function('f',[xk, uk], [f])
    return NewtonLagrangeQP(
        Qk = Qk, Rk=Rk, Sk=Sk, 
        qN = qN, qk=qk, rk=rk, 
        QN = QN, Ak=Ak,
        Bk = Bk, ck=ck, 
        f = f 
    )


def build_cost_and_constraint(prob: Problem) -> FullCostFunction:
    """
    Build the N-step cost function and dynamics constraint function.
    """ 
    x = [cs.SX.sym(f"x{k}", prob.ns) for k in range(prob.N + 1)]
    u = [cs.SX.sym(f"u{k}", prob.nu) for k in range(prob.N)]

    JN = sum((prob.l(xk,uk) for xk, uk in zip(x, u))) + prob.vf(x[-1])
    h  = sum((cs.norm_1(prob.f(xk,uk) - xk1) for xk, uk, xk1 in zip(x[:-1], u, x[1:])))

    x = cs.vertcat(*x)
    u = cs.vertcat(*u)
    c = cs.SX.sym("c")
    phi = cs.Function('phi',[c, x, u], [JN + c*h])
    jac_JN_x = cs.Function('jac_JN_x',[x, u],[cs.jacobian(JN,x)])
    jac_JN_u = cs.Function('jac_JN_u',[x, u],[cs.jacobian(JN,u)])
    h = cs.Function('g',[x,u],[h])
    JN = cs.Function('JN',[x,u],[JN])
    return FullCostFunction(JN, dJdx=jac_JN_x, dJdu=jac_JN_u, phi=phi, h=h)


class Logger:
    """Basic logger that can be used for debugging and monitoring of the internals of the Newton-Lagrange method."""

    def __init__(self, problem: Problem, x0: NLIterate):
        self.iterates = [x0]
        self.full_cost = build_cost_and_constraint(problem)

    def __call__(self, stats: NewtonLagrangeStats):

        # Get the solution 
        z = stats.solution
        # Evaluate the cost 
        cost = float(self.full_cost.JN(to_vec(z.x), to_vec(z.u)))
        # Evaluate the constraint violations
        dynamics_violation = float(self.full_cost.h(to_vec(z.x), to_vec(z.u)))

        # Print progress to terminal 
        print(f"it. {stats.n_its:4d} | JN = {cost:4.2e} | ||h||2 = {dynamics_violation:4.2e}")
        # Store current iterate in the log 
        self.iterates.append(z)


