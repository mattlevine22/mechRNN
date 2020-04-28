import numpy as np
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
from pydoc import locate
import pdb

def get_custom_solver(solver_name, library_path='integratorlibrary'):
    classtr = '{library_path}.{solver_name}'.format(library_path=library_path, solver_name=solver_name)
    custom_solver = locate(classtr)
    if custom_solver is not None:
        return custom_solver
    else:
        return solver_name

def euler_step(fun, t, y, h):
    """Perform a single Euler step.
    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Current value of the derivative, i.e. ``fun(x, y)``.
    h : float
        Step to use.
    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h
    References
    ----------
    """

    y_new = y + h * fun(t, y)

    return y_new


class Euler(OdeSolver):
    """Class for Euler Integration Method."""
    def __init__(self, fun, t0, y0, t_bound, h, vectorized=False, **extraneous):
        super(Euler, self).__init__(fun, t0, y0, t_bound, vectorized, support_complex=True)
        self.h = h
        self.y_old = None

    def _step_impl(self):
        t = self.t
        y = self.y
        h = self.h

        y_new = euler_step(self.fun, t, y, h)
        t_new = t + h

        self.y_old = y

        self.t = t_new
        self.y = y_new

        return True, None

    def _dense_output_impl(self):
        # Q = self.K.T.dot(self.P)
        return EulerDenseOutput(self.t_old, self.t, self.y_old, self.fun)

class EulerDenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, fun):
        super(EulerDenseOutput, self).__init__(t_old, t)
        self.h = t - t_old
        self.y_old = y_old
        self.fun = fun

    def _call_impl(self, t):
        y = euler_step(self.fun, self.t_old, self.y_old, self.h)
        return y[:,None]


