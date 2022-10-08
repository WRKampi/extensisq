import numpy as np
from extensisq.common import RungeKutta


class HE2(RungeKutta):
    """Heun's explicit Runge-Kutta method of order 2, with the explicit Euler
    method as error estimate of order 1 and a free interpolant of order 3.

    This is the simplest embedded Runge-Kutta method [1]_.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray
        ``y``: It can either have shape (n,); then ``fun`` must return
        array_like with shape (n,). Alternatively it can have shape (n, k);
        then ``fun`` must return an array_like with shape (n, k), i.e., each
        column corresponds to a single column in ``y``. The choice between the
        two options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is
        not bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. A vectorized
        implementation offers no advantages for this solver. Default is False.
    nfev_stiff_detect : int, optional
        Number of function evaluations for stiffness detection. This number has
        multiple purposes. If it is set to 0, then stiffness detection is
        disabled. For other (positive) values it is used to represent a
        'considerable' number of function evaluations (nfev). A stiffness test
        is done if many steps fail and each time nfev exceeds integer multiples
        of `nfev_stiff_detect`. For the assessment itself, the problem is
        assessed as non-stiff if the predicted nfev to complete the integration
        is lower than `nfev_stiff_detect`. The default value is 5000.
    sc_params : tuple of size 4, "standard", "G", "H" or "W", optional
        Parameters for the stepsize controller (k*b1, k*b2, a2, g). The step
        size controller is, with k the exponent of the standard controller,
        _n for new and _o for old:
            h_n = h * g**(k*b1 + k*b2) * (h/h_o)**-a2
                * (err/tol)**-b1 * (err_o/tol_o)**-b2
        Predefined parameters are:
            Gustafsson "G" (0.7, -0.4, 0, 0.9),  Watts "W" (2, -1, -1, 0.8),
            Soederlind "S" (0.6, -0.2, 0, 0.9),  and "standard" (1, 0, 0, 0.9).
        The default for this method is "standard".

    References
    ----------
    .. [1] Wikipedia:
           https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Heun%E2%80%93Euler
    """

    # effective number of stages
    n_stages = 2

    # order of the main method
    order = 2

    # order of the secondary embedded method
    error_estimator_order = 1

    # runge kutta coefficient matrix
    A = np.array([
        [0., 0.],
        [1., 0.]])

    # output coefficients (weights)
    B = np.array([0.5, 0.5])

    # time fraction coefficients (nodes)
    C = np.array([0., 1.])

    # error coefficients (weights Bh - B)
    E = np.array([0.5, -0.5, 0.])

    # Parameters for stiffness detection, optional
    stbrad = 2.               # radius of the arc
    tanang = 2.               # tan(valid angle < pi/2)
