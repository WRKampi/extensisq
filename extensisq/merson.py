import numpy as np
from extensisq.common import RungeKutta


class Me4(RungeKutta):
    """Merson's explicit Runge-Kutta method [1]_ of order 4, with an embedded
    method for error estimation of order 3 (or 5 for linear time invariant
    problems) and a free interpolant of order 3 (4th order polynomial).

    This is the oldest embedded Runge-Kutta method. It has a large stability
    domain for a 4th order method, especially on the imaginary axis.

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
        Predefined parameters are [2]_:
            Gustafsson "G" (0.7, -0.4, 0, 0.9),
            Soederlind "S" (0.6, -0.2, 0, 0.9),
            and "standard" (1, 0, 0, 0.9).
        The default for this method is "G".

    References
    ----------
    .. [1] E. Hairer, G. Wanner, S.P. Norsett, "Solving Ordinary Differential
           Equations I", Springer Berlin, Heidelberg, 1993,
           https://doi.org/10.1007/978-3-540-78862-1
    .. [2] G.Söderlind, "Automatic Control and Adaptive Time-Stepping",
           Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310.
           https://doi.org/10.1023/A:1021160023092
    """

    # effective number of stages
    n_stages = 5

    # order of the main method
    order = 4

    # order of the secondary embedded method
    order_secondary = 3

    # time fraction coefficients (nodes)
    C = np.array([0, 1/3, 1/3, 1/2, 1])

    # runge kutta coefficient matrix
    A = np.array([
        [0, 0, 0, 0, 0],
        [1/3, 0, 0, 0, 0],
        [1/6, 1/6, 0, 0, 0],
        [1/8, 0, 3/8, 0, 0],
        [1/2, 0, -3/2, 2, 0]])

    # output coefficients (weights)
    B = np.array([1/6, 0, 0, 2/3, 1/6])

    # error coefficients (weights Bh - B)
    E = np.array([1/10, 0, 3/10, 2/5, 1/5, 0])        # B_hat
    E[:-1] -= B

    P = np.array([
        [1, - 3, 11/3, -3/2],
        [0, 0, 0, 0],
        [0, 27/4, -27/2,  27/4],
        [0, -4, 32/3, -6],
        [0, -13/10, 49/15, -9/5],
        [0, 31/20, -41/10, 51/20]])

    # Parameters for stiffness detection, optional
    stbrad = 3.4
    tanang = 20.

    # Parameters for stepsize control
    sc_params = "G"
