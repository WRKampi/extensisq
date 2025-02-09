import numpy as np
from extensisq.common import RungeKutta


class Ts5(RungeKutta):
    """Explicit Runge-Kutta method of order 5, with an error estimate of order
    4 and a free interpolant of order 4.

    This method mainly differs from RK45 (scipy default) by the values of its
    coefficients. These coefficients have been derived with fewer simplifying
    assumptions [1]_. This results in an increased efficiency in most cases.

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
    .. [1] Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the
           first column simplifying assumption", Computers & Mathematics with
           Applications, Vol. 62, No. 2, pp. 770 - 775, 2011.
           https://doi.org/10.1016/j.camwa.2011.06.002
    .. [2] G.Söderlind, "Automatic Control and Adaptive Time-Stepping",
           Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310.
           https://doi.org/10.1023/A:1021160023092
    """

    order = 5
    order_secondary = 4
    n_stages = 6        # effective nr
    tanang = 3.0
    stbrad = 3.5
    sc_params = "G"

    C = np.array([0, 0.161, 0.327, 0.9, 0.9800255409045097, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0.3354806554923570, 0, 0, 0, 0],
        [0, -6.359448489975075, 4.362295432869581, 0, 0, 0],
        [0, -11.74888356406283, 7.495539342889836, -0.09249506636175525,
            0, 0],
        [0, -12.92096931784711, 8.159367898576159, -0.07158497328140100,
            -0.02826905039406838, 0.0]])
    A[:, 0] = C - A.sum(axis=1)
    B = np.array([
        0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742,
        -3.290069515436081, 2.324710524099774])
    E = np.array([
        0.001780011052226, 0.000816434459657, -0.007880878010262,
        0.144711007173263, -0.582357165452555, 0.458082105929187,
        -1/66])                     # last term corrected with a minus sign
    P = np.array([
        [1, -2.763706197274826, 2.9132554618219126, -1.0530884977290216],
        [0, 0.13169999999999998, -0.2234, 0.1017],
        [0, 3.930296236894751, -5.941033872131505, 2.490627285651253],
        [0, -12.411077166933676, 30.338188630282318, -16.548102889244902],
        [0, 37.50931341651104, -88.1789048947664, 47.37952196281928],
        [0, -27.896526289197286, 65.09189467479368, -34.87065786149661],
        [0, 1.5, -4.0, 2.5]])
