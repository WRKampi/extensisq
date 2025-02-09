import numpy as np
from extensisq.common import RungeKuttaNystrom


class MR6NN(RungeKuttaNystrom):
    """Explicit Runge-Kutta Nystrom method by El-Mikkawy and Rahmo [1]_ of
    order 6. The embedded method has order 4. This method is applicable to
    second order initial value problems only. Moreover, these problems must be
    independent of the first derivative (velocity). Undamped mechanics is an
    example of such a problem.

    The second order problem should be recast in first order form as
    u = [x, v], du = [v, a], with x, v, a variables like, position,
    velocity, acceleration. The derivative function du = f(t, u) should
    calculate only a and pass through v. (The order in u and du matters.) This
    is the same form as for general RKN methods in extensisq. So, although the
    the input of f() contains v, it must not be used in it.

    This method includes a free C2-continuous sixth order interpolant.

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
        two options is determined by `vectorized` argument (see below). For
        this second order problem, y should contain all solution components
        first followed by an equal number of first derivative components of the
        solution. Likewise, the returned array should contain the first
        derivatives first followed by the second derivatives. (The first
        derivatives are identical those in the input and the second derivatives
        are calculated.)
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
    .. [1] M. El-Mikkawy, E.D. Rahmo, "A new optimized non-FSAL embedded
           Runge-Kutta-Nystrom algorithm of orders 6 and 4 in six stages",
           Applied Mathematics and Computation, Vol. 145, Issue 1, 2003,
           pp. 33-43, https://doi.org/10.1016/S0096-3003(02)00436-8
    .. [2] G.Söderlind, "Automatic Control and Adaptive Time-Stepping",
           Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310.
           https://doi.org/10.1023/A:1021160023092
    """
    n_stages = 6
    order = 6
    order_secondary = 4
    sc_params = "G"

    C = np.array([0, 1/77, 1/3, 2/3, 13/15, 1])

    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/11858, 0, 0, 0, 0, 0],
        [-7189/17118, 4070/8559, 0, 0, 0, 0],
        [4007/2403, -589655/355644, 25217/118548, 0, 0, 0],
        [-4477057/843750, 13331783894/2357015625, -281996/5203125,
         563992/7078125, 0, 0],
        [17265/2002, -1886451746/212088107, 22401/31339, 2964/127897,
         178125/5428423, 0]])
    # no Ap

    B = np.array([-341/780, 386683451/661053840, 2853/11840, 267/3020,
                  9375/410176, 0])

    Bp = np.array([-341/780, 29774625727/50240091840, 8559/23680, 801/3020,
                   140625/820352, 847/18240])

    E = np.array([-95/39, 89332243/33052692, 317/3552, 623/5436, 54125/1845792,
                  0, 0])
    E[:-1] -= B

    Ep = np.array([-95/39, 362030669/132210768, 317/2368, 623/1812,
                   270625/1230528, 0, 0])
    Ep[:-1] -= Bp

    P = np.array([
        [1/2, -445/39, 2095/78, -1231/52, 1421/195],
        [0, 56490936887/5024009184, -280556420221/10048018368,
            419129707843/16746697280, -195064224509/25120045920],
        [0, 951/2368, 2853/4736, -31383/23680, 6657/11840],
        [0, -89/151, 267/151, -4539/3020, 623/1510],
        [0, 228125/410176, -1790625/820352, 2184375/820352, -415625/410176],
        [0, 847/1824, -5929/3648, 11011/6080, -5929/9120],
        [0, -2/3, 5/2, -3, 7/6]])
    Pp = P * np.arange(2, 7)    # derivative of P
