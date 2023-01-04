import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult
from warnings import warn
from extensisq import BS5
from extensisq.common import calculate_scale, norm


def _test_functions(fun, t0, y0, ndim, args, B=None):
    """test the functions and embed args (and B)"""
    assert callable(fun), f"{fun.__name__} should be a function"
    n = len(y0)

    # test args
    try:
        _ = [*(args)]
    except TypeError:
        raise TypeError("`args` should be a tuple")

    # embed
    if B is not None:
        args = tuple(args) + (B, )

    def _fun(t, x, fun=fun):
        return np.asarray(fun(t, x, *args))

    # test function call
    try:
        test_value = _fun(t0, y0)
    except Exception:
        raise AssertionError(
            f"the function {fun.__name__} should have signature "
            f"{'f(t, y, *args)' if B is None else 'f(t, y, *args, B)'}")

    # test returned value
    if test_value.ndim != ndim:
        raise ValueError(f"{fun.__name__} should return a {ndim}D array")
    for s in test_value.shape:
        if s != n:
            raise ValueError(f"the array returned by {fun.__name__} should "
                             f"have shape{'(n,)' if ndim==1 else '(n, n)'}")

    # return function with embedded args (and B)
    return _fun


def sensitivity(fun, t_span, y0, jac, B, dfdB, dy0dB, method=BS5, **options):
    """Calculate the sensitivity at the end of a an ODE solution to a real
    parameter B. The internal differentiation method is used. [1]_

    The problem is:
        dy = fun(t, y, B),    y(t0) = y0(B)

    And the result is dy/dB at the endpoint (tf). The problem that is solved
    internally is twice the size of the original problem.

    If the `arg` option is used, the function signatures are f(t, y, *args, B).

    See also `extensisq.sensitivity_y0` to calculate the sentitivity to all
    intial values in one go.

    Parameters
    ----------
    fun : callable
        The function of the ODE that is solved with solve_ivp. The calling
        signature is fun(t, y, B). (Same as for calling solve_ivp)
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. (Same as for calling solve_ivp)
    y0 : array_like, shape (n,)
        Initial state. (Same as for calling solve_ivp)
    jac : callable
        function with signature jac(t, y, B) that returns the Jacobian dfun/dy
        as an n by n array. Unlike for solve_ivp, this is not optional and jac
        cannot be a matrix.
    B : float
        The nominal value of the parameter B.
    dfdB : callable
        function with signature dfdB(t, y, B) that returns dfun/dB as an array
        of size n.
    dy0dB : array_like, shape (n,)
        Derivative dy0/dB of the initial solution y0 to the parameter B.
    full_output : bool, optional
        return the OdeSolution object from the integration containing the
        solution and sensitivity values at each step of the integration.
        Default: False.
    method : solver class or {"RK45", "RK23", "DOP853"}
        The ODE solver that is used. This should be an explicit solver.
        Default: extensisq.BS5
    **options
        Options passed to solve_ivp. The option `vectorized` is ignored.

    Returns
    -------
    sens : array, shape (n,)
        The sensitivity dy/dB at the endpoint is returned as an array of the
        same size as y (size n).
    yf : array, shape (n, )
        The solution at the endpoint
    sol : OdeSolution
        The solver output containing the integrated problem and the
        sensitivities (flattend). Only returned if `full_output` is True.

    References
    ----------
    .. [1] E. Hairer, G. Wanner, S.P. Norsett, "Solving Ordinary Differential
           Equations I", Springer Berlin, Heidelberg, 1993,
           https://doi.org/10.1007/978-3-540-78862-1
    """
    y0 = np.asarray(y0)
    psi0 = np.asarray(dy0dB)
    N = y0.size

    # test inputs
    assert y0.ndim == 1, \
        "`y0` should be a 1d array"
    assert y0.size == psi0.size, \
        "`dy0dB` should have the same size as `y0`"
    assert isinstance(B, float) or isinstance(B, int), \
        f'`B` should be a float, not {type(B)}'
    t0 = t_span[0]
    if method in ("Radau", "BDF", "LSODA"):
        warn("`sensitivity` may not work with implicit methods")
    if options.pop("vectorized", False):
        warn("Vectorization is not supported and is switched off")

    args = options.pop("args", [])
    fun = _test_functions(fun, t0, y0, 1, args, B=B)
    dfdB = _test_functions(dfdB, t0, y0, 1, args, B=B)
    jac = _test_functions(jac, t0, y0, 2, args, B=B)

    # function to integrate
    def total_fun(x, total_y, B, fun=fun, dfdy=jac, dfdB=dfdB, N=N):
        y = total_y[:N]
        psi = total_y[N:]
        dy = fun(x, y)
        dpsi = dfdy(x, y) @ psi + dfdB(x, y)
        return np.concatenate([dy, dpsi])

    total_y0 = np.concatenate([y0, psi0])
    sol = solve_ivp(total_fun, t_span, total_y0, args=(B,), method=method,
                    **options)
    assert sol.success, "IVP solver not converged"

    # output sensitivity at end of integration
    yf = sol.y[:N, -1]
    sens = sol.y[N:, -1]
    return sens, yf, sol


def sensitivity_y0(fun, t_span, y0, jac, method=BS5, **options):
    """Calculate the sensitivity at the end of a an ODE solution to all initial
    values. The internal differentiation method is used. [1]_

    If the size of the original problem is n, the size of the problem that is
    solved to find the sensitivity to all initial values is n*(n+1).

    See also `extensisq.sensitivity` to calculate the sensitivity to a general
    (scalar) parameter.

    Parameters
    ----------
    fun : callable
        The function of the ODE that is solved with solve_ivp. The calling
        signature is fun(t, y). (Same as for calling solve_ivp)
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. (Same as for calling solve_ivp)
    y0 : array_like, shape (n, )
        Initial state. (Same as for calling solve_ivp)
    jac : callable
        function with signature jac(t, y) that returns the Jacobian dfun/dy
        as an n by n array. Unlike for solve_ivp, this is not optional and jac
        cannot be a matrix.
    method : solver class or {"RK45", "RK23", "DOP853"}
        The ODE solver that is used. This should be an explicit solver.
        Default: extensisq.BS5
    **options
        Options passed to solve_ivp. Options passed to solve_ivp. The option
        `vectorized` is ignored.

    Returns
    -------
    sens : array, shape (n, n)
        The sensitivity dy/dy0 at the endpoint is returned as an array of size
        n by n. The elements (i,j) are dy[i]/dy0[j].
    yf : array, shape (n, )
        The solution at the endpoint
    sol : OdeSolution
        The solver output containing the integrated problem and the
        sensitivities (flattend). Only returned if `full_output` is True.

    References
    ----------
    .. [1] E. Hairer, G. Wanner, S.P. Norsett, "Solving Ordinary Differential
           Equations I", Springer Berlin, Heidelberg, 1993,
           https://doi.org/10.1007/978-3-540-78862-1
    """
    y0 = np.asarray(y0)
    N = y0.size
    psi0 = np.eye(N)

    # test inputs
    assert y0.ndim == 1, "`y0` should be a 1d array"
    t0 = t_span[0]
    if method in ("Radau", "BDF", "LSODA"):
        warn("`sensitivity_y0` may not work with implicit methods")
    if options.pop("vectorized", False):
        warn("Vectorization is not supported and is switched off")

    args = options.pop("args", [])
    fun = _test_functions(fun, t0, y0, 1, args)
    jac = _test_functions(jac, t0, y0, 2, args)

    # function to integrate
    def total_fun(x, total_y, fun=fun, dfdy=jac, N=N):
        y = total_y[:N]
        psi = total_y[N:].reshape(N, N)
        dy = fun(x, y)
        dpsi = dfdy(x, y) @ psi
        return np.concatenate([dy, dpsi.reshape(-1)])

    total_y0 = np.concatenate([y0, psi0.reshape(-1)])
    sol = solve_ivp(total_fun, t_span, total_y0, method=method, **options)
    assert sol.success, f"IVP solver not converged; {sol.message}"

    # output sollution and sensitivity at end of integration
    sens = sol.y[N:, -1].reshape(N, N)
    yf = sol.y[:N, -1]
    return sens, yf, sol


def find_periodic_solution(fun, t_span, y0, jac, atol=1e-6, rtol=1e-3,
                           max_iterations=8, **options):
    """Find a periodic solution of the problem, by tuning the initial
    conditions.

    A periodically forced (non-linear) solution may have a periodic solution
    for some specific initial values `y0` that are not known yet. This function
    tries to find this `y0` by running Newton Raphson iteration. The Jacobian
    for it is calculated using function `sensitivity_y0`.

    The period is defined by `t_span` and a resonably good initial estimate of
    `y0` is needed.

    Integrating again with the found y0 can result in a higher residual.
    Integration without using sensitivity_y0 results in different solver step
    sizes, which explains the difference. The tolerances can be set stricter to
    mitigate this.

    Parameters
    ----------
    fun : callable
        The function of the ODE that is solved with solve_ivp. The calling
        signature is fun(t, y). (Same as for calling solve_ivp)
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. The period of the integration tf - t0
        should be exactly one period.
    y0 : array_like, shape (n, )
        Initial state. This should be an approximation of the initial state of
        the periodic solution. The method may fail to converge if this estimate
        is poor.
    jac : callable
        Function with signature jac(t, y) that returns the Jacobian dfun/dy
        as an n by n array. Unlike for solve_ivp, this is not optional and jac
        cannot be a matrix.
    method : solver class or {"RK45", "RK23", "DOP853"}
        The ODE solver that is used. This should be an explicit solver.
        Default: extensisq.BS5
    atol : float, optional
        Absolute tolerance passed to `solve_ivp` and used for the stopping
        criteria of the Newton Raphson iteration. Default value: 1e-6
    rtol : float, optional
        Relative tolerance passed to `solve_ivp` and used for the stopping
        criteria of the Newton Raphson iteration. Default value: 1e-3
    max_iterations : int, optional
        Maximum number of correction to the initial value, by default 8.
    **options
        Options passed to solve_ivp. Options passed to solve_ivp. The option
        `vectorized` is ignored.

    Returns
    -------
    Object with results
        This object contains the OdeSolution entries of the last solve_ivp call
        (from `sensitivity_y0`, including the sensitivity solution), and some
        information from the Newton Raphson iteration:
            opt_success : bool
                True if the Newton Raphson iteration has converged. (Note: Do
                not confuse this with `success` from solve_ivp, which is also
                included in this object.
            opt_y0 : array_like, shape (n, )
                The found initial value for the periodic solution. The value
                from the last iteration is returned no matter the value of
                opt_succes.
            opt_residual : array_like, shape (n, )
                The remaining residual yf - y0.
            opt_nit : int
                The number of Newton Raphson iterations. The number of calls to
                `sensitivity_y0` is opt_nit + 1.
    """
    options["atol"] = atol
    options["rtol"] = rtol
    y0 = np.asarray(y0)
    N = y0.size
    In = np.eye(N)

    # Newton Raphson iteration
    correction_norm_old = None
    rate = np.inf
    for it in range(max_iterations+1):
        # calculate solution and sensitivity
        S0, yf, sol = sensitivity_y0(fun, t_span, y0, jac, **options)
        residual = yf - y0
        jacobian = S0 - In
        correction = np.linalg.solve(jacobian, residual)
        scale = calculate_scale(atol, rtol, y0, yf)

        # assess solution
        correction_norm = norm(correction/scale)
        if correction_norm_old is not None:
            rate = correction_norm/correction_norm_old
        converged_solution = correction_norm < (1. - rate)

        # assess residual
        residual_norm = norm(residual/scale)
        converged_residual = residual_norm < 1.

        if converged_residual and converged_solution:
            # solution converged
            success = True
            break

        # update
        y0 -= correction
        correction_norm_old = correction_norm

    else:
        # solution not converged
        success = False
        # undo last update
        y0 += correction

    return OptimizeResult(opt_y0=y0, opt_success=success,
                          opt_residual=residual, opt_nit=it, **sol)