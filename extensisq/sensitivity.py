import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from extensisq import BS5
from collections import namedtuple


SensitivityOutput = namedtuple("SensitivityOutput", "sensf yf sol")
AdjointSensitivityOutputInt = namedtuple("SensitivityOutput",
                                         "sens G sol_y sol_bw")
AdjointSensitivityOutputEnd = namedtuple("SensitivityOutput",
                                         "sens gf sol_y sol_bw")
PeriodicOutput = namedtuple("PeriodicOutput", "y0 success residual nit sol")


def _test_functions(fun, t0, y0, ndim, args=None, Np=None):
    """test the functions and embed args.
    if np is an integer, the size of the last axis should be np
    """
    assert callable(fun), f"{fun.__name__} should be a function"
    n = y0.size

    # test args
    if args is not None:
        try:
            _ = [*(args)]
        except TypeError:
            raise TypeError("`args` should be a tuple")

        def _fun(t, y, fun=fun, args=args):
            return np.asarray(fun(t, y, *args))
    else:
        _fun = fun

    # test function call
    try:
        test_value = _fun(t0, y0)
    except Exception:
        raise AssertionError(
            f"the function {fun.__name__} should have signature " +
            "f(t, y, *args) where *args is optional")

    # test returned ndim
    if test_value.ndim != ndim:
        raise ValueError(f"{fun.__name__} should return a {ndim}D array")

    # test returned shape
    expected_shape = ndim * [n]
    if Np is not None:
        expected_shape[-1] = Np
    for s, s_ex in zip(test_value.shape, expected_shape):
        if s != s_ex:
            raise ValueError(f"the array returned by {fun.__name__} " +
                             f"should have shape {expected_shape}")

    # return function with embedded args (and B) (and p)
    return _fun


def sens_forward(fun, t_span, y0, jac, dfdp, dy0dp, p, atol=1e-6, rtol=1e-3,
                 method=BS5, dense_output=False, t_eval=None,
                 use_approx_jac=False):
    """Forward sensitivity analysis of an initial value problem.

    The method is called forward sensitivity analysis by [1]_ and internal
    differentiation by [2]_.

    The initial value problem is:
        dy = fun(t, y, p),    y(t0) = y0(p)

    And the result of interest is dy/dp over the integration interval and
    specifically at the endpoint (tf). The problem that is solved internally
    has size ny*(np+1).

    Parameters
    ----------
    fun : callable
        The function of the ODE that is solved with solve_ivp. The calling
        signature is fun(t, y, *p). It should return an array of length (ny,).
        (Same as for calling solve_ivp)
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. (Same as for calling solve_ivp)
    y0 : array_like, shape (ny,)
        Initial state. (Same as for calling solve_ivp)
    jac : callable
        function with signature jac(t, y, *p) that returns the Jacobian
        dfun/dy as an array of size (ny,ny). Unlike for solve_ivp, this is not
        optional and jac should be callable.
    dfdp : callable
        function with signature dfdp(t, y, *p) that returns dfun/dp as an
        array of size (ny,np).
    dy0dp : array_like, shape (ny,np)
        Derivative dy0/dp of the initial solution y0 to the parameter p.
    p : array_like, shape (np,)
        contains the values of the parameters.
    atol : float, or sequence of length ny, optional
        The absolute tolerance for solve_ivp used for solution y. The atol for
        the sensitivity parameters is atol/p_i if p_i is not 0 and atol
        otherwise. Default: 1e-6.
    rtol : float, optional
        The relative tolerance for solve_ivp (used for y and senistivity).
        Default: 1e-3.
    method : solver class or string, optional
        The ODE solver that is used. Default: BS5
    dense_output : bool, optional
        Set this to True if you want the output sol to have a dense output.
        Default: False
    t_eval : array_like or None, optional
        array of output points. The last point in `t_eval` should equal
        t_span[-1]. Default: None
    use_approx_jac : bool, optional
        Use an approximate jacobian for the combined problem. This is only
        relevant for implicit methods. This can save functions calls to
        determine the jacobian numerically, but convergence with an incomplete
        jacobian depends on the solver. Default: False

    Returns
    -------
    sensf : array, shape (ny, np)
        The sensitivity dy/dp at the endpoint.
    yf : array, shape (n, )
        The solution at the endpoint.
    sol : OdeSolution
        The solver output containing the combined problem (flattend).

    References
    ----------
    .. [1] R. Serban, A.C. Hindmarsh, "CVODES: An ODE Solver with Sensitivity
           Analysis Capabilities", 2003
    .. [2] E. Hairer, G. Wanner, S.P. Norsett, "Solving Ordinary Differential
           Equations I", Springer Berlin, Heidelberg, 1993,
           https://doi.org/10.1007/978-3-540-78862-1
    """
    y0 = np.asarray(y0)
    p = np.asarray(p)
    Ny = y0.size
    Np = p.size
    if y0.dtype != np.float:
        raise ValueError("`y0` should have dtype float")

    dy0dp = np.asarray(dy0dp)

    # test inputs
    assert y0.ndim == 1, \
        "`y0` should be a 1d array"
    assert dy0dp.ndim == 2, \
        "`dy0dp` should be a 2d array of size (ny, np)"
    assert (Ny, Np) == dy0dp.shape, \
        "`dy0dp` should be a array of size (ny, np)"
    t0, tf = t_span

    if t_eval is not None:
        assert t_eval[-1] == tf, \
            'if `t_eval` is used, the last point should be t_span[-1]'

    fun = _test_functions(fun, t0, y0, 1, args=p)
    dfdp = _test_functions(dfdp, t0, y0, 2, args=p, Np=Np)
    jac = _test_functions(jac, t0, y0, 2, args=p)

    # set tolerance
    assert isinstance(rtol, float), 'rtol should be a float'
    assert isinstance(atol, float) or len(atol) == Ny, \
        '`atol` should be a float or a sequence of floats of length Ny'
    total_atol = np.empty((Np+1)*Ny)
    total_atol[:Ny] = atol
    for i, _p in enumerate(p, start=1):
        factor = abs(_p)
        factor = factor or 1.
        total_atol[i*Ny:(i+1)*Ny] = atol/factor

    # function to integrate
    def total_fun(t, total_y, fun=fun, dfdy=jac, dfdp=dfdp, Ny=Ny, Np=Np):
        y = total_y[:Ny]
        s = total_y[Ny:].reshape(Ny, Np, order='F')
        dy = fun(t, y)
        ds = dfdy(t, y) @ s + dfdp(t, y)
        return np.concatenate([dy, ds.reshape(-1, order='F')])

    # solve the combined IVP
    s0 = dy0dp
    total_y0 = np.concatenate([y0, s0.reshape(-1, order='F')])
    if not use_approx_jac:
        if method not in ['BDF', 'Radau']:
            sol = solve_ivp(total_fun, t_span, total_y0,
                            atol=total_atol, rtol=rtol, method=method,
                            dense_output=dense_output, t_eval=t_eval)
        else:
            jac_sparsity = np.zeros(2*[Ny*(Np+1)])
            jac_sparsity[:, :Ny] = 1
            for i in range(Np):
                jac_sparsity[(i+1)*Ny:(i+2)*Ny, (i+1)*Ny:(i+2)*Ny] = 1

            sol = solve_ivp(total_fun, t_span, total_y0,
                            atol=total_atol, rtol=rtol, method=method,
                            dense_output=dense_output, t_eval=t_eval,
                            jac_sparsity=jac_sparsity)
    else:

        def total_jac(t, y, jac=jac, Ny=Ny):
            """approximate Jacobian"""
            _y = y[:Ny]
            _jac = jac(t, _y)
            D = (Np + 1)*[_jac]
            return block_diag(*D)

        sol = solve_ivp(total_fun, t_span, total_y0,
                        atol=total_atol, rtol=rtol, method=method,
                        dense_output=dense_output, t_eval=t_eval,
                        jac=total_jac)
    if not sol.success:
        raise RuntimeError("IVP solver not converged")

    # output
    yf = sol.y[:Ny, -1]
    sensf = sol.y[Ny:, -1].reshape(Ny, Np, order='F')
    return SensitivityOutput(sensf, yf, sol)


def sens_adjoint_end(fun, t_span, y0, jac, dfdp, dy0dp, p, g, dgdp, dgdy,
                     method=BS5, rtol=1e-3, atol=1e-6, atol_adj=1e-6,
                     atol_quad=1e-6, sol_y=None):
    """sensitivity for a scalar function of the solution using the adjoint
    method.

    Define a function involving time, the solution of the IVP and parameters p:
    g(t, y, p). `sens_adjoint_tf` calculates its sensitivity at the end of
    the integration interval: dg/dp(tf). See [1]_ for details

    Parameters
    ----------
    fun : callable
        The function of the ODE that is solved with solve_ivp. The calling
        signature is fun(t, y, *p). It should return an array of length (ny,).
        (Same as for calling solve_ivp)
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. (Same as for calling solve_ivp)
    y0 : array_like, shape (ny,)
        Initial state. (Same as for calling solve_ivp)
    jac : callable
        function with signature jac(t, y, *p) that returns the Jacobian
        dfun/dy as an array of size (ny,ny). Unlike for solve_ivp, this is not
        optional and jac should be callable.
    dfdp : callable
        function with signature dfdp(t, y, *p) that returns dfun/dp as an
        array of size (ny,np).
    dy0dp : array_like, shape (ny,np)
        Derivative dy0/dp of the initial solution y0 to the parameter p.
    p : array_like, shape (np,)
        contains the values of the parameters.
    g : callable
        The function to calculate the senistivity of, with signature
        fun(t, y, *p) -> array of size 1.
    dgdp : callable
        The derivative of function g to parameters p, with signature
        dgdp(t, y, *p) -> array of size np.
    dgdy : callable
        The derivative of function g to parameters y, with signature
        dgdy(t, y, *p) -> array of size ny.
    method : solver class or string, optional
        The ODE solver that is used. Default: BS5
    rtol : float, optional
        The relative tolerance for solve_ivp (used for y and sensitivity).
        Default: 1e-3.
    atol : float, or sequence of length ny, optional
        The absolute tolerance for solve_ivp used for solution y.
        Default: 1e-6.
    atol_adj : float, or sequence of length ny, optional
        The absolute tolerance for solve_ivp used for adjoint solution mu,
        by default 1e-6
    atol_quad : _type_, optional
        The absolute tolerance for solve_ivp used for solution of the definite
        integral, by default 1e-6
    sol_y : OdeResult or None, optional
        if an OdeResult of y is already available, including dense outout, then
        it can be provided and the forward integration will be skipped to save
        time. Default None

    Returns
    -------
    sens : array, shape (np)
        The sensitivity dg/dp at the endpoint tf
    gf : float
        The value of g at tf
    sol_y : OdeResult
        The solution of the forward solve of y
    sol_bw : OdeResult
        The solution of the backward solve of mu and the definite integral

    References
    ----------
    .. [1] R. Serban, A.C. Hindmarsh, "CVODES: An ODE Solver with Sensitivity
           Analysis Capabilities", 2003
    """
    # test inputs
    y0 = np.asarray(y0)
    Ny = y0.size
    if y0.ndim != 1:
        raise ValueError("`y0` should be a 1d array")
    if y0.dtype != np.float:
        raise ValueError("`y0` should have dtype float")

    p = np.asarray(p)
    Np = p.size
    if p.ndim != 1:
        raise ValueError("`p` should be a 1d array")
    if p.dtype != np.float:
        raise ValueError("`p` should have dtype float")

    dy0dp = np.asarray(dy0dp)
    if dy0dp.ndim != 2:
        raise ValueError("`dy0dp` should be a 2d array of size (ny, np)")
    _Ny, _Np = dy0dp.shape
    if _Ny != Ny or _Np != Np:
        raise ValueError("`dy0dp` should be a array of shape (ny, np)")

    t0, tf = t_span
    fun = _test_functions(fun, t0, y0, 1, args=p)
    dfdp = _test_functions(dfdp, t0, y0, 2, args=p, Np=Np)
    jac = _test_functions(jac, t0, y0, 2, args=p)
    dgdy = _test_functions(dgdy, t0, y0, 1, args=p)
    dgdp = _test_functions(dgdp, t0, y0, 1, args=p, Np=Np)
    g = _test_functions(g, t0, y0, 1, args=p, Np=1)

    # forward solve of y
    if sol_y is not None:
        if sol_y.sol is None:
            raise ValueError("sol_y should have a dense output")
    else:
        if method in ("LSODA", "BDF", "Radau"):
            # implicit method
            sol_y = solve_ivp(
                fun, t_span, y0, method=method, atol=atol, rtol=rtol,
                dense_output=True, jac=jac)
        else:
            # explicit method
            sol_y = solve_ivp(
                fun, t_span, y0, method=method, atol=atol, rtol=rtol,
                dense_output=True)
        if not sol_y.success:
            raise RuntimeError(
                "IVP solver not converged in forward solve of y")

    # backward solve of adjoint problem with mu and xi combined
    # xi is for the integral
    def fun_bw(t, total_y, y=sol_y.sol, jac=jac, dfdp=dfdp, Ny=Ny):
        _mu = total_y[:Ny]
        _y = y(t)
        _jac = jac(t, _y)
        _dfdp = dfdp(t, _y)
        dmu = -(_jac.T @ _mu)
        dxi = _dfdp.T @ _mu
        return np.concatenate([dmu, dxi])

    yf = sol_y.sol(tf)
    yf_bw = np.concatenate([dgdy(tf, yf), np.zeros(Np)])
    atol_bw = np.zeros(Ny + Np)
    atol_bw[:Ny] = atol_adj
    atol_bw[Ny:] = atol_quad

    if method not in ('LSODA', 'BDF', 'Radau'):
        # explicit method
        sol_bw = solve_ivp(fun_bw, (tf, t0), yf_bw, method=method,
                           atol=atol_bw, rtol=rtol)
    else:
        # implicit method
        def jac_bw(t, _, y=sol_y.sol, jac=jac, dfdp=dfdp, Ny=Ny, Np=Np):
            _y = y(t)
            _jac = jac(t, _y)
            _dfdp = dfdp(t, _y)
            jac_bw = np.zeros((Ny + Np, Ny + Np))
            jac_bw[:Ny, :Ny] = -_jac.T
            jac_bw[Ny:, :Ny] = _dfdp.T
            return jac_bw

        sol_bw = solve_ivp(fun_bw, (tf, t0), yf_bw, method=method,
                           atol=atol_bw, rtol=rtol, jac=jac_bw)
    if not sol_bw.success:
        raise RuntimeError(
            "IVP solver not converged in backward solve of lambda")

    # final result
    mu0 = sol_bw.y[:Ny, -1]
    integral = -sol_bw.y[Ny:, -1]
    sens = dgdp(tf, yf) + mu0 @ dy0dp + integral
    return AdjointSensitivityOutputEnd(sens, g(tf, yf), sol_y, sol_bw)


def sens_adjoint_int(fun, t_span, y0, jac, dfdp, dy0dp, p, g, dgdp, dgdy,
                     method=BS5, rtol=1e-3, atol=1e-6, atol_adj=1e-6,
                     atol_quad=1e-6, sol_y=None):
    """Calculate the sensitivity dG/dp, where G is the integral over t_span of
    a function g(t, y, p). See [1]_ for details

    Parameters
    ----------
    fun : callable
        The function of the ODE that is solved with solve_ivp. The calling
        signature is fun(t, y, *p). It should return an array of length (ny,).
        (Same as for calling solve_ivp)
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. (Same as for calling solve_ivp)
    y0 : array_like, shape (ny,)
        Initial state. (Same as for calling solve_ivp)
    jac : callable
        function with signature jac(t, y, *p) that returns the Jacobian
        dfun/dy as an array of size (ny,ny). Unlike for solve_ivp, this is not
        optional and jac should be callable.
    dfdp : callable
        function with signature dfdp(t, y, *p) that returns dfun/dp as an
        array of size (ny,np).
    dy0dp : array_like, shape (ny,np)
        Derivative dy0/dp of the initial solution y0 to the parameter p.
    p : array_like, shape (np,)
        contains the values of the parameters.
    g : callable
        The function to calculate the senistivity of, with signature
        fun(t, y, *p) -> scalar.
    dgdp : callable
        The derivative of function g to parameters p, with signature
        dgdp(t, y, *p) -> array of size np.
    dgdy : callable
        The derivative of function g to parameters y, with signature
        dgdy(t, y, *p) -> array of size ny.
    method : solver class or string, optional
        The ODE solver that is used. Default: BS5
    rtol : float, optional
        The relative tolerance for solve_ivp (used for y and sensitivity).
        Default: 1e-3.
    atol : float, or sequence of length ny, optional
        The absolute tolerance for solve_ivp used for solution y.
        Default: 1e-6.
    atol_adj : float, or sequence of length ny, optional
        The absolute tolerance for solve_ivp used for adjoint solution lambda,
        by default 1e-6
    atol_quad : _type_, optional
        The absolute tolerance for solve_ivp used for solution of the definite
        integral, by default 1e-6
    sol_y : OdeResult or None, optional
        if an OdeResult of y is already available, including dense outout, then
        it can be provided and the forward integration will be skipped to save
        time. Default None

    Returns
    -------
    sens : array, shape (np)
        The sensitivity dg/dp at the endpoint tf
    G : float
        The value of G
    sol_y : OdeResult
        The solution of the forward solve of y
    sol_bw : OdeResult
        The solution of the backward solve of lambda and the definite integral

    References
    ----------
    .. [1] R. Serban, A.C. Hindmarsh, "CVODES: An ODE Solver with Sensitivity
           Analysis Capabilities", 2003
    """
    # test inputs
    y0 = np.asarray(y0)
    Ny = y0.size
    if y0.ndim != 1:
        raise ValueError("`y0` should be a 1d array")
    if y0.dtype != np.float:
        raise ValueError("`y0` should have dtype float")

    p = np.asarray(p)
    Np = p.size
    if p.ndim != 1:
        raise ValueError("`p` should be a 1d array")
    if p.dtype != np.float:
        raise ValueError("`p` should have dtype float")

    dy0dp = np.asarray(dy0dp)
    if dy0dp.ndim != 2:
        raise ValueError("`dy0dp` should be a 2d array of size (ny, np)")
    _Ny, _Np = dy0dp.shape
    if _Ny != Ny or _Np != Np:
        raise ValueError("`dy0dp` should be a array of shape (ny, np)")

    t0, tf = t_span
    fun = _test_functions(fun, t0, y0, 1, args=p)
    dfdp = _test_functions(dfdp, t0, y0, 2, args=p, Np=Np)
    jac = _test_functions(jac, t0, y0, 2, args=p)
    dgdy = _test_functions(dgdy, t0, y0, 1, args=p)
    dgdp = _test_functions(dgdp, t0, y0, 1, args=p, Np=Np)
    g = _test_functions(g, t0, y0, 1, args=p, Np=1)

    # forward solve of y
    if sol_y is not None:
        if sol_y.sol is None:
            raise ValueError("sol_y should have a dense output")
    else:
        if method in ("LSODA", "BDF", "Radau"):
            # implicit method
            sol_y = solve_ivp(
                fun, t_span, y0, method=method, atol=atol, rtol=rtol,
                dense_output=True, jac=jac)
        else:
            # explicit method
            sol_y = solve_ivp(
                fun, t_span, y0, method=method, atol=atol, rtol=rtol,
                dense_output=True)
        if not sol_y.success:
            raise RuntimeError(
                "IVP solver not converged in forward solve of y")

    # backward solve of adjoint problem with lambda, xi and zeta combined
    # xi is for the integral, zeta is for G
    def fun_bw(t, total_y, y=sol_y.sol, jac=jac, dgdy=dgdy, dgdp=dgdp,
               dfdp=dfdp, g=g, Ny=Ny):
        _lambda = total_y[:Ny]
        _y = y(t)
        _jac = jac(t, _y)
        _dgdy = dgdy(t, _y)
        _dgdp = dgdp(t, _y)
        _dfdp = dfdp(t, _y)
        dlambda = -(_jac.T @ _lambda + _dgdy.T)
        dxi = _dfdp.T @ _lambda + _dgdp
        dzeta = g(t, _y)
        return np.concatenate([dlambda, dxi, dzeta])

    yf_bw = np.zeros(Ny + Np + 1)
    atol_bw = np.zeros(Ny + Np + 1)
    atol_bw[:Ny] = atol_adj
    atol_bw[Ny:] = atol_quad

    if method not in ('LSODA', 'BDF', 'Radau'):
        # explicit method
        sol_bw = solve_ivp(fun_bw, (tf, t0), yf_bw, method=method,
                           atol=atol_bw, rtol=rtol)
    else:
        # implicit method
        def jac_bw(t, _, y=sol_y.sol, jac=jac, dfdp=dfdp, Ny=Ny, Np=Np):
            _y = y(t)
            _jac = jac(t, _y)
            _dfdp = dfdp(t, _y)
            jac_bw = np.zeros((Ny + Np + 1, Ny + Np + 1))
            jac_bw[:Ny, :Ny] = -_jac.T
            jac_bw[Ny:-1, :Ny] = _dfdp.T
            return jac_bw

        sol_bw = solve_ivp(fun_bw, (tf, t0), yf_bw, method=method,
                           atol=atol_bw, rtol=rtol, jac=jac_bw)
    if not sol_bw.success:
        raise RuntimeError(
            "IVP solver not converged in backward solve of lambda")

    # final result
    lambda0 = sol_bw.y[:Ny, -1]
    integral = -sol_bw.y[Ny:-1, -1]
    G = -sol_bw.y[-1, -1]
    sens = lambda0 @ dy0dp + integral
    return AdjointSensitivityOutputInt(sens, G, sol_y, sol_bw)
