import numpy as np
from scipy.integrate._ivp.base import DenseOutput
from scipy.interpolate import CubicHermiteSpline
from extensisq.common import (NFS, NFI, NLS, ESDIRK)


class HS(ESDIRK):
    """Common definitions for the two methods of Hosea and Shampine, including
    dense output."""
    n_stages = 3
    order = 2
    order_secondary = 3
    filter_error = True

    def _dense_output_impl(self):
        """Piecewise cubic Hermite interpolant through the mid point"""
        h = self.h_previous
        t_mid = self.t_old + self.C[1]*h
        y_mid = self.y_old + h*(self.K.T @ self.A[1, :])
        T = [self.t_old, t_mid, self.t]
        Y = np.array([self.y_old, y_mid, self.y])
        dY = self.K.copy()
        if self.direction > 0:
            return PiecewiseCubicDenseOutput(T, Y, dY)
        else:
            return PiecewiseCubicDenseOutput(T[::-1], Y[::-1], dY[::-1])


class PiecewiseCubicDenseOutput(DenseOutput):
    """Cubic, C1 continuous interpolator
    with intermediate point(s) in step interval
    """
    def __init__(self, T, Y, dY):
        t_old = T[0]
        t = T[-1]
        super().__init__(t_old, t)
        self.interpolant = CubicHermiteSpline(T, Y, dY)

    def _call_impl(self, t):
        if t.shape:
            return np.array([self.interpolant(_t) for _t in t]).T
        else:
            return self.interpolant(t)


class TRBDF2(HS):
    """Class `HS2I` and alias `TRBDF2` are implicit ESDIRK methods of order 2
    with an embedded method of order 3 for step size adaptivity. The main
    method is L-stable, the secondary is not. The error prediction is filtered.
    The method includes a piecewise cubic interpolant.

    The method was originally proposed by Banks et al. [3_]. It contains two
    substeps: trapezium followed by BDF. Hosea and Shampine [1]_ discussed many
    of the implementation details that were followed. Other implementation
    details were adopted from the BDF method of scipy and the paper of
    Shampine [2]_.

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
    jac : {None, array_like, sparse_matrix, callable}, optional
        Jacobian matrix of the right-hand side of the system with respect to y,
        required by this method. The Jacobian matrix has shape (n, n) and its
        element (i, j) is equal to ``d f_i / d y_j``.
        There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant. Furthermore, the ODE is assumed to be linear! If the
              If the supplied Jacobian is a constant approximation, but the ODE
              is not linear, then use a callable; see next.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)`` as necessary.
              The return value might be a sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian (options 1 or 2)
        rather than relying on a finite-difference (option 3) approximation.
        The linear ODE assumption entails that only one iteration is done per
        stage, but also that the LU decomposition is done after each change in
        step size. If this is undesirable, then supply the jacobian as a
        callable (option 2).
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This
        argument is ignored if `jac` is not `None`. If the Jacobian has only
        few non-zero elements in *each* row, providing the sparsity structure
        will greatly speed up the computations. A zero entry means that a
        corresponding element in the Jacobian is always zero. If None
        (default), the Jacobian is assumed to be dense.
    M : {None, array_like, sparse}, optional
        The method can solve more general problems (index 1 DAEs) of the form:
            M y' = f(t, y).
        In this case, `M` is a constant matrix of shape (n, n). The user
        supplied M can be a 2D matrix (dense or sparse) or, if M is diagonal,
        a 1D array of the diagonal. Default: None, which implies `M` is the
        identity matrix.
    jac_each_step : bool, optional
        If True, the jacobian is updated each step. Default: False.
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
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by this method, but may result in slower
        execution overall in some circumstances (e.g. small ``len(y0)``).
    sc_params : tuple of size 4, "standard", "G", "S", "S2" or "W", optional
        Parameters for the stepsize controller (k*b1, k*b2, a2, g). The step
        size controller is, with k the exponent of the standard controller,
        _n for new and _o for old:
            h_n = h * g**(k*b1 + k*b2) * (h/h_o)**-a2
                * (err/tol)**-b1 * (err_o/tol_o)**-b2
        sc_params : tuple of size 4, "standard", "G", "H" or "W", optional
        Parameters for the stepsize controller (k*b1, k*b2, a2, g). The step
        size controller is, with k the exponent of the standard controller,
        _n for new and _o for old:
            h_n = h * g**(k*b1 + k*b2) * (h/h_o)**-a2
                * (err/tol)**-b1 * (err_o/tol_o)**-b2
        Predefined parameters are [4]_:
            Gustafsson "G"  (2, -1, -1, 0.8),
            Soederlind "S" (0.6, -0.2, 0, 0.8),
            and "standard" (1, 0, 0, 0.8).
        These coefficients are different than in explicit methods. The default
        for this method is "G".

    References
    ----------
    .. [1] M.E. Hosea, L.F. Shampine, "Analysis and implementation of TR-BDF2",
           Applied Numerical Mathematics, Vol. 20, No. 1-2, pp. 21-37, 1996,
           https://doi.org/10.1016/0168-9274(95)00115-8.
    .. [2] L. F. Shampine, "Implementation of Implicit Formulas for the
           Solution of ODEs", SIAM Journal on Scientific and Statistical
           Computing, Vol. 1, No. 1, pp. 103-118, 1980,
           https://doi.org/10.1137/0901005.
    .. [3] R. E. Bank, W. M. Coughran, W. Fichtner, E. H. Grosse, D. J. Rose
           and R. K. Smith, "Transient Simulation of Silicon Devices and
           Circuits", IEEE Transactions on Computer-Aided Design of Integrated
           Circuits and Systems, Vol. 4, No. 4, pp. 436-451, 1985,
           https://doi.org/10.1109/TCAD.1985.1270142.
    .. [4] G.Söderlind, "Automatic Control and Adaptive Time-Stepping",
           Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310.
           https://doi.org/10.1023/A:1021160023092
    """
    kappa = 0.5

    # coefficients
    s2 = np.sqrt(2)
    d = (2 - s2)/2
    C = np.array([0, 2*d, 1])
    A = np.array([[0, 0, 0],
                  [d, d, 0],
                  [s2/4, s2/4, d]])
    B = A[-1, :]
    Bh = np.array([1/3-s2/12, s2/4+1/3, d/3])
    E = Bh - B
    Az = np.array([[0, 0, 0],
                   [1, 0, 0],
                   [-s2/2, s2/2 + 1, 0]])


class TRX2(HS):
    """Class `HS2Ia` and alias `TRX2` are implicit ESDIRK methods of order 2
    with an embedded method of order 3 for step size adaptivity. The main
    method is A-stable, not L-stable. It is an alternative to HS2I/TRBDF2 "for
    those simulations in which damping at infinity is not appropriate". The
    error prediction is filtered and the method includes a piecewise cubic
    interpolant.

    Like HS2I/TRBDF2 It contains two substeps, but now both are trapezium.
    Hosea and Shampine [1]_ discussed many of the implementation details that
    were followed. Other implementation details were adopted from the BDF
    method of scipy and the paper of Shampine [2]_.

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
    jac : {None, array_like, sparse_matrix, callable}, optional
        Jacobian matrix of the right-hand side of the system with respect to y,
        required by this method. The Jacobian matrix has shape (n, n) and its
        element (i, j) is equal to ``d f_i / d y_j``.
        There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant. Furthermore, the ODE is assumed to be linear! If the
              If the supplied Jacobian is a constant approximation, but the ODE
              is not linear, then use a callable; see next.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)`` as necessary.
              The return value might be a sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian (options 1 or 2)
        rather than relying on a finite-difference (option 3) approximation.
        The linear ODE assumption entails that only one iteration is done per
        stage, but also that the LU decomposition is done after each change in
        step size. If this is undesirable, then supply the jacobian as a
        callable (option 2).
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This
        argument is ignored if `jac` is not `None`. If the Jacobian has only
        few non-zero elements in *each* row, providing the sparsity structure
        will greatly speed up the computations. A zero entry means that a
        corresponding element in the Jacobian is always zero. If None
        (default), the Jacobian is assumed to be dense.
    M : {None, array_like, sparse}, optional
        The method can solve more general problems (index 1 DAEs) of the form:
            M y' = f(t, y).
        In this case, `M` is a constant matrix of shape (n, n). The user
        supplied M can be a 2D matrix (dense or sparse) or, if M is diagonal,
        a 1D array of the diagonal. Default: None, which implies `M` is the
        identity matrix.
    jac_each_step : bool, optional
        If True, the jacobian is updated each step. Default: False.
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
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by this method, but may result in slower
        execution overall in some circumstances (e.g. small ``len(y0)``).
    sc_params : tuple of size 4, "standard", "G", "H" or "W", optional
        Parameters for the stepsize controller (k*b1, k*b2, a2, g). The step
        size controller is, with k the exponent of the standard controller,
        _n for new and _o for old:
            h_n = h * g**(k*b1 + k*b2) * (h/h_o)**-a2
                * (err/tol)**-b1 * (err_o/tol_o)**-b2
        Predefined parameters are [3]_:
            Gustafsson "G"  (2, -1, -1, 0.8),
            Soederlind "S" (0.6, -0.2, 0, 0.8),
            and "standard" (1, 0, 0, 0.8).
        These coefficients are different than in explicit methods. The default
        for this method is "G".

    References
    ----------
    .. [1] M.E. Hosea, L.F. Shampine, "Analysis and implementation of TR-BDF2",
           Applied Numerical Mathematics, Vol. 20, No. 1-2, pp. 21-37, 1996,
           https://doi.org/10.1016/0168-9274(95)00115-8.
    .. [2] L. F. Shampine, "Implementation of Implicit Formulas for the
           Solution of ODEs", SIAM Journal on Scientific and Statistical
           Computing, Vol. 1, No. 1, pp. 103-118, 1980,
           https://doi.org/10.1137/0901005.
    .. [3] G.Söderlind, "Automatic Control and Adaptive Time-Stepping",
           Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310.
           https://doi.org/10.1023/A:1021160023092
    """
    kappa = 1.

    # coefficients
    d = 1/4
    C = np.array([0, 2*d, 1])
    A = np.array([[0, 0, 0],
                  [d, d, 0],
                  [1/4, 1/2, d]])
    B = A[-1, :]
    Bh = np.array([1/6, 2/3, 1/6])
    E = Bh - B
    Az = np.array([[0, 0, 0],
                   [1., 0, 0],
                   [-1, 2, 0]])


# aliases
HS2I = TRBDF2
HS2Ia = TRX2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from math import sin, cos

    plot = False

    norm_factor = 2     # ~ compensate for using RMS-norm instead of max norm
    # sc_params = "standard"
    # sc_params = "S2"
    sc_params = "G"
    rtol = 0.005/norm_factor
    atol = 1e-10/norm_factor
    NFSs = []
    NFIs = []
    NLSs = []

    # linear equation
    y0 = np.array([1., 0.])
    t_span = (0., 12.)

    def fun(t, y):
        dy0 = -500*y[0] + 500*cos(t) - sin(t)
        dy1 = -y[1] + sin(t) + cos(t)
        return np.asarray([dy0, dy1])

    jac = np.array([[-500, 0], [0, -1.]])

    def jac_fun(t, y, jac=jac):
        return jac

    sol1 = solve_ivp(fun, t_span, y0, jac=jac, atol=atol, rtol=rtol,
                     method=TRBDF2, sc_params=sc_params)
    print(f"message case 1: {sol1.message}")
    NFSs.append(NFS[()])
    NFIs.append(NFI[()])
    NLSs.append(NLS[()])

    sol1a = solve_ivp(fun, t_span, y0, jac=jac, atol=atol, rtol=rtol,
                      method=TRX2, sc_params=sc_params, dense_output=True)
    print(f"message case 1a: {sol1a.message}")
    NFSs.append(NFS[()])
    NFIs.append(NFI[()])
    NLSs.append(NLS[()])

    if plot:
        t = np.linspace(*t_span, 1000)
        ref = np.c_[np.cos(t), np.sin(t)]
        plt.plot(t, ref, 'C0', label='ref')
        plt.plot(sol1a.t, sol1a.y.T, '.C1', label='sol')
        plt.plot(t, sol1a.sol(t).T, '--C1', label='sol dense')
        plt.show()

    # D4
    y0 = np.array([1., 1., 0])
    t_span = (0, 50)

    def fun(t, y):
        return np.array([
            -0.013*y[0] - 1000*y[0]*y[2],
            -2500*y[1]*y[2],
            -0.013*y[0] - 1000*y[0]*y[2] - 2500*y[1]*y[2]
            ])

    def jac(t, y):
        return np.array([
            [-0.013 - 1000*y[2], 0, -1000*y[0]],
            [0, -2500*y[2], -2500*y[1]],
            [-0.013 - 1000*y[2], -2500*y[2], -1000*y[0] - 2500*y[1]]
            ])

    sol2 = solve_ivp(fun, t_span, y0, jac=jac, atol=atol, rtol=rtol,
                     method=TRBDF2, sc_params=sc_params)
    print(f"message case 2: {sol2.message}")
    NFSs.append(NFS[()])
    NFIs.append(NFI[()])
    NLSs.append(NLS[()])

    sol2a = solve_ivp(fun, t_span, y0, jac=jac, atol=atol, rtol=rtol,
                      method=TRX2, sc_params=sc_params)
    print(f"message case 2a: {sol2a.message}")
    NFSs.append(NFS[()])
    NFIs.append(NFI[()])
    NLSs.append(NLS[()])

    # non-stif van der pol
    y0 = np.array([0, 0.25])
    t_span = (0, 20)
    eps = 1

    def fun(t, y, eps=eps):
        return np.array([
            y[1],
            eps*(1 - y[0]**2)*y[1] - y[0]
            ])

    def jac(t, y, eps=eps):
        return np.array([
            [0, 1],
            [-2*eps*y[0]*y[1] - 1, eps*(1 - y[0]**2)]
            ])

    sol3 = solve_ivp(fun, t_span, y0, jac=jac, atol=atol, rtol=rtol,
                     method=TRBDF2, sc_params=sc_params)
    print(f"message case 3: {sol3.message}")
    NFSs.append(NFS[()])
    NFIs.append(NFI[()])
    NLSs.append(NLS[()])

    sol3a = solve_ivp(fun, t_span, y0, jac=jac, atol=atol, rtol=rtol,
                      method=TRX2, sc_params=sc_params)
    print(f"message case 3a: {sol3a.message}")
    NFSs.append(NFS[()])
    NFIs.append(NFI[()])
    NLSs.append(NLS[()])

    # Robertson
    y0 = np.array([1., 0, 0])
    t_span = (0, 4e7)

    def fun(t, y):
        return np.array([
            -0.04*y[0] + 10e4*y[1]*y[2],
            0.04*y[0] - 10e4*y[1]*y[2] - 3e7*y[1]**2,
            3e7*y[1]**2
            ])

    def jac(t, y):
        return np.array([
            [-0.04, 10e4*y[2], 10e4*y[1]],
            [0.04, -10e4*y[2] - 6e7*y[1], -10e4*y[1]],
            [0, 6e7*y[1], 0]
            ])

    sol4 = solve_ivp(fun, t_span, y0, jac=jac, atol=atol, rtol=rtol,
                     method=TRBDF2, sc_params=sc_params)
    print(f"message case 4: {sol4.message}")
    NFSs.append(NFS[()])
    NFIs.append(NFI[()])
    NLSs.append(NLS[()])

    solutions = (sol1, sol1a, sol2, sol2a, sol3, sol3a, sol4)
    # output
    print("\ncase/table:               1            1           2           2"
          "           3           3           4")
    print("solver                 TRBDF2        TRX2       TBDF2        TRX2"
          "      TRBDF2        TRX2      TRBDF2")

    ref = (40, 33, 24, 23, 116, 93, 76)
    print("successful steps:    " +
          " ".join(f"{sol.t.size-1:>6}({sol.t.size-1-r:>+3})"
                   for sol, r in zip(solutions, ref)))

    ref = (7, 3, 0, 0, 24, 19, 5)
    print("error failures:      " +
          " ".join(f"{n:>6}({n-r:>+3})" for n, r in zip(NFSs, ref)))

    ref = (0, 0, 0, 0, 1, 2, 5)
    print("iteration failures:  " +
          " ".join(f"{n:>6}({n-r:>+3})" for n, r in zip(NFIs, ref)))

    ref = (139, 105, 75, 114, 557, 482, 399)
    print("fun evaluations:     " +
          " ".join(f"{sol.nfev:>6}({sol.nfev-r:>+3})"
                   for sol, r in zip(solutions, ref)))

    ref = (1, 1, 1, 1, 2, 3, 10)
    print("jac evaluations:     " +
          " ".join(f"{sol.njev:>6}({sol.njev-r:>+3})"
                   for sol, r in zip(solutions, ref)))

    ref = (43, 31, 17, 16, 99, 86, 77)
    print("LU decompositions:   " +
          " ".join(f"{sol.nlu:>6}({sol.nlu-r:>+3})"
                   for sol, r in zip(solutions, ref)))

    ref = (184, 139, 97, 135, 695, 592, 478)
    print("LU solves:           " +
          " ".join(f"{n:>6}({n-r:>+3})" for n, r in zip(NLSs, ref)))


"""This can be compared to the result tables in the paper.
The columns with smoothed first stage and Est are relevant.
The linear problem (1) is treated as linear by supplying a
fixed jacobian (directly instead of though a function).
Many other implementation details may be different, yet the
results of problems 2, 3, and 4 are comparable.
"""
