import numpy as np
from extensisq.kennedy import KC           # use 2 interpolants like KC methods
# from extensisq.common import ESDIRK
# from scipy.special import roots_laguerre


class Kv3I(KC):
    """ESDIRK method of Kvaerno [1]_, ESDIRK3/2a with 4 stages. Main method of
    order 3 and embedded method of order 2 are both are stiffly accururate.
    Only the main method is L-stable.

    The method includes two interpolants. The default is C0 continuous and the
    alternative is C1 continuous. The error of the two interpolants is similar.

    The implementation is similar to the implementation of extensisq methods of
    Hosea. It adopts details from the BDF method of scipy and the paper of
    Shampine [2]_ and cubic Hermite spline extrapolation for prediction of some
    of the stages.

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
    interpolant : {'C0', 'C1'}, optional
        Interpolant to use: C0 or C1 continuous. Default is 'C0'.
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
    .. [1] A. Kvaerno, "Singly Diagonally Implicit Runge-Kutta Methods with an
           Explicit First Stage", BIT Numerical Mathematics, Vol. 44, pp.
           489-502, 2004, https://doi.org/10.1023/B:BITN.0000046811.70614.38
    .. [2] L. F. Shampine, "Implementation of Implicit Formulas for the
           Solution of ODEs", SIAM Journal on Scientific and Statistical
           Computing, Vol. 1, No. 1, pp. 103-118, 1980,
           https://doi.org/10.1137/0901005.
    .. [3] G.Söderlind, "Automatic Control and Adaptive Time-Stepping",
           Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310.
           https://doi.org/10.1023/A:1021160023092
    """
    # Kvaerno ESDIRK32a
    n_stages = 4
    order = 3
    order_secondary = 2
    kappa = 1/15                # E has large coefficients
    filter_error = False        # embedded methods has bounded Rh(inf) = 0.9569
    # coefficients
    d = 0.435866521508459       # = 1./roots_laguerre(3)[0][1]
    C = np.array([0, 2*d, 1, 1])
    A = np.array([
        [0, 0, 0, 0],
        [d, d, 0, 0],
        [(6*d - 4*d*d - 1)/(4*d), (1 - 2*d)/(4*d), d, 0],
        [(6*d - 1)/(12*d), -1/(24*d - 12)/d, (6*d - 6*d*d - 1)/(6*d - 3), d]])
    B = A[-1, :]                # R(inf) = 0
    Bh = A[-2, :]               # R_h(inf) = 0.9569
    E = Bh - B
    Az = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1 - 1/(2*d), 1/(2*d), 0, 0],
        [0, 0, 1, 0]])
    # C0 interpolant
    P0 = np.array([
        [1, -1.07357009006976, 0.382380060046507],
        [0, 4.47169016526534, -2.98112677684356],
        [1.0452602553351, -5.79624015039116, 3.51574001514907],
        [-1.0452602553351, 2.39812007519558, -0.91699329835202]])
    # C1 interpolant, similar error as C0
    P1 = np.array([
        [1, -1.07357009006976, 0.382380060046507, 0],
        [0, 4.47169016526534, -2.98112677684356, 0],
        [0, 0.252689145887228, -5.44633781140245, 3.95840878560824],
        [0, -3.65080922108287, 8.04508452819957, -3.95840878560825]])
    P = P0      # default
