import numpy as np
from extensisq.common import ESDIRK


class KC(ESDIRK):
    """Allow selection of interpolator for Kennedy-Carpenter methods and
    disable error filtering. Otherwise it is just a ESDIRK class.
    """

    filter_error = False                        # embedded methods are L-stable

    def __init__(self, *args, interpolant='C0', **kwargs):
        if interpolant == 'C0':
            self.P = self.P0
        elif interpolant == 'C1':
            self.P = self.P1
        else:
            raise ValueError(f'Unknown interpolant {interpolant}, '
                             'must be "C0" or "C1"')
        return super().__init__(*args, **kwargs)


class KC3I(KC):
    """ESDIRK method of Kenedy and Carpenter [1]_, ESDIRK3(2)5L[2]SA.
    Main method of order 3 and embedded method of order 2 are both L-stable.

    The method includes two interpolants: a quartic C1 continuous interpolant
    and a cubic C0 continuous interpolant. The default is the cubic C0
    interpolant.

    Unfortunately, the interpolant in the paper does not match the output at
    the endpoint and is not C0 continuous. Therefore, an alternative C0
    interpolant is constructed. It may not follow all principles of the paper.

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
    .. [1] C. A. Kennedy, M.H. Carpenter, "Diagonally Implicit Runge-Kutta
           Methods for Ordinary Differential Equations. A Review", Langley
           Research Center, document ID 20160005923,
           https://ntrs.nasa.gov/citations/20160005923.
    .. [2] L. F. Shampine, "Implementation of Implicit Formulas for the
           Solution of ODEs", SIAM Journal on Scientific and Statistical
           Computing, Vol. 1, No. 1, pp. 103-118, 1980,
           https://doi.org/10.1137/0901005.
    .. [3] G.Söderlind, "Automatic Control and Adaptive Time-Stepping",
           Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310.
           https://doi.org/10.1023/A:1021160023092
    """
    # ESDIRK3(2)5L[2]SA
    n_stages = 5
    order = 3
    order_secondary = 2
    kappa = 1.

    # coefficients
    d = 9/40
    s2 = np.sqrt(2)
    C = np.array([0, 2*d, 9*(2+s2)/40, 3/5, 1])
    A = np.array([
        [0, 0, 0, 0, 0],
        [d, d, 0, 0, 0],
        [9*(1+s2)/80, 9*(1+s2)/80, d, 0, 0],
        [(22+15*s2)/(80*(1+s2)), (22+15*s2)/(80*(1+s2)), -7/(40*(1+s2)), d, 0],
        [(2398+1205*s2)/(2835*(4+3*s2)), (2398+1205*s2)/(2835*(4+3*s2)),
         -2374*(1+2*s2)/(2835*(5+3*s2)), 5827/7560, d]])
    B = A[-1, :]
    Bh = np.array([
        4555948517383/24713416420891, 4555948517383/24713416420891,
        -7107561914881/25547637784726, 30698249/44052120, 49563/233080])
    E = Bh - B
    Az = np.array([  # Cubic Hermite spline inter/extrapolations
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [-s2/2, s2/2+1, 0, 0, 0],
        [-5/3+4*s2/3, 0, 8/3-4*s2/3, 0, 0],
        [(3367-2380*s2)/27, 1420/27-340*s2/9, (3400*s2-4760)/27, 0, 0]
        # [0, 0, 1, 0, 0]  # simple alternative for last stage
        ])
    # quartic C1 continuous free interpolant through 3rd order accurate
    # L-stable point at c = 3/5 (not the stage point, these are 2nd order).
    # Compared to the cubic Hermite interpolation, this reduces T4 slightly
    # throughout the step.
    P1 = np.array([
        [1, (265868*s2-529769)/87405, (123167299-72831508*s2)/16519545,
         (1966607*s2-3096146)/1101303],
        [0, (265868*s2-209284)/87405, (51582604-72831508*s2)/16519545,
         (1966607*s2-1260641)/1101303],
        [0, 531736*(1-s2)/87405, 145663016*(s2-1)/16519545,
         3933214*(1-s2)/1101303],
        [0, 694367/349620, -4895596/5506515, -958103/2936808],
        [0, 44967/116540, -25397/29135, 33137/46616]])
    # cubic C0 interpolant (through same L-stable point at 3/5). The first two
    # stages are kept identical. The only remaining free parameter is set to
    # sqrt(2)/2, which is near optimal for ||T4||.
    P0 = np.array([
        [(9035708 - 5275768*s2)/1835505, (-81688633 + 53056007*s2)/5506515,
         (5384 - 3596*s2)/567],
        [(9035708 - 5275768*s2)/1835505, (-81688633 + 53056007*s2)/5506515,
         (5384 - 3596*s2)/567],
        [-2567144/367101 + 8593664*s2/1835505,
         (124385486 - 90449038*s2)/5506515, (-8368 + 6184*s2)/567],
        [-345291/203945 + 23*s2/30, 2185191/326312 - 92*s2/45,
         -800/189 + 23*s2/18],
        [-4644/29135 + 3*s2/10, 17919/46616 - 4*s2/5, s2/2]])
    # from paper (output at endpoint doesn't match B):
    # Pp = np.array([
    #     [18390937872020/16547330141131, -25205650154962/13994999269151,
    #      4643928352124/5273763430929],
    #     [18390937872020/16547330141131, -25205650154962/13994999269151,
    #      4643928352124/5273763430929],
    #     [45873276387100/11281280648079, -83784512863764/7610748870347,
    #      87858418189205/12798018608062],
    #     [-169812/40789, 493900/40789, -8139200/1101303],
    #     [-6561/5827, 14580/5827, -7200/5827]])
    P = P0  # default


class KC4I(KC):
    """ESDIRK method of Kenedy and Carpenter [1]_, ESDIRK4(3)6L[2]SA.
    Main method of order 4 and embedded method of order 3 are both L-stable.

    The method includes two interpolants: a quartic C1 continuous interpolant
    and a quartic C0 continuous interpolant. The default is the cubic C0
    interpolant which was given in the paper.

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
    .. [1] C. A. Kennedy, M.H. Carpenter, "Diagonally Implicit Runge-Kutta
           Methods for Ordinary Differential Equations. A Review", Langley
           Research Center, document ID 20160005923,
           https://ntrs.nasa.gov/citations/20160005923.
    .. [2] L. F. Shampine, "Implementation of Implicit Formulas for the
           Solution of ODEs", SIAM Journal on Scientific and Statistical
           Computing, Vol. 1, No. 1, pp. 103-118, 1980,
           https://doi.org/10.1137/0901005.
    .. [3] G.Söderlind, "Automatic Control and Adaptive Time-Stepping",
           Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310.
           https://doi.org/10.1023/A:1021160023092
    """
    # ESDIRK4(3)6L[2]SA
    n_stages = 6
    order = 4
    order_secondary = 3
    kappa = 1/2

    # coefficients
    d = 1/4
    s2 = np.sqrt(2)
    C = np.array(
        [0, 2*d, (2-s2)/4, 5/8, 26/25, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [d, d, 0, 0, 0, 0],
        [(1-s2)/8, (1-s2)/8, d, 0, 0, 0],
        [(5-7*s2)/64, (5-7*s2)/64, 7*(1+s2)/32, d, 0, 0],
        [(-13796-54539*s2)/125000, (-13796-54539*s2)/125000,
         (506605+132109*s2)/437500, 166*(-97+376*s2)/109375, d, 0],
        [(1181-987*s2)/13782, (1181-987*s2)/13782, 47*(-267+1783*s2)/273343,
         -16*(-22922+3525*s2)/571953, -15625*(97+376*s2)/90749876, d]
        ])
    B = A[-1, :]
    Bh = np.array([
        -480923228411/4982971448372, -480923228411/4982971448372,
        6709447293961/12833189095359, 3513175791894/6748737351361,
        -498863281070/6042575550617, 2077005547802/8945017530137])
    E = Bh - B
    Az = np.array([  # Cubic Hermite spline inter/extrapolations
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [s2/2, 1-s2/2, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [-33366/30625+68973*s2/61250, -33366/30625+68973*s2/61250,
         -926197/214375+403961*s2/214375, 1607696/214375-886772*s2/214375,
         0, 0],
        [0, 0, 0, 0, 1, 0]
        ])
    # C1 continuous quartic interpolant, 4th order accurate. The only free
    # parameter is set for optimal ||T5||. No L-stable mid-points are used.
    P1 = np.array([
        [1.0, -4.045351040436539, 5.028351540730211, -1.998588135329389],
        [0, 2.37553630334257, -4.813423146828023, 2.422299208449723],
        [0, 3.813260102423869, -6.075889521194925, 2.650287089684259],
        [0, -2.329870780844837, 6.666832039978334, -3.835188639561323],
        [0, 0.5884164513971198, -1.609852984449971, 0.9131815126389187],
        [0, -0.4019910358821925, 0.803982071764385, -0.1519910358821925]])
    # quartic interpolant from paper, C0 continuous, 4th order accurate.
    P0 = np.array([
        [11963910384665/12483345430363, 11963910384665/12483345430363,
         -28603264624/1970169629981, -3524425447183/2683177070205,
         -17173522440186/10195024317061, 27308879169709/13030500014233],
        [-69996760330788/18526599551455, -69996760330788/18526599551455,
         102610171905103/26266659717953, 74957623907620/12279805097313,
         113853199235633/9983266320290, -84229392543950/6077740599399],
        [32473635429419/7030701510665, 32473635429419/7030701510665,
         -38866317253841/6249835826165, -26705717223886/4265677133337,
         -121105382143155/6658412667527, 1102028547503824/51424476870755],
        [-14668528638623/8083464301755, -14668528638623/8083464301755,
         21103455885091/7774428730952, 30155591475533/15293695940061,
         119853375102088/14336240079991, -63602213973224/6753880425717]]).T
    P = P0  # default


class KC4Ia(KC):
    """ESDIRK method of Kenedy and Carpenter [1]_, ESDIRK4(3)7L[2]SA.
    Main method of order 4 and embedded method of order 3 are both L-stable.
    The difference with KC4I is that this method has a additional stage an a
    diagonal ESDIRK a-coefficient of only 1/8 (versus 1/4). The error per step
    should be considerably smaller than that of KC4I.

    The method includes two interpolants: a quartic C1 continuous interpolant
    and a quartic C0 continuous interpolant. The default is the C0 interpolant.
    The paper does not provide interpolants, so these are constructed for
    extensisq and may not follow all the principles of the paper.

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
    .. [1] C. A. Kennedy, M.H. Carpenter, "Diagonally implicit Runge-Kutta
           methods for stiff ODEs", Applied Numerical Mathematics, Vol. 146,
           pp. 221-244, 2019, https://doi.org/10.1016/j.apnum.2019.07.008.
    .. [2] L. F. Shampine, "Implementation of Implicit Formulas for the
           Solution of ODEs", SIAM Journal on Scientific and Statistical
           Computing, Vol. 1, No. 1, pp. 103-118, 1980,
           https://doi.org/10.1137/0901005.
    .. [3] G.Söderlind, "Automatic Control and Adaptive Time-Stepping",
           Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310.
           https://doi.org/10.1023/A:1021160023092
    """
    # ESDIRK4(3)7L[2]SA
    n_stages = 7
    order = 4
    order_secondary = 3
    kappa = 0.2

    # coefficients
    d = 1/8
    C = np.array(
        [0, 2*d, 1200237871921/16391473681546, 1/2, 395/567, 89/126, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [d, d, 0, 0, 0, 0, 0],
        [-39188347878/1513744654945, -39188347878/1513744654945, d,
         0, 0, 0, 0],
        [1748874742213/5168247530883, 1748874742213/5168247530883,
         -1748874742213/5795261096931, d, 0, 0, 0],
        [-6429340993097/17896796106705, -6429340993097/17896796106705,
         9711656375562/10370074603625, 1137589605079/3216875020685, d, 0, 0],
        [405169606099/1734380148729, 405169606099/1734380148729,
         -264468840649/6105657584947, 118647369377/6233854714037,
         683008737625/4934655825458, d, 0],
        [-5649241495537/14093099002237, -5649241495537/14093099002237,
         5718691255176/6089204655961, 2199600963556/4241893152925,
         8860614275765/11425531467341, -3696041814078/6641566663007, d]
        ])
    B = A[-1, :]
    Bh = np.array([
        -1517409284625/6267517876163, -1517409284625/6267517876163,
        8291371032348/12587291883523, 5328310281212/10646448185159,
        5405006853541/7104492075037, -4254786582061/7445269677723, 19/140])
    E = Bh - B
    Az = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0.7071067811865475, 0.2928932188134525, 0, 0, 0, 0, 0],
        [-1, 2, 0, 0, 0, 0, 0],
        [-3.44712536068413, -3.44712536068413, 5.597060516320929,
         2.297190205047331, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [-1.669640264560983, -1.669640264560983, 2.655940863056583,
         -0.1224584413467299, -0.8905466663459783, 2.696344773758092, 0]
        ])
    # Quartic C1 continuous interpolant with 4th order accuracy. The
    # interpolant has two free parameters that were set by minimizing the
    # squared error.
    P1 = np.array([
        [1, -0.2618535743659094, -4.079699311306614,  2.940701270662915],
        [0,   7.936037051221989, -17.47548056248241,  9.138591896250813],
        [0,  -5.459341005691339,  14.67529166947831, -8.276798249263065],
        [0,   -1.71616804025474,  5.506505216089204, -3.271794891939533],
        [0,  -1.357746919580548,  5.817533967829905, -3.684277016082155],
        [0,  0.4746238387074965, -3.175253679682295,  2.144128340407973],
        [0,  0.3844486499630515, -1.268897299926103,  1.009448649963051]])
    # Quartic C0 continuous interpolant with 4th order accuracy. The first to
    # stages are kept identical. The interpolant passes through 2 L-stable
    # points (at c=0.08493322596570153 and c=0.5709617099460419).
    P0 = np.array([
        [1.751737353544831, -11.67529655238061, 19.33290157583574,
         -9.810193992009567],
        [1.751737353544831, -11.67529655238061, 19.33290157583574,
         -9.810193992009567],
        [-1.65533494856435, 17.31058741533003, -30.87602826958563,
         16.15992821734386],
        [-1.585702145515156, 12.32790098775139, -18.38973592743151,
         8.166079369090214],
        [1.480670047540331, -13.43221397813055, 25.48988276667239,
         -12.76282880391497],
        [-0.6392836047534802, 6.127838415643804, -12.78698063999065,
         6.741924328533505],
        [-0.1038240557970074, 1.016480264166559, -2.102941081336072,
         1.315284872966521]])
    P = P0  # default
