import numpy as np
from warnings import warn
from scipy.integrate._ivp.rk import norm, SAFETY, MAX_FACTOR, MIN_FACTOR
from extensisq.common import RungeKutta, HornerDenseOutput, NFS


class BS45(RungeKutta):
    """Explicit Runge-Kutta method of order 5(4).

    This uses the Bogacki-Shampine pair of formulas [1]_. It is designed
    to be more efficient than the Dormand-Prince pair (RK45 in scipy).

    There are two independent fourth order estimates of the local error.
    The fifth order method is used to advance the solution (local
    extrapolation). Coefficients from [2]_ are used.

    The interpolator for dense output is of fifth order and needs three
    additional derivative function evaluations (when used). A free, fourth
    order interpolator is also available as method BS45_i.

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

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian. Is always 0 for this solver as
        it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine, "An efficient Runge-Kutta (4,5) pair",
           Computers & Mathematics with Applications, Vol. 32, No. 6, 1996,
           pp. 15-28, ISSN 0898-1221.
           https://doi.org/10.1016/0898-1221(96)00141-1
    .. [2] RKSUITE: https://www.netlib.org/ode/rksuite/
    """

    order = 5
    error_estimator_order = 4
    n_stages = 7        # the effective nr (total nr of stages is 8)
    n_extra_stages = 3  # for dense output
    tanang = 5.2
    stbrad = 3.9

    # time step fractions
    C = np.array([0, 1/6, 2/9, 3/7, 2/3, 3/4, 1])

    # coefficient matrix, including row of last stage
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/6, 0, 0, 0, 0, 0, 0],
        [2/27, 4/27, 0, 0, 0, 0, 0],
        [183/1372, -162/343, 1053/1372, 0, 0, 0, 0],
        [68/297, -4/11, 42/143, 1960/3861, 0, 0, 0],
        [597/22528, 81/352, 63099/585728, 58653/366080, 4617/20480, 0, 0],
        [174197/959244, -30942/79937, 8152137/19744439, 666106/1039181,
         -29421/29068, 482048/414219, 0]
    ])

    # coefficients for propagating method
    B = np.array([587/8064, 0, 4440339/15491840, 24353/124800,
                  387/44800, 2152/5985, 7267/94080])

    # coefficients for first error estimation method
    E_pre = np.array([-3/1280, 0, 6561/632320, -343/20800, 243/12800, -1/95])

    # coefficients for main error estimation method
    E = np.array([2479/34992, 0, 123/416, 612941/3411720, 43/1440,
                  2272/6561, 79937/1113912, 3293/556956])
    E[:-1] -= B   # convert to error coefficients

    # extra time step fractions for dense output
    C_extra = np.array([1/2, 5/6, 1/9])

    # coefficient matrix for dense output
    A_extra = np.array([
        [455/6144, -837888343715/13176988637184,
            98719073263/1551965184000],
        [0, 30409415/52955362, 1307/123552],
        [10256301/35409920, -48321525963/759168069632,
            4632066559387/70181753241600],
        [2307361/17971200, 8530738453321/197654829557760,
            7828594302389/382182512025600],
        [-387/102400, 1361640523001/1626788720640, 40763687/11070259200],
        [73/5130, -13143060689/38604458898, 34872732407/224610586200],
        [-7267/215040, 18700221969/379584034816, -2561897/30105600],
        [1/32, -5831595/847285792, 1/10],
        [0, -5183640/26477681, -1/10],
        [0, 0, -1403317093/11371610250]]).T

    # coefficients for interpolation (high order, default)
    P = np.array([
        [0, -11513270273/3502699200, -87098480009/5254048800,
            -2048058893/59875200, -1620741229/50038560,
            -12134338393/1050809760],
        [0, 0, 0, 0, 0, 0],
        [0, -29327744613/2436866432, -69509738227/1218433216,
            -39991188681/374902528, -539868024987/6092166080,
            -33197340367/1218433216],
        [0, -2382590741699/331755652800, -16209923456237/497633479200,
            -333945812879/5671036800, -7896875450471/165877826400,
            -284800997201/19905339168],
        [0, -36591193/86486400, -32406787/18532800, -633779/211200,
            -103626067/43243200, -540919/741312],
        [0, -611586736/89131185, -3357024032/1871754885, 183022264/5332635,
            30405842464/623918295, 7157998304/374350977],
        [0, -65403/15680, -385151/15680, -1620541/31360, -719433/15680,
            -138073/9408],
        [1, 149/16, 2501/64, 4715/64, 3991/64, 1245/64],
        [0, 16, 199/3, 103, 71, 55/3],
        [0, -423642896/126351225, -11411880511/379053675, -26477681/359975,
            -1774004627/25270245, -1774004627/75810735],
        [0, 12, 59, 117, 105, 35]])

    def __init__(self, fun, t0, y0, t_bound, nfev_stiff_detect=5000,
                 **extraneous):
        super(BS45, self).__init__(
            fun, t0, y0, t_bound, nfev_stiff_detect=nfev_stiff_detect,
            **extraneous)
        # custom initialization to create extended storage for dense output
        self.K_extended = np.zeros((self.n_stages+self.n_extra_stages+1,
                                    self.n), dtype=self.y.dtype)
        self.K = self.K_extended[:self.n_stages+1]
        # y_old is used for first error assessment, it should not be None
        self.y_old = self.y - self.direction * self.h_abs * self.f

    def _step_impl(self):

        # mostly follows the scipy implementation of RungeKutta
        t = self.t
        y = self.y

        # limit step size
        h_abs = self.h_abs
        min_step = max(self.h_min_a * (abs(t) + h_abs), self.h_min_b)
        h_abs = min(self.max_step, max(min_step, h_abs))

        # handle final integration steps
        d = abs(self.t_bound - t)               # remaining interval
        if d < 2 * h_abs:
            if d >= min_step:
                if h_abs < d:
                    # h_abs < d < 2 * h_abs:
                    # split d over last two steps ("look ahead").
                    # This reduces the chance of a very small last step.
                    h_abs = max(0.5 * d, min_step)
                else:
                    # d <= h_abs:
                    # don't step over t_bound
                    h_abs = d
            else:
                # d < min_step:
                # use linear extrapolation in this rare case
                h = self.t_bound - t
                y_new = y + h * self.f
                self.h_previous = h
                self.y_old = y
                self.t = self.t_bound
                self.y = y_new
                self.f = None                    # to signal _dense_output_impl
                warn('\nLinear extrapolation was used in the final step.')
                return True, None

        # loop until step accepted
        step_accepted = False
        step_rejected = False
        while not step_accepted:

            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            # calculate stages, except last two
            self.K[0] = self.f
            for i in range(1, self.n_stages - 1):
                self._rk_stage(h, i)

            # calculate pre_error_norm
            error_norm_pre = self._estimate_error_norm_pre(y, h)

            # reject step if pre_error too large
            if error_norm_pre > 1:
                step_rejected = True
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm_pre ** self.error_exponent)
                NFS[()] += 1
                continue

            # calculate next stage
            self._rk_stage(h, self.n_stages - 1)

            # calculate error_norm and solution
            y_new, error_norm = self._comp_sol_err(y, h)

            # and evaluate
            if error_norm < 1:
                step_accepted = True
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)
                if step_rejected:
                    factor = min(1, factor)
                h_abs *= factor

            else:
                step_rejected = True
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** self.error_exponent)
                NFS[()] += 1

        # store for next step and interpolation
        self.h_previous = h
        self.y_old = y
        self.h_abs = h_abs
        self.f = self.K[self.n_stages].copy()

        # output
        self.t = t_new
        self.y = y_new

        # stiffness detection
        if self.maxfcn:
            self.havg = 0.9 * self.havg + 0.1 * h     # exp moving average
            self._stiff()
            self.okstp += 1
            if self.okstp == 20:
                self.havg = h
                self.jflstp = 0

        return True, None

    def _estimate_error_norm_pre(self, y, h):
        # first error estimate
        # y_new is not available yet for scale, so use y_old instead
        scale = self.atol + self.rtol * 0.5*(np.abs(y) + np.abs(self.y_old))
        err = h * (self.K[:6, :].T @ self.E_pre)
        return norm(err / scale)

    def _dense_output_impl(self):
        h = self.h_previous
        K = self.K_extended

        # calculate the required extra stages
        for s, (a, c) in enumerate(zip(self.A_extra, self.C_extra),
                                   start=self.n_stages+1):
            dy = K[:s, :].T @ a[:s] * h
            K[s] = self.fun(self.t_old + c * h, self.y_old + dy)

        # form Q. Usually: Q = K.T @ self.P
        # but rksuite recommends to group summations to mitigate round-off:
        Q = np.empty((K.shape[1], self.P.shape[1]), dtype=K.dtype)
        Q[:, 0] = self.K[7]
        KP = K * self.P[:, 1, np.newaxis]                       # term for t**2
        Q[:, 1] = (KP[4] + ((KP[5] + KP[7]) + KP[0]) + ((KP[2] + KP[8]) +
                   KP[9]) + ((KP[3] + KP[10]) + KP[6]))
        KP = K * self.P[:, 2, np.newaxis]                       # term for t**3
        Q[:, 2] = (KP[4] + KP[5] + ((KP[2] + KP[8]) + (KP[9] + KP[7]) +
                   KP[0]) + ((KP[3] + KP[10]) + KP[6]))
        KP = K * self.P[:, 3, np.newaxis]                       # term for t**4
        Q[:, 3] = (((KP[3] + KP[7]) + (KP[6] + KP[5]) + KP[4]) + ((KP[9] +
                   KP[8]) + (KP[2]+KP[10]) + KP[0]))
        KP = K * self.P[:, 4, np.newaxis]                       # term for t**5
        Q[:, 4] = ((KP[9] + KP[8]) + ((KP[6] + KP[5]) + KP[4]) + ((KP[3] +
                   KP[7]) + (KP[2] + KP[10]) + KP[0]))
        KP = K * self.P[:, 5, np.newaxis]                       # term for t**6
        Q[:, 5] = (KP[4] + ((KP[9] + KP[7]) + (KP[6] + KP[5])) + ((KP[3] +
                   KP[8]) + (KP[2] + KP[10]) + KP[0]))

        # Rksuite uses Horner's rule to evaluate the polynomial. Moreover,
        # the polynomial definition is different: looking back from the end
        # of the step instead of forward from the start.
        # The call is modified to accomodate:
        return HornerDenseOutput(self.t, self.t+h, self.y, Q)


class BS45_i(BS45):
    """As BS45, but with free 4th order interpolant for dense output. Suffix _i
    for interpolant.

    The source [1]_ refers to the thesis of Bogacki for a free interpolant, but
    this could not be found. Instead, the interpolant is constructed following
    the steps in [3]_.

    For the best accuracy use BS45 instead. This BS45_i method can be more
    efficient for several problems if dense_output is used.

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

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian. Is always 0 for this solver as
        it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine, "An efficient Runge-Kutta (4,5) pair",
           Computers & Mathematics with Applications, Vol. 32, No. 6, 1996,
           pp. 15-28, ISSN 0898-1221.
           https://doi.org/10.1016/0898-1221(96)00141-1
    .. [2] RKSUITE: https://www.netlib.org/ode/rksuite/
    .. [3] Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the
           first column simplifying assumption", Computers & Mathematics with
           Applications, Vol. 62, No. 2, pp. 770 - 775, 2011.
           https://doi.org/10.1016/j.camwa.2011.06.002
    """

    # Bogacki published a free interpolant in his thesis, but I was not able to
    # find a copy of it. Instead, I constructed an interpolant using sympy and
    # the approach in [3]_.
    # This free 4th order interpolant has a leading error term ||T5|| that has
    # maximum in [0,1] of 5.47 e-4. This is higher than the corresponding term
    # of the embedded fourth order method: 1.06e-4.

    # overwrite P
    P = np.array([
        [1, -2773674729811/735370896960,
            316222661411/52526492640,
            -1282818361681/294148358784,
            6918746667/5836276960],
        [0, 0, 0, 0, 0],
        [0, 1594012432639617/282545840187520,
            -303081611134977/20181845727680,
            1643668176796011/113018336075008,
            -14071997888919/2883120818240],
        [0, -47637453654133/20485332129600,
            125365109861131/10242666064800,
            -135424370922463/8194132851840,
            2582696138393/379358002400],
        [0, 1915795112337/817078774400,
            -557453242737/58362769600,
            3958638678747/326831509760,
            -285784868817/58362769600],
        [0, -1490252641456/654939705105,
            692325952352/93562815015,
            -808867306376/130987941021,
            4887837472/3465289445],
        [0, 824349534931/571955142080,
            -895925604353/122561816160,
            2443928282393/228782056832,
            -5528580993/1167255392],
        [0, -38480331/36476731,
            226874786/36476731,
            -374785310/36476731,
            186390855/36476731]])

    def __init__(self, fun, t0, y0, t_bound, **extraneous):
        super(BS45, self).__init__(fun, t0, y0, t_bound, **extraneous)
        # y_old is used for first error assessment, it should not be None
        self.y_old = self.y - self.direction * self.h_abs * self.f

    def _dense_output_impl(self):
        return super(BS45, self)._dense_output_impl()
