import numpy as np
from extensisq.common import (norm, MAX_FACTOR, MAX_FACTOR_SWITCH, RungeKutta,
                              HornerDenseOutput, NFS, calculate_scale)


class BS5(RungeKutta):
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
    sc_params : tuple of size 4, "standard", "G", "H" or "W", optional
        Parameters for the stepsize controller (k*b1, k*b2, a2, g). The step
        size controller is, with k the exponent of the standard controller,
        _n for new and _o for old:
            h_n = h * g**(k*b1 + k*b2) * (h/h_o)**-a2
                * (err/tol)**-b1 * (err_o/tol_o)**-b2
        Predefined parameters are:
            Gustafsson "G" (0.7, -0.4, 0, 0.9),  Watts "W" (2, -1, -1, 0.8),
            Soederlind "S" (0.6, -0.2, 0, 0.9),  and "standard" (1, 0, 0, 0.9).
        The default for this method is "W".
    interpolant : 'best', 'low' or 'free', optional
        Select the interpolant for dense output. The option 'best' is for the
        accurate fifth order interpolant described in [1], which needs 3 extra
        function evaluations per step. The option 'low' is for a less accurate
        fifth order interpolant which needs 'only' one extra function
        evaluation per step. 'free' is for a free fourth order interpolant.
        Recommendations: 'best' for events, 'free' for simple plotting or for
        long integrations for which the gobal error dominates. The accuracy of
        the method itself does not change and the extra function evaluations
        are only done at steps for which dense output is requested. The default
        is 'low': a safe option for which the method does not loose much of its
        performance when it is used with dense output.

    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine, "An efficient Runge-Kutta (4,5) pair",
           Computers & Mathematics with Applications, Vol. 32, No. 6, 1996,
           pp. 15-28.
           https://doi.org/10.1016/0898-1221(96)00141-1
    .. [2] RKSUITE: https://www.netlib.org/ode/rksuite/
    """

    order = 5
    error_estimator_order = 4
    n_stages = 7            # the effective nr (total nr of stages is 8)
    n_extra_stages = 3      # for dense output
    tanang = 5.2
    stbrad = 3.9
    sc_params = "W"

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
    # this is actually the main error estimator
    E_pre = np.array([-3/1280, 0, 6561/632320, -343/20800, 243/12800, -1/95])

    # coefficients for post error estimation method
    # this can account for sudden changes above c=3/4, which the main error
    # estimate cannot.
    E = np.array([2479/34992, 0, 123/416, 612941/3411720, 43/1440,
                  2272/6561, 79937/1113912, 3293/556956])
    E[:-1] -= B     # convert to error coefficients

    # extra time step fractions for dense output
    C_extra = np.array([1/2, 5/6, 1/9])

    # coefficient matrix for dense output
    A_extra = np.array([
        [455/6144, -837888343715/13176988637184, 98719073263/1551965184000],
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

    # This is the original, very accurate, 5th order interpolant of [1, 2].
    # It needs 3 extra function evaluations per step.
    Pbest = np.array([
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

    # This is an up to 12x less accurate, 5th order, low cost interpolant that
    # needs one extra function evaluation per step, instead of three.
    Plow = np.array([
        [1, -155/36, 16441/2016, -56689/8064, 757/336],
        [0, 0, 0, 0, 0],
        [0, 6561/988, -14727987/774592, 60538347/3098368, -13321017/1936480],
        [0, 2401/702, -603337/56160, 2740913/224640, -24353/5200],
        [0, 0, -387/2240, 3483/8960, -1161/5600],
        [0, 1408/513, -45536/3591, 67960/3591, -17216/1995],
        [0, 0, -7267/4704, 21801/6272, -7267/3920],
        [0, -1/2, 4, -15/2, 4],
        [0, -8, 32, -40, 16]])

    # This is a low accuracy, 4th order interpolant. It is free: it does not
    # need extra function evaluations.
    P = np.array([
        [1, -2773674729811/735370896960, 316222661411/52526492640,
            -1282818361681/294148358784, 6918746667/5836276960],
        [0, 0, 0, 0, 0],
        [0, 1594012432639617/282545840187520, -303081611134977/20181845727680,
            1643668176796011/113018336075008, -14071997888919/2883120818240],
        [0, -47637453654133/20485332129600, 125365109861131/10242666064800,
            -135424370922463/8194132851840, 2582696138393/379358002400],
        [0, 1915795112337/817078774400, -557453242737/58362769600,
            3958638678747/326831509760, -285784868817/58362769600],
        [0, -1490252641456/654939705105, 692325952352/93562815015,
            -808867306376/130987941021, 4887837472/3465289445],
        [0, 824349534931/571955142080, -895925604353/122561816160,
            2443928282393/228782056832, -5528580993/1167255392],
        [0, -38480331/36476731, 226874786/36476731,
            -374785310/36476731, 186390855/36476731]])

    def __init__(self, fun, t0, y0, t_bound, nfev_stiff_detect=5000,
                 sc_params='standard', interpolant='low', **extraneous):
        super(BS5, self).__init__(
            fun, t0, y0, t_bound, nfev_stiff_detect=nfev_stiff_detect,
            sc_params=sc_params, **extraneous)
        # custom initialization to create extended storage for dense output
        if interpolant not in ('best', 'low', 'free'):
            raise ValueError(
                "interpolant should be one of: 'best', 'low', 'free'")
        self.interpolant = interpolant
        if self.interpolant == 'best':
            self.K_extended = np.zeros((self.n_stages+self.n_extra_stages + 1,
                                        self.n), dtype=self.y.dtype)
            self.K = self.K_extended[:self.n_stages+1]
        elif self.interpolant == 'low':
            self.K_extended = np.zeros((self.n_stages + 2,
                                        self.n), dtype=self.y.dtype)
            self.K = self.K_extended[:self.n_stages+1]
        else:
            self.K_extended = self.K

    def _step_impl(self):

        # mostly follows the scipy implementation of RungeKutta
        t = self.t
        y = self.y

        h_abs, min_step = self._reassess_stepsize(t, y)

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
                h_abs *= max(
                    self.MIN_FACTOR,
                    self.safety * error_norm_pre ** self.error_exponent)

                NFS[()] += 1
                if self.nfev_stiff_detect:
                    self.jflstp += 1                  # for stiffness detection
                continue

            # calculate next stage
            self._rk_stage(h, self.n_stages - 1)

            # calculate error_norm and solution
            y_new, error_norm = self._comp_sol_err(y, h)

            # and evaluate
            if error_norm < 1:
                step_accepted = True

                if error_norm == 0.:
                    factor = self.MAX_FACTOR

                elif self.standard_sc:
                    factor = self.safety * error_norm ** self.error_exponent
                    self.standard_sc = False

                else:
                    # use second order SC controller
                    h_ratio = h / self.h_previous
                    factor = self.safety_sc * (
                        error_norm ** self.minbeta1 *
                        self.error_norm_old ** self.minbeta2 *
                        h_ratio ** self.minalpha)
                    factor = min(self.MAX_FACTOR, max(self.MIN_FACTOR, factor))

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                if factor < MAX_FACTOR_SWITCH:
                    # reduce MAX_FACTOR when on scale.
                    self.MAX_FACTOR = MAX_FACTOR

            else:
                if np.isnan(error_norm) or np.isinf(error_norm):
                    return False, "Overflow or underflow encountered."

                step_rejected = True
                h_abs *= max(self.MIN_FACTOR,
                             self.safety * error_norm ** self.error_exponent)

                NFS[()] += 1
                self.jflstp += 1                      # for stiffness detection

        # store for next step and interpolation
        self.h_previous = h
        self.y_old = y
        self.h_abs = h_abs
        self.f = self.K[self.n_stages].copy()
        self.error_norm_old = error_norm

        # output
        self.t = t_new
        self.y = y_new

        # stiffness detection
        self._diagnose_stiffness()

        return True, None

    def _estimate_error_norm_pre(self, y, h):
        # first error estimate
        # y_new is not available yet for scale, so use y_pre instead
        y_pre = y + h * (self.K[:6].T @ self.A[6, :6])
        scale = calculate_scale(self.atol, self.rtol, y, y_pre)
        err = h * (self.K[:6, :].T @ self.E_pre)
        return norm(err / scale)

    def _dense_output_impl(self):
        h = self.h_previous
        K = self.K_extended

        if self.interpolant == 'free':
            Q = K.T @ self.P
            return HornerDenseOutput(self.t_old, self.t, self.y_old, Q)

        elif self.interpolant == 'low':
            s = self.n_stages + 1
            dy = K[:s, :].T @ self.A_extra[0, :s] * h
            K[s] = self.fun(self.t_old + self.C_extra[0] * h, self.y_old + dy)
            Q = K.T @ self.Plow
            return HornerDenseOutput(self.t_old, self.t, self.y_old, Q)

        # else: the accurate interpolant
        # calculate the required extra stages
        for s, (a, c) in enumerate(zip(self.A_extra, self.C_extra),
                                   start=self.n_stages+1):
            dy = K[:s, :].T @ a[:s] * h
            K[s] = self.fun(self.t_old + c * h, self.y_old + dy)

        # form Q. Usually: Q = K.T @ self.P as for the other interpolants,
        # but RKSuite groups summations to mitigate round-off:
        Q = np.empty((K.shape[1], self.Pbest.shape[1]), dtype=K.dtype)
        Q[:, 0] = self.K[7]
        KP = K * self.Pbest[:, 1, np.newaxis]                   # term for t**2
        Q[:, 1] = (KP[4] + ((KP[5] + KP[7]) + KP[0]) + ((KP[2] + KP[8]) +
                   KP[9]) + ((KP[3] + KP[10]) + KP[6]))
        KP = K * self.Pbest[:, 2, np.newaxis]                   # term for t**3
        Q[:, 2] = (KP[4] + KP[5] + ((KP[2] + KP[8]) + (KP[9] + KP[7]) +
                   KP[0]) + ((KP[3] + KP[10]) + KP[6]))
        KP = K * self.Pbest[:, 3, np.newaxis]                   # term for t**4
        Q[:, 3] = (((KP[3] + KP[7]) + (KP[6] + KP[5]) + KP[4]) + ((KP[9] +
                   KP[8]) + (KP[2]+KP[10]) + KP[0]))
        KP = K * self.Pbest[:, 4, np.newaxis]                   # term for t**5
        Q[:, 4] = ((KP[9] + KP[8]) + ((KP[6] + KP[5]) + KP[4]) + ((KP[3] +
                   KP[7]) + (KP[2] + KP[10]) + KP[0]))
        KP = K * self.Pbest[:, 5, np.newaxis]                   # term for t**6
        Q[:, 5] = (KP[4] + ((KP[9] + KP[7]) + (KP[6] + KP[5])) + ((KP[3] +
                   KP[8]) + (KP[2] + KP[10]) + KP[0]))

        # RKSuite's, polynomial definition is different from scipy's: looking
        # back from the end of the step instead of forward from the start.
        # The call is modified to accomodate:
        return HornerDenseOutput(self.t, self.t+h, self.y, Q)
