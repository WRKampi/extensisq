import numpy as np
from extensisq.common import (
    RungeKutta, NFS, norm, MAX_FACTOR, MAX_FACTOR_SWITCH, calculate_scale)


class CFMR7osc(RungeKutta):
    """Explicit Runge-Kutta method of (algebraic) order 7, with an error
    estimate of order 5 and a free interpolant of (algebraic) order 5.

    This method by Calvo, Franco, Montijano and Randez is tuned to get a
    dispersion order of 10 and a dissipation order of 9. This is beneficial
    for problems with oscillating solutions (and linear problems in general).
    It can outperform methods of higher algebraic order, such as `DOP853`,
    for such problems.

    The interpolant that has been added is of dispersion order 6 and
    dissipation order 7.

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
        The default for this method is "S".

    References
    ----------
    .. [1] M. Calvo, J.M. Franco, J.I. Montijano, L. Randez, "Explicit
           Runge-Kutta methods for initial value problems with oscillating
           solutions", Journal of Computational and Applied Mathematics, Vol.
           76, No. 1–2, 1996, pp. 195-212.
           https://doi.org/10.1016/S0377-0427(96)00103-3
    """
    order = 7
    error_estimator_order = 5
    n_stages = 9
    tanang = 40
    stbrad = 4.7
    sc_params = "S"

    # time step fractions
    C = np.array([0, 4/63, 2/21, 1/7, 7/17, 13/24, 7/9, 91/100, 1])

    # coefficient matrix
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4/63, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/42, 1/14, 0, 0, 0, 0, 0, 0, 0],
        [1/28, 0, 3/28, 0, 0, 0, 0, 0, 0],
        [12551/19652, 0, -48363/19652, 10976/4913, 0, 0, 0, 0, 0],
        [-36616931/27869184, 0, 2370277/442368, -255519173/63700992,
         226798819/445906944, 0, 0, 0, 0],
        [-10401401/7164612, 0, 47383/8748, -4914455/1318761, -1498465/7302393,
         2785280/3739203, 0, 0, 0],
        [181002080831/17500000000, 0, -14827049601/400000000,
         23296401527134463/857600000000000, 2937811552328081/949760000000000,
         -243874470411/69355468750, 2857867601589/3200000000000, 0, 0],
        [-228380759/19257212, 0, 4828803/113948, -331062132205/10932626912,
         -12727101935/3720174304, 22627205314560/4940625496417,
         -268403949/461033608, 3600000000000/19176750553961, 0]])

    # coefficients for propagating method
    B = np.array([
        95/2366, 0, 0, 3822231133/16579123200, 555164087/2298419200,
        1279328256/9538891505, 5963949/25894400, 50000000000/599799373173,
        28487/712800])

    # coefficients for error estimation, note E[-2] = 0.
    E = np.array([
        1689248233/50104356120, 0, 0, 1/4, 28320758959727/152103780259200,
        66180849792/341834007515, 31163653341/152322513280,
        36241511875000/394222326561063, 28487/712800, 0])
    E[:-1] -= B

    # coefficients for interpolation (dense output)
    # free 5th order interpolant, Optimal T620
    P = np.array([
        [1, -6248/1183, 12056/1183, -1345/182, 97/169, 160/169],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 661103345/110527488, -12471420661/828956160,
         11535693337/1105274880, 2373818279/1381593600, -823543/287832],
        [0, 39338391/20894720, -975441759/114920960, 9504105153/459683840,
         -1803301911/82086400, 417605/51304],
        [0, -7599771648/1907778301, 41855533056/1907778301,
         -5870997504/146752177, 295068082176/9538891505,
         -1517322240/173434391],
        [0, 684531/517888, -11632653/1294720, 14012109/739840,
         -13714677/924800, 2187/578],
        [0, 16000000000/18175738581, -2410000000000/599799373173,
         10000000000/2197067301, -2000000000/28561874913,
         -4000000000/3173541657],
        [0, 28487/23760, -199409/35640, 370331/47520, -199409/59400, 0],
        [0, -2, 10, -15, 7, 0]])

    # redefine _step_impl() to save 1 evaluation for each rejected step
    def _step_impl(self):
        # mostly follows the scipy implementation of scipy's RungeKutta
        t = self.t
        y = self.y

        h_abs, min_step = self._reassess_stepsize(t, y)
        if h_abs is None:
            # linear extrapolation for last step
            return True, None

        # loop until the step is accepted
        step_accepted = False
        step_rejected = False
        while not step_accepted:

            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h

            # calculate stages needed for error evaluation
            self.K[0] = self.f
            for i in range(1, self.n_stages-1):
                self._rk_stage(h, i)

            # evaluate error with premature y_pre instead of y_new for weight
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

            # calculate last stage needed for output
            self._rk_stage(h, self.n_stages-1)

            # calculate error norm and solution (now with proper weight)
            y_new, error_norm = self._comp_sol_err(y, h)

            # evaluate error
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
                step_rejected = True
                h_abs *= max(self.MIN_FACTOR,
                             self.safety * error_norm ** self.error_exponent)

                NFS[()] += 1
                self.jflstp += 1                      # for stiffness detection

                if np.isnan(error_norm) or np.isinf(error_norm):
                    return False, "Overflow or underflow encountered."

        # evaluate ouput point for interpolation and next step
        self.K[self.n_stages] = self.fun(t + h, y_new)

        # store for next step, interpolation and stepsize control
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
        y_pre = y + h * (self.K[:8].T @ self.A[8, :8])
        scale = calculate_scale(self.atol, self.rtol, y, y_pre)
        err = h * (self.K[:8, :].T @ self.E[:8])
        return norm(err / scale)
