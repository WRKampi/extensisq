import numpy as np
from math import sqrt, copysign
from warnings import warn
import logging
from scipy.integrate._ivp.common import (
    validate_max_step, validate_first_step, warn_extraneous)
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
# from scipy.integrate._ivp.rk import MIN_FACTOR, MAX_FACTOR


MIN_FACTOR = 0.5
MAX_FACTOR = 2.0
MAX_FACTOR_SWITCH = 4.0
MAX_FACTOR0 = 8.0
NFS = np.array(0)                                         # failed step counter


def validate_tol(rtol, atol, y):
    """Validate tolerance values. atol can be scalar or array-like, rtol a
    scalar. The bound values are from RKSuite. These differ from those in
    scipy. Bounds are applied without warning.
    """
    atol = np.asarray(atol)
    if atol.ndim > 0 and atol.shape != (y.size, ):
        raise ValueError("`atol` has wrong shape.")
    if np.any(atol < 0):
        raise ValueError("`atol` must be positive.")
    if not isinstance(rtol, float):
        raise ValueError("`rtol` must be a float.")
    if rtol < 0:
        raise ValueError("`rtol` must be positive.")

    # atol cannot be exactly zero.
    # For double precision float: sqrt(tiny) ~ 1.5e-154
    tiny = np.finfo(y.dtype).tiny
    atol = np.maximum(atol, sqrt(tiny))

    # rtol is bounded from both sides.
    # The lower bound is lower than in scipy.
    epsneg = np.finfo(y.dtype).epsneg
    rtol = np.minimum(np.maximum(rtol, 10 * epsneg), 0.01)
    return rtol, atol


def calculate_scale(atol, rtol, y, y_new):
    """calculate a scaling vector for the error estimate"""
    return atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
    # the other popular option is:
    # return atol + rtol * 0.5*(np.abs(y) + np.abs(y_new))


def norm(x):
    """Compute RMS norm."""
    return (np.real(x @ x.conjugate()) / x.size) ** 0.5


class RungeKutta(OdeSolver):
    """Base class for explicit runge kutta methods.

    This implementation mainly follows the scipy implementation. The current
    differences are:
      - Conventional (non FSAL) methods are detected and failed steps cost
        one function evaluation less than with the scipy implementation.
      - A different, more elaborate estimate for the size of the first step
        is used.
      - Horner's rule is used for dense output calculation.
      - A failed step counter is added.
      - The stepsize near the end of the integration is different:
        - look ahead to prevent too small step sizes
      - the min_step accounts for the distance between C-values
      - a different tolerance validation is used.
      - stiffness detection is added, can be turned off
      - second order stepsize control is added.
    """

    # effective number of stages
    n_stages: int = NotImplemented

    # order of the main method
    order: int = NotImplemented

    # order of the secondary embedded method
    error_estimator_order: int = NotImplemented

    # runge kutta coefficient matrix
    A: np.ndarray = NotImplemented              # shape: [n_stages, n_stages]

    # output coefficients (weights)
    B: np.ndarray = NotImplemented              # shape: [n_stages]

    # time fraction coefficients (nodes)
    C: np.ndarray = NotImplemented              # shape: [n_stages]

    # error coefficients (weights Bh - B); for non-FSAL methods E[-1] == 0.
    E: np.ndarray = NotImplemented              # shape: [n_stages + 1]

    # dense output interpolation coefficients, optional
    P: np.ndarray = NotImplemented              # shape: [n_stages + 1,
    #                                                     order_polynomial]

    # Parameters for stiffness detection, optional
    stbrad: float = NotImplemented              # radius of the arc
    tanang: float = NotImplemented              # tan(valid angle < pi/2)

    # Parameters for stepsize control, optional
    sc_params = "standard"                      # tuple, or str

    MAX_FACTOR = MAX_FACTOR0                    # initially
    MIN_FACTOR = MIN_FACTOR

    def _init_min_step_parameters(self):
        """Define the parameters h_min_a and h_min_b for the min_step rule:
            min_step = max(h_min_a * abs(t), h_min_b)
        from RKSuite.
        """

        # minimum difference between distinct C-values
        cdiff = 1.
        for c1 in self.C:
            for c2 in self.C:
                diff = abs(c1 - c2)
                if diff:
                    cdiff = min(cdiff, diff)
        if cdiff < 1e-3:
            cdiff = 1e-3
            logging.warning(
                'Some C-values of this Runge Kutta method are nearly the '
                'same but not identical. This limits the minimum stepsize'
                'You may want to check the implementation of this method.')

        # determine min_step parameters
        epsneg = np.finfo(self.y.dtype).epsneg
        tiny = np.finfo(self.y.dtype).tiny
        h_min_a = 10 * epsneg / cdiff
        h_min_b = sqrt(tiny)
        return h_min_a, h_min_b

    def _init_stiffness_detection(self, nfev_stiff_detect):
        if not (isinstance(nfev_stiff_detect, int) and nfev_stiff_detect >= 0):
            raise ValueError(
                "`nfev_stiff_detect` must be a non-negative integer.")
        self.nfev_stiff_detect = nfev_stiff_detect
        if NotImplemented in (self.stbrad, self.tanang):
            # disable stiffness detection if not implemented
            if nfev_stiff_detect not in (5000, 0):
                warn("This method does not implement stiffness detection. "
                     "Changing the value of nfev_stiff_detect does nothing.")
            self.nfev_stiff_detect = 0
        self.jflstp = 0                         # failed step counter, last 40
        if self.nfev_stiff_detect:
            self.okstp = 0                      # successful step counter
            self.havg = 0.0                     # average stepsize

    def _init_sc_control(self, sc_params):
        coefs = {"G": (0.7, -0.4, 0, 0.9),
                 "S": (0.6, -0.2, 0, 0.9),
                 "W": (2, -1, -1, 0.8),
                 "standard": (1, 0, 0, 0.9)}
        # use default controller of method if not specified otherwise
        sc_params = sc_params or self.sc_params
        if (isinstance(sc_params, str) and sc_params in coefs):
            kb1, kb2, a, g = coefs[sc_params]
        elif isinstance(sc_params, tuple) and len(sc_params) == 4:
            kb1, kb2, a, g = sc_params
        else:
            raise ValueError('sc_params should be a tuple of length 3 or one '
                             'of the strings "G", "S", "W" or "standard"')
        # set all parameters
        self.minbeta1 = kb1 * self.error_exponent
        self.minbeta2 = kb2 * self.error_exponent
        self.minalpha = -a
        self.safety = g
        self.safety_sc = g ** (kb1 + kb2)
        self.standard_sc = True                                # for first step

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf, rtol=1e-3,
                 atol=1e-6, vectorized=False, first_step=None,
                 nfev_stiff_detect=5000, sc_params=None, **extraneous):
        warn_extraneous(extraneous)
        super(RungeKutta, self).__init__(fun, t0, y0, t_bound, vectorized,
                                         support_complex=True)
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.y)
        self.f = self.fun(self.t, self.y)
        if self.f.dtype != self.y.dtype:
            raise TypeError('dtypes of solution and derivative do not match')
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self._init_stiffness_detection(nfev_stiff_detect)
        self.h_min_a, self.h_min_b = self._init_min_step_parameters()
        self._init_sc_control(sc_params)

        # size of first step:
        if first_step is None:
            b = self.t + self.direction * min(
                abs(self.t_bound - self.t), self.max_step)
            self.h_abs = abs(h_start(
                self.fun, self.t, b, self.y, self.f,
                self.error_estimator_order, self.rtol, self.atol))
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)

        self.K = np.empty((self.n_stages + 1, self.n), self.y.dtype)
        self.FSAL = 1 if self.E[self.n_stages] else 0
        self.h_previous = None
        self.y_old = None
        NFS[()] = 0                                # global failed step counter

    def _step_impl(self):
        # mostly follows the scipy implementation of scipy's RungeKutta
        t = self.t
        y = self.y

        h_abs, min_step = self._reassess_stepsize(t, y)

        # loop until the step is accepted
        step_accepted = False
        step_rejected = False
        while not step_accepted:

            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h

            # calculate stages needed for output
            self.K[0] = self.f
            for i in range(1, self.n_stages):
                self._rk_stage(h, i)

            # calculate error norm and solution
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

        if not self.FSAL:
            # evaluate ouput point for interpolation and next step
            self.K[self.n_stages] = self.fun(t + h, y_new)

        # store for next step, interpolation and stepsize control
        self.h_previous = h
        self.y_old = y
        self.h_abs = h_abs
        self.f_old = self.f
        self.f = self.K[self.n_stages].copy()
        self.error_norm_old = error_norm

        # output
        self.t = t_new
        self.y = y_new

        # stiffness detection
        self._diagnose_stiffness()

        return True, None

    def _reassess_stepsize(self, t, y):
        # limit step size
        h_abs = self.h_abs
        min_step = max(self.h_min_a * (abs(t) + h_abs), self.h_min_b)
        if h_abs < min_step or h_abs > self.max_step:
            h_abs = min(self.max_step, max(min_step, h_abs))
            self.standard_sc = True

        # handle final integration steps
        d = abs(self.t_bound - t)                     # remaining interval
        if d < 2 * h_abs:
            if d > h_abs:
                # h_abs < d < 2 * h_abs: "look ahead".
                # split d over last two steps. This reduces the chance of a
                # very small last step.
                h_abs = max(0.5 * d, min_step)
                self.standard_sc = True
            else:
                # d <= h_abs: Don't step over t_bound
                h_abs = d

        return h_abs, min_step

    def _estimate_error(self, K, h):
        # exclude K[-1] if not FSAL. It could contain nan or inf
        return h * (K[:self.n_stages + self.FSAL].T @
                    self.E[:self.n_stages + self.FSAL])

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _comp_sol_err(self, y, h):
        """Compute solution and error.
        The calculation of `scale` differs from scipy: The average instead of
        the maximum of abs(y) of the current and previous steps is used.
        """
        y_new = y + h * (self.K[:self.n_stages].T @ self.B)
        scale = calculate_scale(self.atol, self.rtol, y, y_new)

        if self.FSAL:
            # do FSAL evaluation if needed for error estimate
            self.K[self.n_stages, :] = self.fun(self.t + h, y_new)

        error_norm = self._estimate_error_norm(self.K, h, scale)
        return y_new, error_norm

    def _rk_stage(self, h, i):
        """compute a single RK stage"""
        dy = h * (self.K[:i, :].T @ self.A[i, :i])
        self.K[i] = self.fun(self.t + self.C[i] * h, self.y + dy)

    def _dense_output_impl(self):
        """return denseOutput, detect if step was extrapolated linearly"""

        if isinstance(self.P, np.ndarray):
            # normal output
            Q = self.K.T @ self.P
            return HornerDenseOutput(self.t_old, self.t, self.y_old, Q)

        # if no interpolant is implemented
        return CubicDenseOutput(self.t_old, self.t, self.y_old, self.y,
                                self.f_old, self.f)

    def _diagnose_stiffness(self):
        """Stiffness detection.

        Test only if there are many recent step failures, or after many
        function evaluations have been done.

        Warn the user if the problem is diagnosed as stiff.

        Original source: RKSuite.f, https://www.netlib.org/ode/rksuite/
        """

        if self.nfev_stiff_detect == 0:
            return

        self.okstp += 1
        h = self.h_previous
        self.havg = 0.9 * self.havg + 0.1 * h              # exp moving average

        # reset after the first 20 steps to:
        # - get stepsize on scale
        # - reduce the effect of a possible initial transient
        if self.okstp == 20:
            self.havg = h
            self.jflstp = 0

        # There are lots of failed steps (lotsfl = True) if 10 or more step
        # failures occurred in the last 40 successful steps.
        if self.okstp % 40 == 39:
            lotsfl = self.jflstp >= 10
            self.jflstp = 0                               # reset each 40 steps
        else:
            lotsfl = False

        # Test for stifness after each nfev_stiff_detect evaluations
        # then toomch = True
        many_steps = self.nfev_stiff_detect//self.n_stages
        toomch = self.okstp % many_steps == many_steps - 1

        # If either too much work has been done or there are lots of failed
        # steps, test for stiffness.
        if toomch or lotsfl:

            # Regenerate weight vector
            avgy = 0.5 * (np.abs(self.y) + np.abs(self.y_old))
            tiny = np.finfo(self.y.dtype).tiny
            wt = np.maximum(avgy, sqrt(tiny))
            # and error vector, wich is a good initial perturbation vector
            v0 = np.atleast_1d(self._estimate_error(self.K, self.h_previous))

            # stiff_a determines whether the problem is stiff. In some
            # circumstances it is UNSURE.  The decision depends on two things:
            # whether the step size is being restricted on grounds of stability
            # and whether the integration to t_bound can be completed in no
            # more than nfev_stiff_detect function evaluations.
            stif, rootre = stiff_a(
                self.fun, self.t, self.y, self.h_previous, self.havg,
                self.t_bound, self.nfev_stiff_detect, wt, self.f, v0,
                self.n_stages, self.stbrad, self.tanang)

            # inform the user about stiffness with warning messages
            # the messages about remaining work have been removed from the
            # original code.
            if stif is None:
                # unsure about stiffness
                if rootre is None:
                    # no warning is given
                    logging.info('Stiffness detection did not converge')
                if not rootre:
                    # A complex pair of roots has been found near the imaginary
                    # axis, where the stability boundary of the method is not
                    # well defined.
                    # A warning is given only if there are many failed steps.
                    # This reduces the chance of a false positive diagnosis.
                    if lotsfl:
                        warn('Your problem has a complex pair of dominant '
                             'roots near the imaginary axis.  There are '
                             'many recently failed steps.  You should '
                             'probably change to a code intended for '
                             'oscillatory problems.')
                    else:
                        logging.info(
                            'The problem has a complex pair of dominant roots '
                            'near the imaginary axis.  There are not many '
                            'failed steps.')
                else:
                    # this should not happen
                    logging.warning(
                        'stif=None, rootre=True; this should not happen')
            elif stif:
                # the problem is stiff
                if rootre is None:
                    # this should not happen
                    logging.warning(
                        'stif=True, rootre=None; this should not happen')
                elif rootre:
                    warn('Your problem has a real dominant root '
                         'and is diagnosed as stiff.  You should probably '
                         'change to a code intended for stiff problems.')
                else:
                    warn('Your problem has a complex pair of dominant roots '
                         'and is diagnosed as stiff.  You should probably '
                         'change to a code intended for stiff problems.')
            else:
                # stif == False
                # no warning is given
                if rootre is None:
                    logging.info(
                        'Stiffness detection has diagnosed the problem as '
                        'non-stiff, without performing power iterations')
                elif rootre:
                    logging.info(
                        'The problem has a real dominant root '
                        'and is not stiff')
                else:
                    logging.info(
                        'The problem has a complex pair of dominant roots '
                        'and is not stiff')


def h_start(df, a, b, y, yprime, morder, rtol, atol):
    """h_shart computes a starting step size to be used in solving initial
    value problems in ordinary differential equations.

    This method is developed by H.A. Watts and described in [1]_. This function
    is a Python translation of the Fortran source code [2]_. The two main
    modifications are:
        using the RMS norm from scipy.integrate
        allowing for complex valued input

    Parameters
    ----------
    df : callable
        Right-hand side of the system. The calling signature is fun(t, y).
        Here t is a scalar. The ndarray y has has shape (n,) and fun must
        return array_like with the same shape (n,).
    a : float
        This is the initial point of integration.
    b : float
        This is a value of the independent variable used to define the
        direction of integration. A reasonable choice is to set `b` to the
        first point at which a solution is desired. You can also use `b , if
        necessary, to restrict the length of the first integration step because
        the algorithm will not compute a starting step length which is bigger
        than abs(b-a), unless `b` has been chosen too close to `a`. (it is
        presumed that h_start has been called with `b` different from `a` on
        the machine being used.
    y : array_like, shape (n,)
        This is the vector of initial values of the n solution components at
        the initial point `a`.
    yprime : array_like, shape (n,)
        This is the vector of derivatives of the n solution components at the
        initial point `a`.  (defined by the differential equations in
        subroutine `df`)
    morder : int
        This is the order of the formula which will be used by the initial
        value method for taking the first integration step.
    rtol : float
        Relative tolereance used by the differential equation method.
    atol : float or array_like
        Absolute tolereance used by the differential equation method.

    Returns
    -------
    float
        An appropriate starting step size to be attempted by the differential
        equation method.

    References
    ----------
    .. [1] H.A. Watts, "Starting step size for an ODE solver", Journal of
           Computational and Applied Mathematics, Vol. 9, No. 2, 1983,
           pp. 177-191, ISSN 0377-0427.
           https://doi.org/10.1016/0377-0427(83)90040-7
    .. [2] Slatec Fortran code dstrt.f.
           https://www.netlib.org/slatec/src/
    """

    # needed to pass scipy unit test:
    if y.size == 0:
        return np.inf

    # compensate for modified call list
    neq = y.size
    spy = np.empty_like(y)
    pv = np.empty_like(y)
    etol = atol + rtol * np.abs(y)

    # `small` is a small positive machine dependent constant which is used for
    # protecting against computations with numbers which are too small relative
    # to the precision of floating point arithmetic. `small` should be set to
    # (approximately) the smallest positive DOUBLE PRECISION number such that
    # (1. + small) > 1.  on the machine being used. The quantity small**(3/8)
    # is used in computing increments of variables for approximating
    # derivatives by differences.  Also the algorithm will not compute a
    # starting step length which is smaller than 100*small*ABS(A).
    # `big` is a large positive machine dependent constant which is used for
    # preventing machine overflows. A reasonable choice is to set big to
    # (approximately) the square root of the largest DOUBLE PRECISION number
    # which can be held in the machine.
    big = sqrt(np.finfo(y.dtype).max)
    small = np.nextafter(np.finfo(y.dtype).epsneg, 1.0)

    # following dhstrt.f from here
    dx = b - a
    absdx = abs(dx)
    relper = small**0.375

    # compute an approximate bound (dfdxb) on the partial derivative of the
    # equation with respect to the independent variable.  protect against an
    # overflow.  also compute a bound (fbnd) on the first derivative locally.
    da = copysign(max(min(relper * abs(a), absdx), 100.0 * small * abs(a)), dx)
    da = da or relper * dx
    sf = df(a + da, y)                                               # evaluate
    yp = sf - yprime
    delf = norm(yp)
    dfdxb = big
    if delf < big * abs(da):
        dfdxb = delf / abs(da)
    fbnd = norm(sf)

    # compute an estimate (dfdub) of the local lipschitz constant for the
    # system of differential equations. this also represents an estimate of the
    # norm of the jacobian locally.  three iterations (two when neq=1) are used
    # to estimate the lipschitz constant by numerical differences.  the first
    # perturbation vector is based on the initial derivatives and direction of
    # integration.  the second perturbation vector is formed using another
    # evaluation of the differential equation.  the third perturbation vector
    # is formed using perturbations based only on the initial values.
    # components that are zero are always changed to non-zero values (except
    # on the first iteration).  when information is available, care is taken to
    # ensure that components of the perturbation vector have signs which are
    # consistent with the slopes of local solution curves.  also choose the
    # largest bound (fbnd) for the first derivative.

    # perturbation vector size is held constant for all iterations.  compute
    # this change from the size of the vector of initial values.
    dely = relper * norm(y)
    dely = dely or relper
    dely = copysign(dely, dx)
    delf = norm(yprime)
    fbnd = max(fbnd, delf)

    if delf:
        # use initial derivatives for first perturbation
        spy[:] = yprime
        yp[:] = yprime
    else:
        # cannot have a null perturbation vector
        spy[:] = 0.0
        yp[:] = 1.0
        delf = norm(yp)

    dfdub = 0.0
    lk = min(neq + 1, 3)
    for k in range(1, lk + 1):

        # define perturbed vector of initial values
        pv[:] = y + dely / delf * yp

        if k == 2:
            # use a shifted value of the independent variable in computing
            # one estimate
            yp[:] = df(a + da, pv)                                   # evaluate
            pv[:] = yp - sf

        else:
            # evaluate derivatives associated with perturbed vector and
            # compute corresponding differences
            yp[:] = df(a, pv)                                        # evaluate
            pv[:] = yp - yprime

        # choose largest bounds on the first derivative and a local lipschitz
        # constant
        fbnd = max(fbnd, norm(yp))
        delf = norm(pv)
        if delf >= big * abs(dely):
            # protect against an overflow
            dfdub = big
            break
        dfdub = max(dfdub, delf / abs(dely))

        if k == lk:
            break

        # choose next perturbation vector
        delf = delf or 1.0
        if k == 2:
            dy = y.copy()                                                 # vec
            dy[:] = np.where(dy, dy, dely / relper)
        else:
            dy = pv.copy()                              # abs removed (complex)
            dy[:] = np.where(dy, dy, delf)
        spy[:] = np.where(spy, spy, yp)

        # use correct direction if possible.
        yp[:] = np.where(spy, np.copysign(dy.real, spy.real), dy.real)
        if np.issubdtype(y.dtype, np.complexfloating):
            yp[:] += 1j*np.where(spy, np.copysign(dy.imag, spy.imag), dy.imag)
        delf = norm(yp)

    # compute a bound (ydpb) on the norm of the second derivative
    ydpb = dfdxb + dfdub * fbnd

    # define the tolerance parameter upon which the starting step size is to be
    # based.  a value in the middle of the error tolerance range is selected.
    tolexp = np.log10(etol)
    tolsum = tolexp.sum()
    tolmin = min(tolexp.min(), big)
    tolp = 10.0 ** (0.5 * (tolsum / neq + tolmin) / (morder + 1))

    # compute a starting step size based on the above first and second
    # derivative information

    # restrict the step length to be not bigger than abs(b-a).
    # (unless b is too close to a)
    h = absdx
    if ydpb == 0.0 and fbnd == 0.0:
        # both first derivative term (fbnd) and second derivative term (ydpb)
        # are zero
        if tolp < 1.0:
            h = absdx * tolp
    elif ydpb == 0.0:
        #  only second derivative term (ydpb) is zero
        if tolp < fbnd * absdx:
            h = tolp / fbnd
    else:
        # second derivative term (ydpb) is non-zero
        srydpb = sqrt(0.5 * ydpb)
        if tolp < srydpb * absdx:
            h = tolp / srydpb

    # further restrict the step length to be not bigger than  1/dfdub
    if dfdub:                                              # `if` added (div 0)
        h = min(h, 1.0 / dfdub)

    # finally, restrict the step length to be not smaller than
    # 100*small*abs(a).  however, if a=0. and the computed h underflowed to
    # zero, the algorithm returns small*abs(b) for the step length.
    h = max(h, 100.0 * small * abs(a))
    h = h or small * abs(b)

    # now set direction of integration
    h = copysign(h, dx)
    return h


class HornerDenseOutput(DenseOutput):
    """use Horner's rule for the evaluation of the dense output polynomials.
    """
    def __init__(self, t_old, t, y_old, Q):
        super(HornerDenseOutput, self).__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q * self.h
        self.y_old = y_old

    def _call_impl(self, t):

        # scaled time
        x = (t - self.t_old) / self.h

        # Horner's rule:
        y = self.Q.T[-1, :, np.newaxis] * x
        for q in reversed(self.Q.T[:-1]):
            y += q[:, np.newaxis]
            y *= x
        y += self.y_old[:, np.newaxis]

        if t.shape:
            return y
        else:
            return y[:, 0]


class CubicDenseOutput(DenseOutput):
    """Cubic, C1 continuous interpolator
    """
    def __init__(self, t_old, t, y_old, y, f_old, f):
        super(CubicDenseOutput, self).__init__(t_old, t)
        self.h = t - t_old
        self.y_old = y_old
        self.f_old = f_old
        self.y = y
        self.f = f

    def _call_impl(self, t):
        # scaled time
        x = (t - self.t_old) / self.h

        # qubic hermite spline:
        h00 = (1.0 + 2.0*x) * (1.0 - x)**2
        h10 = x * (1.0 - x)**2 * self.h
        h01 = x**2 * (3.0 - 2.0*x)
        h11 = x**2 * (x - 1.0) * self.h

        # output
        y = (h00 * self.y_old[:, np.newaxis] + h10 * self.f_old[:, np.newaxis]
             + h01 * self.y[:, np.newaxis] + h11 * self.f[:, np.newaxis])

        if t.shape:
            return y
        else:
            return y[:, 0]


def stiff_a(fun, x, y, hnow, havg, xend, maxfcn, wt, fxy, v0, cost,
            stbrad, tanang):
    """`stiff_a` diagnoses stiffness for an explicit Runge-Kutta code.
    It may be useful for other explicit methods too.

    A nonlinear power method is used to find the dominant eigenvalues of the
    problem. These are compared to the stability regions of the method to
    assess if the problem is stiff. Convergence of the power iterations is
    carefully monitored [1]. This is a Python translation of [2].

    The assessment is not free. Several derivative function evaluations are
    done. This function should not be called each step, but only if either many
    step failures have been observed, or a lot of work has been done.

    Support for complex valued problems is added in a quick and dirty way.
    The original complex vectors are represented by real vectors of twice the
    length with the real and imaginary parts of the original. If the dominant
    eigenvalue of the problem is complex, then its complex conjugate is also
    found. This does not seem to interfere with the stiffness assessment.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is fun(t, y).
        Here t is a scalar. The ndarray y has has shape (n,) and fun must
        return array_like with the same shape (n,).
    x : float
        Current value of the independent variable (time); `t` in scipy
    y : array_like, shape (n,)
        Current approximate solution.
    hnow : float
        Size of the last integration step.
    havg : float
        Average step size.
    xend : float
        Final value of the independent variable (time): `t_bound` in scipy.
    maxfcn : int
        An acceptable number of function evaluations for the integration. The
        Stiffness test is not done if `xend` is expected to be reached in fewer
        than `maxfcn` steps; the value False is returned in that case.
        Prediction uses `havg` and `cost`.
    wt : array_like, shape (n,)
        Array with positive, non-zero weights. Values near the magnitude of `y`
        during the last integration step should work well.
    fxy : array_like, shape (n,)
        fxy = fun(x, y). It is an input argument because it's usually available
        from the integrator.
    v0 : array_like, shape (n,)
        A perturbation vector to start the power iteration to find the dominant
        eigenvalues for stiffness detection. The error vector (difference
        between two embedded solutions) works well for Runge Kutta methods.
    cost : int
        Effective number of function evaluations per step if the integration
        method.
    stbrad : float
        The stability boundary of the method is approximated as a circular arc
        around the origin. `stbrad` is the radius of this arc.
    tanang : float
        The stability boundary of the method is approximated as a circular arc
        around the origin. This approximation becomes inaccurate near the
        imaginary axis. `tanang` is the tangent of the angle in the upper left
        half-plane for which the approximation is still accurate. A eigenvalue
        z for which
            -z.imag/z.real = tan(angle(-z)) <= tanang
        can be assessed. Outside this range, the value None (unsure) is
        returned. A warning is given in that case, because this information may
        still be useful to the user.

    Returns
    -------
    stif : None or bool
        Result of the stiffness detection: True if the problem is stiff,
        False if it is not stiff, and None if unsure.
    rootre : None or bool
        Complex type of root found in stiffness detection: True if the root is
        real, False if it is complex, and None if unknown.

    References
    ----------
    [1] L.F. Shampine, "Diagnosing Stiffness for Runge–Kutta Methods", SIAM
    Journal on Scientific and Statistical Computing, Vol. 12, No. 2, 1991,
    pp. 260-272, https://doi.org/10.1137/0912015
    [2] Original source: RKSuite.f, https://www.netlib.org/ode/rksuite/
    """

    epsneg = np.finfo(y.dtype).epsneg
    rootre = None       # uncertain

    # If the problem is complex, create double length real vectors of the
    # original complex vectors
    if np.issubdtype(y.dtype, np.complexfloating):
        def expand_complex(v): return np.concatenate((v.real, v.imag))
        def contract_complex(v): return v[:v.size//2] + v[v.size//2:]*1j
        def f(x, y): return expand_complex(fun(x, contract_complex(y)))
        y = expand_complex(y)
        fxy = expand_complex(fxy)
        v0 = expand_complex(v0)
        wt = np.concatenate((wt, wt))
    else:
        f = fun

    # If the current step size differs substantially from the average,
    # the problem is not stiff.
    if abs(hnow/havg) > 5 or abs(hnow/havg) < 0.2:
        stif = False
        return stif, rootre

    # The average step size is used to predict the cost in function evaluations
    # of finishing the integration to xend.  If this cost is no more than
    # maxfcn, the problem is declared not stiff: If the step size is being
    # restricted on grounds of stability, it will stay close to havg.
    # The prediction will then be good, but the cost is too low to consider the
    # problem stiff.  If the step size is not close to havg, the problem is not
    # stiff.  Either way there is no point to testing for a step size
    # restriction due to stability.
    xtrfcn = cost * abs((xend - x) / havg)
    if xtrfcn <= maxfcn:
        stif = False
        return stif, rootre

    # There have been many step failures or a lot of work has been done.  Now
    # we must determine if this is due to the stability characteristics of the
    # formula.  This is done by calculating the dominant eigenvalues of the
    # local Jacobian and then testing whether havg corresponds to being on the
    # boundary of the stability region.

    # The size of y[:] provides scale information needed to approximate
    # the Jacobian by differences.
    ynrm = sqrt((y/wt) @ (y/wt))
    sqrrmc = sqrt(epsneg)
    scale = ynrm * sqrrmc
    if scale == 0.0:
        # Degenerate case.  y[:] is (almost) the zero vector so the scale is
        # not defined.  The input vector v0[:] is the difference between y[:]
        # and a lower order approximation to the solution that is within the
        # error tolerance.  When y[:] vanishes, v0[:] is itself an acceptable
        # approximate solution, so we take scale from it, if this is possible.
        ynrm = sqrt((v0/wt) @ (v0/wt))
        scale = ynrm * sqrrmc
        if scale == 0.0:
            stif = None       # uncertain
            return stif, rootre

    v0v0 = (v0/wt) @ (v0/wt)
    if v0v0 == 0.0:
        # Degenerate case.  v0[:] is (almost) the zero vector so cannot
        # be used to define a direction for an increment to y[:].  Try a
        # "random" direction.
        v0[:] = 1.0
        v0v0 = (v0/wt) @ (v0/wt)

    v0nrm = sqrt(v0v0)
    v0 /= v0nrm
    v0v0 = 1.0

    # Use a nonlinear power method to estimate the two dominant eigenvalues.
    # v0[:] is often very rich in the two associated eigenvectors.  For this
    # reason the computation is organized with the expectation that a minimal
    # number of iterations will suffice.  Indeed, it is necessary to recognize
    # a kind of degeneracy when there is a dominant real eigenvalue.  The
    # subroutine stiff_b does this.  In the first try, ntry = 0, a Rayleigh
    # quotient for such an eigenvalue is initialized as rold.  After each
    # iteration, REROOT computes a new Rayleigh quotient and tests whether the
    # two approximations agree to one tenth of one per cent and the eigenvalue,
    # eigenvector pair satisfy a stringent test on the residual.  rootre = True
    # signals that a single dominant real root has been found.
    maxtry = 8
    for ntry in range(maxtry):

        v1, v1v1 = stiff_d(v0, havg, x, y, f, fxy, wt, scale, v0v0)

        # The quantity sqrt(v1v1/v0v0) is a lower bound for the product of havg
        # and a Lipschitz constant.  If it should be LARGE, stiffness is not
        # restricting the step size to the stability region.  The principle is
        # clear enough, but the real reason for this test is to recognize an
        # extremely inaccurate computation of v1v1 due to finite precision
        # arithmetic in certain degenerate circumstances.
        LARGE = 1.0e10
        if sqrt(v1v1) > LARGE * sqrt(v0v0):
            stif = None       # uncertain
            rootre = None     # uncertain
            return stif, rootre

        v0v1 = (v0/wt) @ (v1/wt)
        if ntry == 0:
            rold = v0v1 / v0v0
            # This is the first Rayleigh quotient approximating the product of
            # havg and a dominant real eigenvalue.  If it should be very small,
            # the problem is not stiff.  It is important to test for this
            # possibility so as to prevent underflow and degeneracies in the
            # subsequent iteration.
            cubrmc = epsneg ** (1/3)
            if abs(rold) < cubrmc:
                stif = False
                rootre = None     # uncertain
                return stif, rootre

        else:
            rold, rho, root1, root2, rootre = stiff_b(v1v1, v0v1, v0v0, rold)
            if rootre:
                break

        v2, v2v2 = stiff_d(v1, havg, x, y, f, fxy, wt, scale, v1v1)
        v0v2 = (v0/wt) @ (v2/wt)
        v1v2 = (v1/wt) @ (v2/wt)
        rold, rho, root1, root2, rootre = stiff_b(v2v2, v1v2, v1v1, rold)
        if rootre:
            break

        # Fit a quadratic in the eigenvalue to the three successive iterates
        # v0[:], v1[:], v2[:] of the power method to get a first approximation
        # to a pair of eigenvalues.  A test made earlier in stiff_b implies
        # that the quantity det1 here will not be too small.
        det1 = v0v0 * v1v1 - v0v1**2
        alpha1 = (-v0v0 * v1v2 + v0v1 * v0v2)/det1
        beta1 = (v0v1 * v1v2 - v1v1 * v0v2)/det1

        # Iterate again to get v3, test again for degeneracy, and then fit a
        # quadratic to v1[:], v2[:], v3[:] to get a second approximation to a
        # pair of eigenvalues.
        v3, v3v3 = stiff_d(v2, havg, x, y, f, fxy, wt, scale, v2v2)
        v1v3 = (v1/wt) @ (v3/wt)
        v2v3 = (v2/wt) @ (v3/wt)
        rold, rho, root1, root2, rootre = stiff_b(v3v3, v2v3, v2v2, rold)
        if rootre:
            break
        det2 = v1v1 * v2v2 - v1v2**2
        alpha2 = (-v1v1 * v2v3 + v1v2 * v1v3)/det2
        beta2 = (v1v2 * v2v3 - v2v2 * v1v3)/det2

        # First test the residual of the quadratic fit to see if we might
        # have determined a pair of eigenvalues.
        res2 = abs(v3v3 + v2v2*alpha2**2 + v1v1*beta2**2 + 2*v2v3*alpha2 +
                   2*v1v3*beta2 + 2*v1v2*alpha2*beta2)
        if res2 <= 1e-6 * v3v3:
            # Calculate the two approximate pairs of eigenvalues.
            r1, r2 = stiff_c(alpha1, beta1)
            root1, root2 = stiff_c(alpha2, beta2)

            # The test for convergence is done on the larger root of the second
            # approximation.  It is complicated by the fact that one pair of
            # roots might be real and the other complex.  First calculate the
            # spectral radius rho of havg*J as the magnitude of root1.  Then
            # see if one of the roots r1, r2 is within one per cent of root1.
            # A subdominant root may be very poorly approximated if its
            # magnitude is much smaller than rho -- this does not matter in our
            # use of these eigenvalues.
            rho = sqrt(root1[0]**2 + root1[1]**2)
            D1 = (root1[0] - r1[0])**2 + (root1[1] - r1[1])**2
            D2 = (root1[0] - r2[0])**2 + (root1[1] - r2[1])**2
            DIST = sqrt(min(D1, D2))
            if DIST <= 0.001*rho:
                break

        # Do not have convergence yet.  Because the iterations are cheap, and
        # because the convergence criterion is stringent, we are willing to try
        # a few iterations.
        v3nrm = sqrt(v3v3)
        v0 = v3/v3nrm
        v0v0 = 1.0

    else:
        # Iterations did not converge
        stif = None       # uncertain
        rootre = None     # uncertain
        return stif, rootre

    # Iterations have converged

    # We now have the dominant eigenvalues.  Decide if the average step size is
    # being restricted on grounds of stability.  Check the real parts of the
    # eigenvalues.  First see if the dominant eigenvalue is in the left half
    # plane -- there won't be a stability restriction unless it is.  If there
    # is another eigenvalue of comparable magnitude with a positive real part,
    # the problem is not stiff.  If the dominant eigenvalue is too close to the
    # imaginary axis, we cannot diagnose stiffness.

    rootre = root1[1] == 0.0

    # print(ntry, (root1[0] + root1[1]*1j)/havg,
    #             (root2[0] + root2[1]*1j)/havg, rootre)

    if root1[0] > 0.0:
        stif = False
        return stif, rootre

    rho2 = sqrt(root2[0]**2 + root2[1]**2)
    if rho2 >= 0.9 * rho and root2[0] > 0.0:
        stif = False
        return stif, rootre

    if abs(root1[1]) > abs(root1[0]) * tanang:
        stif = None       # uncertain
        return stif, rootre

    # If the average step size corresponds to being well within the stability
    # region, the step size is not being restricted because of stability.
    stif = rho >= 0.9 * stbrad
    return stif, rootre


def stiff_b(v1v1, v0v1, v0v0, rold):
    """called from stiff_a().

    Decide if the iteration has degenerated because of a strongly dominant
    real eigenvalue.  Have just computed the latest iterate.  v1v1 is its dot
    product with itself, v0v1 is the dot product of the previous iterate with
    the current one, and v0v0 is the dot product of the previous iterate with
    itself.  rold is a previous Rayleigh quotient approximating a dominant real
    eigenvalue.  It must be computed directly the first time the subroutine is
    called.  It is updated each call to stiff_b, hence is available for
    subsequent calls.

    If there is a strongly dominant real eigenvalue, rootre is set True,
    root1[:] returns the eigenvalue, rho returns the magnitude of the
    eigenvalue, and root2[:] is set to zero.

    Original source: RKSuite.f, https://www.netlib.org/ode/rksuite/
    """
    # real and imag parts of roots are returned in a list
    root1 = [0.0, 0.0]
    root2 = [0.0, 0.0]

    r = v0v1 / v0v0
    rho = abs(r)
    det = v0v0 * v1v1 - v0v1**2
    res = abs(det / v0v0)
    rootre = det == 0.0 or (res <= 1e-6 * v1v1 and
                            abs(r - rold) <= 0.001 * rho)
    if rootre:
        root1[0] = r
    rold = r
    return rold, rho, root1, root2, rootre


def stiff_c(alpha, beta):
    """called from stiff_a().

    This subroutine computes the two complex roots r1 and r2 of the
    quadratic equation x**2 + alpha*x + beta = 0.  The magnitude of r1 is
    greater than or equal to the magnitude of r2. r1 and r2 are returned as
    vectors of two components with the first being the real part of the complex
    number and the second being the imaginary part.

    Original source: RKSuite.f, https://www.netlib.org/ode/rksuite/
    """
    # real and imag parts of roots are returned in a list
    r1 = [0.0, 0.0]
    r2 = [0.0, 0.0]

    temp = alpha / 2
    disc = temp**2 - beta
    if disc == 0.0:
        # Double root.
        r1[0] = r2[0] = -temp
        return r1, r2

    sqdisc = sqrt(abs(disc))
    if disc < 0.0:
        # Complex conjugate roots.
        r1[0] = r2[0] = -temp
        r1[1] = sqdisc
        r2[1] = -sqdisc
    else:
        # Real pair of roots.  Calculate the bigger one in r1[0].
        if temp > 0.0:
            r1[0] = -temp - sqdisc
        else:
            r1[0] = -temp + sqdisc
        r2[0] = beta / r1[0]
    return r1, r2


def stiff_d(v, havg, x, y, f, fxy, wt, scale, vdotv):
    """called from stiff_a().

    For an input vector v[:], this subroutine computes a vector z[:] that
    approximates the product havg*J*V where havg is an input scalar and J is
    the Jacobian matrix of a function f evaluated at the input arguments
    (x, y[:]).  This function is defined by a subroutine of the form
    f[:] = f(t, u) that when given t and u[:], returns the value of the
    function in f[:].  The input vector fxy[:] is defined by f(x, y).  Scaling
    is a delicate matter.  A weighted Euclidean norm is used with the
    (positive) weights provided in wt[:].  The input scalar scale is the square
    root of the unit roundoff times the norm of y[:].  The square of the norm
    of the input vector v[:] is input as vdotv.  The routine outputs the square
    of the norm of the output vector z[:] as zdotz.

    Original source: RKSuite.f, https://www.netlib.org/ode/rksuite/
    """
    # scale v[:] so that it can be used as an increment to y[:]
    # for an accurate difference approximation to the Jacobian.
    temp1 = scale/sqrt(vdotv)
    z = f(x, y + temp1 * v)                                          # evaluate

    # Form the difference approximation.  At the same time undo
    # the scaling of v[:] and introduce the factor of havg.
    z = havg/temp1 * (z - fxy)
    zdotz = (z/wt) @ (z/wt)
    return z, zdotz
