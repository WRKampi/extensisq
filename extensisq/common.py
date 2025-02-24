import numpy as np
from math import sqrt, copysign
from warnings import warn
import logging
from scipy.integrate._ivp.common import (
    validate_max_step, validate_first_step, warn_extraneous, num_jac)
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse, csc_array, diags_array, eye_array
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from scipy.optimize import root

NFS = np.array(0)                                         # failed step counter
NFI = np.array(0)                                    # failed iteration counter
NLS = np.array(0)                                # linear system solves counter

MIN_FACTOR = 0.2
MAX_FACTOR = 4.0
MAX_FACTOR0 = 10

# for NR iteration
NEWTON_MAXITER = 5
MAX_RATE = 0.2
MAX_FACTOR_NRF = 0.5
# factor for stepsize at NR iteration failure:
#     MIN_FACTOR < MAX_RATE/Rate < MAX_FACTOR_NRF


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
    rtol = np.minimum(np.maximum(rtol, 10 * epsneg), 0.1)
    return rtol, atol


def calculate_scale(atol, rtol, y, y_new, _mean=False):
    """calculate a scaling vector for the error estimate"""
    if _mean:
        return atol + rtol * 0.5*(np.abs(y) + np.abs(y_new))
    return atol + rtol * np.maximum(np.abs(y), np.abs(y_new))


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
    order_secondary: int = NotImplemented

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

    max_factor = MAX_FACTOR0                    # initially
    min_factor = MIN_FACTOR

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
        coefs = {"G": (0.7, -0.4, 0, 0.9),      # Gustafsson
                 "S": (0.6, -0.2, 0, 0.9),      # Soderlind
                 "standard": (1, 0, 0, 0.9)}
        # use default controller of method if not specified otherwise
        sc_params = sc_params or self.sc_params
        if (isinstance(sc_params, str) and sc_params in coefs):
            kb1, kb2, a, g = coefs[sc_params]
        elif isinstance(sc_params, tuple) and len(sc_params) == 4:
            kb1, kb2, a, g = sc_params
        else:
            raise ValueError('sc_params should be a tuple of length 4 or one '
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
                 nfev_stiff_detect=5000, sc_params=None, support_complex=True,
                 **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized,
                         support_complex=support_complex)
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.y)
        self.f = self.fun(self.t, self.y)
        if self.f.dtype != self.y.dtype:
            raise TypeError('dtypes of solution and derivative do not match')
        order_error = min(self.order_secondary, self.order)
        self.error_exponent = -1 / (order_error + 1)
        self._init_stiffness_detection(nfev_stiff_detect)
        self.h_min_a, self.h_min_b = self._init_min_step_parameters()
        self.tiny_err = self.h_min_b
        self._init_sc_control(sc_params)

        # size of first step:
        if first_step is None:
            b = self.t + self.direction * min(
                abs(self.t_bound - self.t), self.max_step)
            self.h_abs = abs(h_start(
                self.fun, self.t, b, self.y, self.f,
                self.order_secondary, self.rtol, self.atol))
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

                if error_norm < self.tiny_err:
                    factor = self.max_factor
                    self.standard_sc = True

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
                    factor = min(self.max_factor, max(self.min_factor, factor))

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                if factor < MAX_FACTOR:
                    # reduce max_factor when on scale.
                    self.max_factor = MAX_FACTOR

            else:
                step_rejected = True
                h_abs *= max(self.min_factor,
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
        """Compute solution and error"""
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
            stif, rootre, root = stiff_a(
                self.fun, self.t, self.y, self.h_previous, self.havg,
                self.t_bound, self.nfev_stiff_detect, wt, self.f, v0,
                self.n_stages)

            if root is not None:
                root1, root2, rho = root
                # this is cut from stiff_a, because it compares the found roots
                # to the limits 'stbrad' and 'tanang', which are specific to
                # explicit Runge Kutta methods.

                # We now have the dominant eigenvalues.  Decide if the average
                # step size is being restricted on grounds of stability.  Check
                # the real parts of the eigenvalues.  First see if the dominant
                # eigenvalue is in the left half plane -- there won't be a
                # stability restriction unless it is.  If there is another
                # eigenvalue of comparable magnitude with a positive real part,
                # the problem is not stiff.  If the dominant eigenvalue is too
                # close to the imaginary axis, we cannot diagnose stiffness.

                rootre = root1[1] == 0.0
                if root1[0] > 0.0:
                    stif = False
                else:
                    rho2 = sqrt(root2[0]**2 + root2[1]**2)
                    if rho2 >= 0.9 * rho and root2[0] > 0.0:
                        stif = False
                    elif abs(root1[1]) > abs(root1[0]) * self.tanang:
                        stif = None       # uncertain
                    else:
                        # If the average step size corresponds to being well
                        # within the stability region, the step size is not
                        # being restricted because of stability.
                        stif = rho >= 0.9 * self.stbrad

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


def h_start(df, a, b, y, yprime, morder, rtol, atol,
            J=None, T=None, returnT=False):
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
    J : array_like, shape (n, n), optional
        The Jacobian matrix of the system. If this is supplied, its norm can be
        calculated directly, rather than estimated by sampling.
    T : array_like, shape (n,)
    returnT : bool, optional
        If true, only return the estimate of the derivative of df to time.

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
    da = copysign(max(min(relper * abs(a), absdx), 100. * small * abs(a)), dx)
    da = da or relper * dx
    if T is None:
        sf = df(a + da, y)                                           # evaluate
    else:
        sf = yprime + da * T
    yp = sf - yprime
    delf = norm(yp)
    dfdxb = big
    if delf < big * abs(da):
        dfdxb = delf / abs(da)
    fbnd = norm(sf)
    if returnT:
        return yp/da

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

    # added: if the jacobian matrix is supplied, the iterative calculation is
    # not done, but its Frobenius norm is calculated directly.

    if J is None:
        # perturbation vector size is held constant for all iterations.
        # Compute this change from the size of the vector of initial values.
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
                yp[:] = df(a + da, pv)                               # evaluate
                pv[:] = yp - sf

            else:
                # evaluate derivatives associated with perturbed vector and
                # compute corresponding differences
                yp[:] = df(a, pv)                                    # evaluate
                pv[:] = yp - yprime

            # choose largest bounds on the first derivative and a local
            # lipschitz constant
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
                dy = y.copy()                                             # vec
                dy[:] = np.where(dy, dy, dely / relper)
            else:
                dy = pv.copy()                          # abs removed (complex)
                dy[:] = np.where(dy, dy, delf)
            spy[:] = np.where(spy, spy, yp)

            # use correct direction if possible.
            yp[:] = np.where(spy, np.copysign(dy.real, spy.real), dy.real)
            if np.issubdtype(y.dtype, np.complexfloating):
                yp[:] += 1j*np.where(spy, np.copysign(dy.imag, spy.imag),
                                     dy.imag)
            delf = norm(yp)
    else:
        # added option of direct calculation if J is supplied
        dfdub = np.linalg.norm(J)

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
        super().__init__(t_old, t)
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
        super().__init__(t_old, t)
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


def stiff_a(fun, x, y, hnow, havg, xend, maxfcn, wt, fxy, v0, cost):
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
    roots: None or tuple (root1, root2, rho)
        a tuple with the two roots (lists [re, im]) and the magnitude of root1
        (if the iteration is done and has converged), or None (if the iteration
        was not necessary or did not converge).

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
        root = None
        return stif, rootre, root

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
        root = None
        return stif, rootre, root

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
            root = None
            return stif, rootre, root

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
            root = None
            return stif, rootre, root

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
                root = None
                return stif, rootre, root

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
        root = None
        return stif, rootre, root

    # Iterations have converged
    # further processing is done in the caller
    stif = None     # still unknown, to be analysed from the roots
    return stif, rootre, (root1, root2, rho)


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


class RungeKuttaNystrom(RungeKutta):
    """Base class for explicit runge kutta nystrom methods.
        [v, a] = fun(t, [u, v])
    """
    # parameters of the stability rectangle
    stbre: float = NotImplemented              # size on real axis
    stbim: float = NotImplemented              # size on imaginary axis
    tanang: float = NotImplemented              # tan(valid angle < pi/2)

    # Besides A, B, C, E, and P, we need the derivative matrices Ap, Bp, Ep,
    # and Pp.
    # Ap should not be defined for a velocity independent method.
    Ap: np.ndarray = NotImplemented              # shape: [n_stages, n_stages]
    Bp: np.ndarray = NotImplemented              # shape: [n_stages]
    Ep: np.ndarray = NotImplemented              # shape: [n_stages + 1]
    Pp: np.ndarray = NotImplemented              # shape: [n_stages + 1,
    #                                                     order_polynomial]

    def _init_stiffness_detection(self, nfev_stiff_detect):
        if not (isinstance(nfev_stiff_detect, int) and nfev_stiff_detect >= 0):
            raise ValueError(
                "`nfev_stiff_detect` must be a non-negative integer.")
        self.nfev_stiff_detect = nfev_stiff_detect
        if NotImplemented in (self.stbre, self.stbim, self.tanang):
            # disable stiffness detection if not implemented
            if nfev_stiff_detect not in (5000, 0):
                warn("This method does not implement stiffness detection. "
                     "Changing the value of nfev_stiff_detect does nothing.")
            self.nfev_stiff_detect = 0
        self.jflstp = 0                         # failed step counter, last 40
        if self.nfev_stiff_detect:
            self.okstp = 0                      # successful step counter
            self.havg = 0.0                     # average stepsize

    def __init__(self, fun, t0, y0, t_bound, nfev_stiff_detect=5000,
                 **extraneous):
        super().__init__(fun, t0, y0, t_bound, **extraneous)
        # extraneous can include:
        # max_step, rtol, sc_params, atol, vectorized, first_step
        self._init_stiffness_detection(nfev_stiff_detect)
        self.n = self.y.size // 2
        # check if fun is a correctly structured 2nd order problem
        msg = ('This method is for second order problems'
               ' and `fun` should have signature: [v, a] = fun(t, [x, v]).')
        if (self.y.size % 2) or not np.all(self.y[self.n:] == self.f[:self.n]):
            raise AssertionError(msg)
        elif np.all(self.y[self.n:] == self.y[:self.n]):
            y_test = self.y.copy()
            y_test[self.n:] *= 1 + 1e-8
            y_test[self.n:] += 1e-8
            if not np.all(fun(t0, y_test)[:self.n] == y_test[self.n:]):
                raise AssertionError(msg)
        if self.Ap is NotImplemented:
            msg = ("This method is for velocity independent ODEs, "
                   "but `fun` seems velocity dependent.")
            y_test = self.y.copy()
            y_test[self.n:] *= 1 + 1e-8
            y_test[self.n:] += 1e-8
            if not np.all(fun(t0, y_test)[self.n:] == self.f[self.n:]):
                raise AssertionError(msg)
            self.Ap = np.zeros((self.n_stages, self.n_stages))          # dummy
        # check Ep, (E is already checked)
        if self.Ep[-1] != 0.:
            self.FSAL = 1
        # need storage for accelerations only
        self.K = np.empty((self.n_stages + 1, self.n), self.y.dtype)
        self.f = self.f[self.n:]
        self.fun_first_order = fun

        def fun(*args, fun=self.fun, n=self.n):
            return fun(*args)[n:]

        self.fun = fun

    def _rk_stage(self, h, i):
        """compute a single RK stage"""
        dt = self.C[i] * h
        du = (self.K[:i, :].T @ self.A[i, :i])*h**2 + dt*self.y[self.n:]
        dv = (self.K[:i, :].T @ self.Ap[i, :i])*h
        dy = np.concatenate((du, dv))
        self.K[i] = self.fun(self.t + dt, self.y + dy)

    def _comp_sol_err(self, y, h):
        du = (self.K[:self.n_stages, :].T @ self.B) * h**2 \
            + h * self.y[self.n:]
        dv = (self.K[:self.n_stages, :].T @ self.Bp) * h
        dy = np.concatenate((du, dv))
        y_new = y + dy
        scale = calculate_scale(self.atol, self.rtol, y, y_new)

        if self.FSAL:
            # do FSAL evaluation if needed for error estimate
            self.K[self.n_stages, :] = self.fun(self.t + h, y_new)

        error_norm = self._estimate_error_norm(self.K, h, scale)
        return y_new, error_norm

    def _estimate_error(self, K, h):
        eu = (self.K[:self.n_stages + self.FSAL, :].T @
              self.E[:self.n_stages + self.FSAL]) * h**2
        ev = (self.K[:self.n_stages + self.FSAL, :].T @
              self.Ep[:self.n_stages + self.FSAL]) * h
        e = np.concatenate((eu, ev))
        return e

    def _dense_output_impl(self):

        if isinstance(self.P, np.ndarray) and isinstance(self.Pp, np.ndarray):
            Q = self.K.T @ self.P
            Qp = self.K.T @ self.Pp
            return HornerDenseOutputNystrom(
                self.t_old, self.t, self.y_old, Q, Qp)

        return QuinticHermiteDenseOutput(
            self.t_old, self.t, self.y_old, self.y, self.f_old, self.f)

    def _diagnose_stiffness(self):
        """Stiffness detection. Copied from RungeKutta, but adapted for a
        rectangular stability domain

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

            # back to first order form                                   change
            f = np.concatenate((self.y[self.n:], self.f))
            fun = self.fun_first_order
            stif, rootre, root = stiff_a(
                fun, self.t, self.y, self.h_previous, self.havg, self.t_bound,
                self.nfev_stiff_detect, wt, f, v0, self.n_stages)

            if root is not None:
                root1, root2, rho = root

                # this is cut from stiff_a, because it compares the found roots
                # to the limits 'stbrad' and 'tanang', which are specific to
                # explicit Runge Kutta methods.

                # We now have the dominant eigenvalues.  Decide if the average
                # step size is being restricted on grounds of stability.  Check
                # the real parts of the eigenvalues.  First see if the dominant
                # eigenvalue is in the left half plane -- there won't be a
                # stability restriction unless it is.  If there is another
                # eigenvalue of comparable magnitude with a positive real part,
                # the problem is not stiff.  If the dominant eigenvalue is too
                # close to the imaginary axis, we cannot diagnose stiffness.

                rootre = root1[1] == 0.0
                if root1[0] > 0.0:
                    stif = False
                else:
                    rho2 = sqrt(root2[0]**2 + root2[1]**2)
                    if rho2 >= 0.9 * rho and root2[0] > 0.0:
                        stif = False
                    elif abs(root1[1]) > abs(root1[0]) * self.tanang:
                        stif = None       # uncertain
                    else:
                        # If the average step size corresponds to being well
                        # within the stability region, the step size is not
                        # being restricted because of stability.
                        stif = (abs(root1[0]) >= 0.85 * self.stbre or  # change
                                abs(root1[1]) >= 0.9 * self.stbim)

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
                else:                                                  # change
                    if abs(root1[0]) >= 0.9 * self.stbre:
                        warn('Your problem has a complex pair of dominant '
                             'roots and is diagnosed as stiff '
                             '(large real part).  You should probably change '
                             'to a code intended for stiff problems.')
                    elif abs(root1[1]) >= 0.9 * self.stbim:
                        warn('Your problem has a complex pair of dominant '
                             'roots and is diagnosed as stiff '
                             '(large imaginary part).  You should probably '
                             'change to a code intended for stiff problems.')
                    else:
                        logging.warning(
                            'stif=True, rootre=False, not out of bounds; '
                            'this should not happen')
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


class HornerDenseOutputNystrom(DenseOutput):
    """use Horner's rule for the evaluation of the dense output polynomials.
    """
    def __init__(self, t_old, t, y_old, Q, Qp):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q * self.h**2
        self.Qp = Qp * self.h
        self.y_old = y_old
        self.n = self.y_old.size // 2

    def _call_impl(self, t):
        # scaled time
        xi = (t - self.t_old) / self.h

        # Velocity, Horner's rule
        v = self.Qp.T[-1, :, np.newaxis] * xi
        for q in reversed(self.Qp.T[:-1]):
            v += q[:, np.newaxis]
            v *= xi
        v += self.y_old[self.n:, np.newaxis]

        # Displacement, Horner's rule
        u = self.Q.T[-1, :, np.newaxis] * xi
        for q in reversed(self.Q.T[:-1]):
            u += q[:, np.newaxis]
            u *= xi
        u += self.y_old[self.n:, np.newaxis] * self.h
        u *= xi
        u += self.y_old[:self.n, np.newaxis]

        y = np.concatenate((u, v), axis=0)

        if t.shape:
            return y
        else:
            return y[:, 0]


class QuinticHermiteDenseOutput(DenseOutput):
    """Quintic, C2 continuous interpolator for 2nd order ODEs
    (C2 for x, C1 for v, order=5)"""

    P = np.array([[1, 0, 0, -10, 15, -6],
                  [0, 1, 0, -6, 8, -3],
                  [0, 0, 1/2, -3/2, 3/2, -1/2],
                  [0, 0, 0, 10, -15, 6],
                  [0, 0, 0, -4, 7, -3],
                  [0, 0, 0, 1/2, -1, 1/2]])
    Pp = P[:, 1:] * np.arange(1, 6)

    def __init__(self, t_old, t, y_old, y, f_old, f):
        super().__init__(t_old, t)
        self.h = t - t_old
        n = y.size//2
        self.x_old = y_old[:n]
        self.v_old = y_old[n:]
        self.a_old = f_old
        self.x = y[:n]
        self.v = y[n:]
        self.a = f

    def _call_impl(self, t):
        # scaled time
        h = self.h
        xi = (t - self.t_old) / h

        Q = np.array(
            [self.x_old, self.v_old*h, self.a_old*h**2,
             self.x, self.v*h, self.a*h**2]).T @ self.P
        Qp = np.array(
            [self.x_old/h, self.v_old, self.a_old*h,
             self.x/h, self.v, self.a*h]).T @ self.Pp

        # Horner's rule:
        x = Q.T[-1, :, np.newaxis] * np.ones_like(xi)
        for q in reversed(Q.T[:-1]):
            x *= xi
            x += q[:, np.newaxis]

        v = Qp.T[-1, :, np.newaxis] * np.ones_like(xi)
        for q in reversed(Qp.T[:-1]):
            v *= xi
            v += q[:, np.newaxis]

        y = np.concatenate((x, v), axis=0)
        if t.shape:
            return y
        else:
            return y[:, 0]

    def _call_impl_old(self, t):
        # scaled time
        h = self.h
        xi = (t - self.t_old) / h

        # quintic hermite spline:
        d1 = 1. - 10.*xi**3 + 15.*xi**4 - 6.*xi**5
        d2 = (xi - 6.*xi**3 + 8.*xi**4 - 3.*xi**5) * h
        d3 = (xi**2 - 3.*xi**3 + 3.*xi**4 - xi**5) * 0.5*h**2
        d4 = (xi**3 - 2.*xi**4 + xi**5) * 0.5*h**2
        d5 = (-4.*xi**3 + 7.*xi**4 - 3.*xi**5) * h
        d6 = 10.*xi**3 - 15.*xi**4 + 6.*xi**5

        # derivative
        dp1 = (-30.*xi**2 + 60.*xi**3 - 30.*xi**4)/h
        dp2 = 1. - 18.*xi**2 + 32.*xi**3 - 15.*xi**4
        dp3 = (2*xi - 9.*xi**2 + 12.*xi**3 - 5*xi**4) * 0.5*h
        dp4 = (3*xi**2 - 8.*xi**3 + 5*xi**4) * 0.5*h
        dp5 = -12.*xi**2 + 28.*xi**3 - 15.*xi**4
        dp6 = (30.*xi**2 - 60.*xi**3 + 30.*xi**4)/h

        # output
        x = (d1*self.x_old[:, np.newaxis] + d6*self.x[:, np.newaxis] +
             d2*self.v_old[:, np.newaxis] + d5*self.v[:, np.newaxis] +
             d3*self.a_old[:, np.newaxis] + d4*self.a[:, np.newaxis])
        v = (dp1*self.x_old[:, np.newaxis] + dp6*self.x[:, np.newaxis] +
             dp2*self.v_old[:, np.newaxis] + dp5*self.v[:, np.newaxis] +
             dp3*self.a_old[:, np.newaxis] + dp4*self.a[:, np.newaxis])
        y = np.concatenate((x, v), axis=0)

        if t.shape:
            return y
        else:
            return y[:, 0]


class ESDIRK(OdeSolver):
    """Base class for explicit first stage diagonal implicit runge kutta
    methods. FSAL is assumed (First Same As Last).
    ... Hosea, Shampine, BDF ...
    Jacobian and LU update strategy follows scipy implementation of BDF.
    If jac is supplied as an array, the solver assumes a linear system:
    it refactors LU each step and terminates the NR after one iteration.
    """

    # total number of stages (not the effective Nr)
    n_stages: int = NotImplemented

    # order of the main method
    order: int = NotImplemented

    # order of the secondary embedded method
    order_secondary: int = NotImplemented

    # NR iteration tolerance facor
    kappa: float = NotImplemented

    # runge kutta coefficient array
    d: float = NotImplemented                   # diagonal value
    A: np.ndarray = NotImplemented              # shape: [n_stages, n_stages]
    B: np.ndarray = NotImplemented              # shape: [n_stages]
    C: np.ndarray = NotImplemented              # shape: [n_stages]
    E: np.ndarray = NotImplemented              # shape: [n_stages]

    # dense output interpolation coefficients, optional
    P: np.ndarray = NotImplemented              # shape: [n_stages, poly order]

    # for prediction of z
    Az: np.ndarray = NotImplemented              # shape: [n_stages, n_stages]

    # Parameters for stepsize control, optional
    sc_params = "G"

    filter_error = False
    max_factor = MAX_FACTOR0          # initially
    min_factor = MIN_FACTOR

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

    def _init_sc_control(self, sc_params):
        # implement different controller, see Kennedy 2019
        coefs = {"G": (2., -1., -1., 0.8),     # Gustafsson
                 "S": (1.1, -0.7, -1., 0.8),   # Soderlind
                 "standard": (1, 0, 0, 0.8)}
        # use default controller of method if not specified otherwise
        sc_params = sc_params or self.sc_params
        if (isinstance(sc_params, str) and sc_params in coefs):
            kb1, kb2, a, g = coefs[sc_params]
        elif isinstance(sc_params, tuple) and len(sc_params) == 4:
            kb1, kb2, a, g = sc_params
        else:
            raise ValueError('sc_params should be a tuple of length 4 or one '
                             'of the strings "G", "S" or "standard"')
        # set all parameters
        self.minbeta1 = kb1 * self.error_exponent
        self.minbeta2 = kb2 * self.error_exponent
        self.minalpha = -a
        self.safety = g
        self.safety_sc = g ** (kb1 + kb2)
        self.standard_sc = True                                # for first step

    def _validate_jac(self, jac, sparsity):
        t0 = self.t
        y0 = self.y
        if jac is None:
            if sparsity is not None:
                if issparse(sparsity):
                    sparsity = csc_array(sparsity)
                groups = group_columns(sparsity)
                sparsity = (sparsity, groups)

            def jac_wrapped(t, y):
                self.njev += 1
                f = self.fun_single(t, y)
                J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f,
                                             self.atol, self.jac_factor,
                                             sparsity)
                return J
            J = jac_wrapped(t0, y0)
        elif callable(jac):
            J = jac(t0, y0)
            self.njev += 1
            if issparse(J):
                J = csc_array(J, dtype=y0.dtype)

                def jac_wrapped(t, y):
                    self.njev += 1
                    return csc_array(jac(t, y), dtype=y0.dtype)
            else:
                J = np.asarray(J, dtype=y0.dtype)

                def jac_wrapped(t, y):
                    self.njev += 1
                    return np.asarray(jac(t, y), dtype=y0.dtype)

            if J.shape != (self.n, self.n):
                raise ValueError(
                    f"`jac` is expected to have shape {(self.n, self.n)},"
                    f" but actually has {J.shape}.")
        else:
            if issparse(jac):
                J = csc_array(jac, dtype=y0.dtype)
            else:
                J = np.asarray(jac, dtype=y0.dtype)
            if J.shape != (self.n, self.n):
                raise ValueError(
                    f"`jac` is expected to have shape {(self.n, self.n)},"
                    f" but actually has {J.shape}.")
            jac_wrapped = None
        return jac_wrapped, J

    def _set_lu_functions(self, J):
        if issparse(J):
            def lu(A):
                self.nlu += 1
                return splu(A)

            def solve_lu(LU, b):
                NLS[()] += 1
                return LU.solve(b)

        else:
            # not sparse
            def lu(A):
                self.nlu += 1
                return lu_factor(A, overwrite_a=True)

            def solve_lu(LU, b):
                NLS[()] += 1
                return lu_solve(LU, b, overwrite_b=True)

        return lu, solve_lu

    def _handle_M(self, M):
        """convert M to correct format, detect if DAE.
        """
        isDAE = False
        M_details = {}
        if M is None:
            M_mat = eye_array(self.n)
            return M_mat, M_details, isDAE
        elif issparse(M):
            # make dense for this function
            M = M.toarray()
        M = np.asarray(M)
        ndim = M.ndim
        if ndim not in {1, 2}:
            raise ValueError("M should be a 1D or 2D array")
        for n in M.shape:
            if n != self.n:
                raise ValueError("M should have shape (n,) or (n, n)")
        if ndim == 1:
            # convert to 2D for analysis
            M = np.diag(M)
        else:
            # check if 2D M is diagonal matrix, but keep M 2D
            d = np.diagonal(M)
            if np.all(M - np.diag(d) == 0.):
                ndim = 1
        # check if DAE using SVD
        U, s, Vh = np.linalg.svd(M)
        cond_lim = s[0] * self.n**2 * np.finfo(self.y.dtype).eps
        nAE = np.sum(s < cond_lim)
        isDAE = nAE > 0
        # store M in correct format
        M_mat = M
        if ndim == 1:
            M_mat = diags_array(np.diagonal(M))
        else:
            if self.sparse:
                M_mat = csc_array(M)
        if isDAE:
            U = csc_array(U)
            Vh = csc_array(Vh)
        sliceEA = np.s_[-nAE:]
        M_details = {'svd': (U, s, Vh), 's_AE': sliceEA}
        return M_mat, M_details, isDAE

    def _consistent_ICs(self):
        """Make the initial conditions and the derivative consistent.
        call only for DAEs
        """
        assert self.isDAE, 'Only call this for DAEs'
        J = self.J
        if self.sparse:
            J = J.todense()
        jac = self.jac
        if jac is None:
            def jac(t, y, J=J):
                self.njev += 1
                return J
        elif self.sparse:
            def jac(t, y):
                return self.jac(t, y).todense()
        fun = self.fun_single
        t = self.t
        y = self.y
        f = self.f

        # check index
        s_v = self.M_details['s_AE']
        s_u = np.s_[:s_v.start]
        U, s, Vh = self.M_details['svd']
        Gvv = (U.T @ J @ Vh.T)[s_v, s_v]
        index1 = np.linalg.matrix_rank(Gvv) == Gvv.shape[1]
        if not index1:
            raise ValueError(
                "The index of the DAE seems to be larger than 1."
                " This method is not suitable for solving it.")

        # find consistent y0 if needed
        b = U.T @ f
        consistent_y = np.allclose(b[s_v], 0.)
        if consistent_y:
            u = (Vh @ y)[s_u]
        else:
            y0 = y.copy()
            z0 = Vh @ y0
            v0 = z0[s_v]
            u = z0[s_u]     # fixed

            def funC(v, u=u, t=t, U=U, Vh=Vh):
                y = Vh.T @ np.r_[u, v]
                f = fun(t, y)
                gv = (U.T @ f)[s_v]
                return gv

            def jacC(v, u=u, t=t, U=U, Vh=Vh):
                y = Vh.T @ np.r_[u, v]
                Gvv = (U.T @ jac(t, y) @ Vh.T)[s_v, s_v]
                self.njev -= 1                  # don't count these evaluations
                return Gvv

            solC = root(funC, v0, jac=jacC)
            if not solC.success:
                raise ValueError(
                    "Cannot find consistent initial conditions."
                    " Try to give a better y0")
            v = solC.x
            y = Vh.T @ np.r_[u, v]
            f = fun(t, y)
            J = jac(t, y)              # J at updated y0
            self.njev -= 1             # don't count this evaluation
            if not np.allclose(y, y0, rtol=self.rtol, atol=self.atol):
                warn(f"\nInitial conditions are changed to y0 = {y} to"
                     "\nmake them consistent with the algebraic constraints."
                     "\nThis is not updated in OdeResults.y if t_eval is None."
                     "\nCall solve_ivp again with a consistent y0 if this is"
                     "\na problem.")

        # find consistent derivatives yp0,
        # important for interpolation of the first step
        # start with numerical time derivative of fun calculated by h_start
        b = t + self.direction * min(abs(self.t_bound - t), self.max_step)
        fdot = h_start(fun, t, b, y, f, None,
                       self.rtol, self.atol, returnT=True)
        gdot = U.T @ fdot
        gudot, gvdot = gdot[s_u], gdot[s_v]
        g = U.T @ f
        gu = g[s_u]         # gv = g[s_v]
        G = U.T @ J @ Vh.T
        Guu, Guv = G[s_u, s_u], G[s_u, s_v]
        Gvv, Gvu = G[s_v, s_v], G[s_v, s_u]
        udot = gu/s[s_u]    # fixed
        vdot = -np.linalg.solve(Gvv, gvdot + Gvu @ udot)
        ydot = Vh.T @ np.r_[udot, vdot]
        # Effective Jac and T for problem when reduced to an ODE
        # h_start can use these.
        S = Guv @ np.linalg.solve(Gvv, Gvu)
        Tr = np.diag(1/s[s_u]) @ (gudot + Guv @ vdot)
        Jr = np.diag(1/s[s_u]) @ (Guu + S)
        kwargs_hstart = {'y': u, 'yprime': udot, 'J': Jr, 'T': Tr}

        if self.sparse:
            J = csc_array(J)
        return y, ydot, J, kwargs_hstart

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf, rtol=1e-3,
                 atol=1e-6, jac=None, jac_sparsity=None,
                 vectorized=False, first_step=None, sc_params=None,
                 jac_each_step=False, M=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized,
                         support_complex=True)
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.y)
        self.f = self.fun(self.t, self.y)   # temp
        if self.f.dtype != self.y.dtype:
            raise TypeError('dtypes of solution and derivative do not match')
        self.h_min_a, self.h_min_b = self._init_min_step_parameters()
        self.tiny_err = self.y.size**0.5 * np.finfo(self.y.dtype).eps**0.8
        order = min(self.order_secondary, self.order)
        self.error_exponent = -1 / (order + 1)
        self._init_sc_control(sc_params)

        self.K = np.empty((self.n_stages, self.n), self.y.dtype)
        self.h_previous = None
        self.y_old = None
        NFS[()] = 0                                # global failed step counter
        NFI[()] = 0                           # global failed iteration counter
        NLS[()] = 0                       # global linear system solves counter

        # add these checks to tests for ESDIRK and remove here
        for _d in np.diag(self.A)[2:]:
            assert _d == self.d, \
                "diagonal must have same entries, except A[0, 0] == 0"
        assert self.A[0, 0] == 0., "first stage should be explict"
        assert self.C[0] == 0., "first stage should be explict"
        assert np.all(self.A[-1, :] == self.B), " B should equal A[-1, :]"

        # jac matters
        # J is considered current if it is determined at the start of the
        # current step, not at (t_stage, y_predict). I chooose to use the same
        # J throughout the step. This has the disadvantage of lost work if the
        # last stage fails. The advantage is reduced complexity and quicker
        # stepsize reduction.
        self.current_J = True
        self.jac_each_step = jac_each_step
        self.jac_factor = None      # (for numerical jac approximation)
        self.jac, self.J = self._validate_jac(jac, jac_sparsity)
        self.sparse = issparse(self.J)
        self.linear = self.jac is None
        self.Rate = -np.inf                         # maximum of stage rates
        self.lu, self.solve_lu = self._set_lu_functions(self.J)
        self.LU = None
        self.h_LU = None
        self.Sc = eye_array(self.n)       # matrix to scale algebraic equations
        self.Niter = 0
        self.M_mat, self.M_details, self.isDAE = self._handle_M(M)
        if self.isDAE:
            self.y, yp0, self.J, kwargs_hstart = self._consistent_ICs()
        else:
            M_mat = self.M_mat
            if issparse(M_mat):
                M_mat = M_mat.todense()
            LU_M = lu_factor(M_mat)
            yp0 = lu_solve(LU_M, self.f)
        self.yp = yp0

        # size of first step:
        if first_step is not None:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        else:
            b = self.t + self.direction * min(
                abs(self.t_bound - self.t), self.max_step)
            if self.isDAE:
                # use only the ODE portion of the DAE as calculated in
                # _consistent_ICs.
                self.h_abs = abs(h_start(
                    fun, self.t, b, morder=order,
                    rtol=self.rtol, atol=self.atol, **kwargs_hstart))
            else:

                def fun_ext(t, y, LU_M=LU_M):
                    """explicit function for yp including M solve"""
                    return lu_solve(LU_M, self.fun_single(t, y))

                # Using the iterations seems to give a better estimate than
                # supplying the possibly ill conditioned J to h_start.
                self.h_abs = abs(h_start(
                    fun_ext, self.t, b, self.y, yp0,
                    order, self.rtol, self.atol))

    def _step_impl(self):
        t = self.t
        y = self.y
        K = self.K
        h_abs, min_step = self._reassess_stepsize(t, self.h_abs)

        # first stage
        K[0, :] = self.yp                                # smoothed first stage
        # K[0, :] = solve(M, self.fun(t, y))             # explicit first stage

        # evaluate LU for changed step size
        self._preemptive_lu_and_jac(h_abs, t, y, self.Niter)
        LU = self.LU
        Sc = self.Sc

        # loop until the step is accepted
        step_accepted = False
        step_rejected = False
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h

            if (LU is None) or self.jac_each_step or (
                                            self.linear and (h != self.h_LU)):
                self.h_LU = h
                if not self.isDAE:
                    LU = self.lu(self.M_mat - h * self.d * self.J)
                else:
                    # scale the algebraic equations by 1/(hd).
                    s_AE = self.M_details['s_AE']
                    sc = np.ones(self.n)
                    sc[s_AE] = 1/(h*self.d)
                    U = self.M_details['svd'][0]
                    Sc = U @ diags_array(sc) @ U.T
                    LU = self.lu(Sc @ (self.M_mat - h * self.d * self.J))

            self.Rate = -np.inf
            self.Niter = 0
            for s in range(1, self.n_stages):
                t_stage = t + self.C[s] * h
                psi = y + h*(K[:s, :].T @ self.A[s, :s])
                z_predict = h*(K[:s, :].T @ self.Az[s, :s])

                converged, z, rate, niter = self._solve_implicit_stage(
                    t_stage, z_predict, h, psi, y, LU, self.M_mat, Sc)
                self.Rate = max(rate, self.Rate)
                self.Niter = max(niter, self.Niter)
                if not converged:
                    break                           # retry step from the start

                # stage converged, store result
                K[s] = z/h                                # don't evaluate fun!

            if not converged:
                NFI[()] += 1
                if not self.current_J:                            # 1. update J
                    self.J = self.jac(t, y)
                    self.current_J = True
                    LU = None
                    continue
                else:                                      # 2. reduce stepsize
                    factor = MAX_RATE/self.Rate
                    h_abs *= max(MIN_FACTOR, min(factor, MAX_FACTOR_NRF))
                    LU = None
                    # stepsize reduction interacts with sc controller
                    step_rejected = True
                    self.standard_sc = True
                    continue

            # all implicit stage iterations have converged
            # calculate solution and error norm
            # y_new = y + h * (K.T @ self.B)                   # less efficient
            y_new = psi + self.d*z
            scale = calculate_scale(self.atol, self.rtol, y, y_new)
            err = h * (K.T @ self.E)                                      # est
            if self.filter_error:
                err = self.M_mat @ self.solve_lu(LU, Sc @ err)            # Est
            error_norm = norm(err/scale)

            # propose new stepsize
            step_accepted, h_abs = self._assess_error_and_stepsize(
                error_norm, h_abs, step_rejected)
            step_rejected = not step_accepted

        # step accepted, set or store for next step
        self.y_old = y.copy()
        self.yp_old = self.f.copy()
        self.yp = K[-1, :].copy()
        self.error_norm_old = error_norm
        self.h_previous = h
        self.h_abs = h_abs
        self.LU = LU
        self.Sc = Sc
        self.current_J = self.jac is None

        # output
        self.t = t_new
        self.y = y_new
        return True, None

    def _preemptive_lu_and_jac(self, h_abs, t, y, niter):
        if self.jac_each_step and not self.current_J:
            self.J = self.jac(t, y)                           # calculate new J
            self.current_J = True
            self.LU = None                                      # demand new LU
        elif self.Rate > 0:
            h = h_abs*self.direction
            h_ratio = h/self.h_previous
            h_ratio_LU = h/self.h_LU
            # ALT
            rate_predict = self.Rate*h_ratio
            rate_predict_LU = abs(h_ratio_LU - 1)
            rate_predict_JAC = rate_predict - rate_predict_LU
            if niter > 2 and rate_predict_JAC > MAX_RATE:
                self.J = self.jac(t, y)                       # calculate new J
                self.LU = None                                     # and new LU
            elif rate_predict_LU > MAX_RATE:
                self.LU = None                                  # demand new LU

    def _assess_error_and_stepsize(self, error_norm, h_abs, step_rejected):
        # evaluate error
        if error_norm < 1:
            step_accepted = True
            if error_norm < self.tiny_err:
                factor = self.max_factor
                self.standard_sc = True
            elif self.standard_sc:
                factor = min(self.safety * error_norm**self.error_exponent,
                             self.max_factor)
                if self.max_factor == MAX_FACTOR:
                    self.standard_sc = False
            else:
                # use second order controller
                h_ratio = h_abs*self.direction/self.h_previous
                factor = self.safety_sc*(
                        error_norm**self.minbeta1 *
                        self.error_norm_old**self.minbeta2 *
                        h_ratio**self.minalpha)
                # don't limit factor from second order controller?
                factor = max(self.min_factor, min(factor, self.max_factor))
            if step_rejected:
                factor = min(1, factor)
                self.standard_sc = True
        else:
            step_accepted = False
            NFS[()] += 1
            factor = max(self.safety * error_norm**self.error_exponent,
                         self.min_factor)
            self.standard_sc = True

        # reduce max_factor when on scale
        if factor < MAX_FACTOR:
            self.max_factor = MAX_FACTOR

        # new step size
        h_abs *= factor
        return step_accepted, h_abs

    def _reassess_stepsize(self, t, h_abs):
        # limit step size
        min_step = max(self.h_min_a * (abs(t) + h_abs), self.h_min_b)
        if h_abs < min_step or h_abs > self.max_step:
            h_abs = min(self.max_step, max(min_step, h_abs))
            self.standard_sc = True

        # handle final integration steps
        d = abs(self.t_bound - t)                          # remaining interval
        if (abs(d/h_abs - 1) < 1e-2) or (d < h_abs):
            # d <= h_abs: Don't step over t_bound or stop just below it.
            h_abs = d

        return h_abs, min_step

    def _solve_implicit_stage(self, t_stage, z_predict, h, psi, y, LU, M, Sc):
        z = z_predict.copy()
        dz_norm_old = -0.       # not used, only defined to satisfy linter
        rate = -np.inf
        converged = False
        for k in range(NEWTON_MAXITER):
            # evaluate fun with current z
            y_predict = psi + self.d*z
            f = self.fun(t_stage, y_predict)
            if not np.all(np.isfinite(f)):
                break

            # calculate residual and update
            z_residual = h*f - M @ z
            z_update = self.solve_lu(LU, Sc @ z_residual)               # solve
            z += z_update
            scale = calculate_scale(self.atol, self.rtol, y, y_predict)
            dz_norm = norm(z_update/scale)

            # assess convergence
            if self.linear:
                # linear system solves directly (if LU and J are current)
                assert self.current_J and (h == self.h_LU), \
                    "J and LU must be current for direct linear system solve"
                return True, z, rate, 1
            if dz_norm <= self.tiny_err:
                # good prediction, loose tolerance, edge case
                # rate accuracy limited by machine precision
                converged = True
                break

            # Cannot evaluate rate in first iteration.
            # Therefore main assessment starts after k == 0.
            if k:
                if rate < 0 or dz_norm_old > self.kappa:
                    rate = max(rate, dz_norm/dz_norm_old)
                if (rate >= 1) or (dz_norm*rate**(NEWTON_MAXITER - k)
                                   >= self.kappa*(1 - rate)):
                    # divergence or
                    # unlikely convergence in remaining iterations
                    break
                if dz_norm * rate < self.kappa * (1 - rate):
                    # normal convergence (most common case)
                    converged = True
                    break

            # update for next iteration
            dz_norm_old = dz_norm

        return converged, z, rate, k+1    # (k+1 = number of iterations)

    def _dense_output_impl(self):
        """return denseOutput, revert to cubic hermite interpolant if no
        specific interpolant is implemented.
        """
        if isinstance(self.P, np.ndarray):
            # normal output
            Q = self.K.T @ self.P
            return HornerDenseOutput(self.t_old, self.t, self.y_old, Q)

        # if no interpolant is implemented
        return CubicDenseOutput(
            self.t_old, self.t, self.y_old, self.y, self.yp_old, self.yp)

    def _estimate_error(self, K, h):
        """For unit test only"""
        err = h * (K.T @ self.E)                                          # est
        return err

    def _estimate_error_norm(self, K, h, scale):
        """For unit test only"""
        return norm(self._estimate_error(K, h) / scale)
