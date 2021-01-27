import numpy as np
from math import sqrt, copysign
from scipy.integrate._ivp.common import (
    validate_max_step, validate_tol, norm, validate_first_step,
    warn_extraneous)
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
from scipy.integrate._ivp.rk import SAFETY, MIN_FACTOR, MAX_FACTOR


NFS = np.array(0)                                         # failed step counter


class RungeKutta(OdeSolver):
    """Modified RungeKutta class for conventional and FSAL explicit methods.

    This implementation mainly follows the scipy implementation. The current
    differences are:
      - Conventional (non FSAL) methods are detected and failed steps cost
        one function evaluation less than with the scipy implementation.
      - Linear extrapolation is used in the rare case in which a step ends
        very close to the integration bound.
      - A different, more elaborate estimate for the size of the first step
        is used.
      - Horner's rule is used for dense output calculation.
      - a failed step counter is added.
    """

    # ### implement ###
    # n_stages : int
    #    effective number of stages
    n_stages = NotImplemented

    # order : int
    #    order of the main method
    order = NotImplemented

    # error_estimator_order : int
    #    order of the secondary embedded method
    error_estimator_order = NotImplemented

    # A : ndarray[n_stages, n_stages]
    #    runge kutta coefficient matrix
    A = NotImplemented

    # B : ndarray[n_stages]
    #    output coefficients (weights)
    B = NotImplemented

    # C : ndarray[n_stages]
    #    time fraction coefficients (nodes)
    C = NotImplemented

    # E : ndarray[n_stages + 1]
    #    error coefficients (weights Bh - B)
    E = NotImplemented

    # P : ndarray[n_stages + 1, order_polynomial]
    #    interpolation coefficients
    P = NotImplemented

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf, rtol=1e-3,
                 atol=1e-6, vectorized=False, first_step=None, **extraneous):
        # mostly follows the scipy implementation of RungeKutta
        warn_extraneous(extraneous)
        super(RungeKutta, self).__init__(fun, t0, y0, t_bound, vectorized,
                                         support_complex=True)
        self.FSAL = 1 if self.E[self.n_stages] else 0
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)
        if first_step is None:
            self.h_abs = abs(h_start(self.fun, self.t, self.t_bound, self.y,
                                     self.f, self.order, self.rtol, self.atol))
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.K = np.empty((self.n_stages + 1, self.n), self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None
        NFS[()] = 0                                 # reset failed step counter

    def _step_impl(self):
        # mostly follows the scipy implementation of RungeKutta
        t = self.t
        y = self.y

        # limit step size
        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        h_abs = min(self.max_step, max(min_step, self.h_abs))

        # use extrapolation in the rare cases where the previous step ended
        # too close to t_bound.
        d = abs(self.t_bound - t)
        if d < min_step:
            h = self.t_bound - t
            y_new = y + h * self.f
            self.h_previous = h
            self.y_old = y
            self.t = self.t_bound
            self.y = y_new
            self.f = None                          # used by _dense_output_impl
            return True, None

        # don't step over t_bound
        h_abs = min(h_abs, d)

        # loop until step accepted
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
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)
                if step_rejected:
                    factor = min(1, factor)
                h_abs *= factor

                if not self.FSAL:
                    # evaluate ouput point for interpolation and next step
                    self.K[self.n_stages] = self.fun(t + h, y_new)
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
        return True, None

    def _estimate_error(self, K, h):
        # exclude K[-1] if not FSAL. It could contain nan or inf
        return h * (K[:self.n_stages + self.FSAL].T @
                    self.E[:self.n_stages + self.FSAL])

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _comp_sol_err(self, y, h):
        # compute solution and error norm of step
        y_new = y + h * (self.K[:self.n_stages].T @ self.B)
        scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_new))
        if self.FSAL:
            # do evaluation, only if needed for error estimate
            self.K[self.n_stages, :] = self.fun(self.t + h, y_new)
        error_norm = self._estimate_error_norm(self.K, h, scale)
        return y_new, error_norm

    def _rk_stage(self, h, i):
        # compute a single RK stage
        dy = h * (self.K[:i, :].T @ self.A[i, :i])
        self.K[i] = self.fun(self.t + self.C[i] * h, self.y + dy)

    def _dense_output_impl(self):
        if self.f is None:
            # output was extrapolated linearly
            return LinearDenseOutput(self.t_old, self.t, self.y_old, self.y)
        # normal output
        Q = self.K.T @ self.P
        return HornerDenseOutput(self.t_old, self.t, self.y_old, Q)


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
    morder : integer
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

    Reference
    ---------
    .. [1] H.A. Watts, Starting step size for an ODE solver, Journal of
           Computational and Applied Mathematics, Vol. 9, No. 2, 1983,
           pp. 177-191, ISSN 0377-0427.
           https://doi.org/10.1016/0377-0427(83)90040-7
    .. [2] Fortran code dstrt.f from https://www.netlib.org/slatec/src/
    """

    # needed to pass scipy unit test:
    if y.size == 0:
        return np.inf

    # compensate for modified call list
    neq = y.size
    spy = np.empty_like(y)
    pv = np.empty_like(y)
    etol = atol + rtol * np.abs(y)

    # `SMALL` is a small positive machine dependent constant which is used for
    # protecting against computations with numbers which are too small relative
    # to the precision of floating point arithmetic. `SMALL` should be set to
    # (approximately) the smallest positive DOUBLE PRECISION number such that
    # (1. + SMALL) > 1.  on the machine being used. The quantity SMALL**(3/8)
    # is used in computing increments of variables for approximating
    # derivatives by differences.  Also the algorithm will not compute a
    # starting step length which is smaller than 100*SMALL*ABS(A).
    # `BIG` is a large positive machine dependent constant which is used for
    # preventing machine overflows. A reasonable choice is to set big to
    # (approximately) the square root of the largest DOUBLE PRECISION number
    # which can be held in the machine.
    BIG = sqrt(np.finfo(y.dtype).max)
    SMALL = np.nextafter(np.finfo(y.dtype).epsneg, 1.0)

    # following dhstrt.f from here
    dx = b - a
    absdx = abs(dx)
    relper = SMALL**0.375

    # compute an approximate bound (dfdxb) on the partial derivative of the
    # equation with respect to the independent variable.  protect against an
    # overflow.  also compute a bound (fbnd) on the first derivative locally.
    da = copysign(max(min(relper * abs(a), absdx), 100.0 * SMALL * abs(a)), dx)
    da = da or relper * dx
    sf = df(a + da, y)                                               # evaluate
    yp = sf - yprime
    delf = norm(yp)
    dfdxb = BIG
    if delf < BIG*abs(da):
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
        if delf >= BIG * abs(dely):
            # protect against an overflow
            dfdub = BIG
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
    tolmin = min(tolexp.min(), BIG)
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
    # 100*SMALL*abs(a).  however, if a=0. and the computed h underflowed to
    # zero, the algorithm returns SMALL*abs(b) for the step length.
    h = max(h, 100.0 * SMALL * abs(a))
    h = h or SMALL * abs(b)

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

        # need this `if` to pass scipy's unit tests. I'm not sure why.
        if t.shape:
            return y
        else:
            return y[:, 0]


class LinearDenseOutput(DenseOutput):
    """Linear interpolator.

    This class can be used if the output was obtained by extrapolation (if the
    end point was too close to perform a normal integration step).
    """
    def __init__(self, t_old, t, y_old, y):
        super(LinearDenseOutput, self).__init__(t_old, t)
        self.h = t - t_old
        self.y_old = y_old[:, np.newaxis]
        self.dy = (y - y_old)[:, np.newaxis]

    def _call_impl(self, t):
        t = np.atleast_1d(t)
        x = (t - self.t_old) / self.h

        # linear interpolation
        y = x * self.dy
        y += self.y_old

        # need this `if` to pass scipy's unit tests. I'm not sure why.
        if t.shape:
            return y
        else:
            return y[:, 0]


class CubicDenseOutput(HornerDenseOutput):
    """Cubic, C1 continuous interpolator
    """
    def __init__(self, t_old, t, y_old, y, f_old, f):
        # transform input
        h = t - t_old
        dy = (y - y_old)
        K = np.stack([f_old, dy/h, f])
        P = np.array([[1.0, -2.0, 1.0],
                      [0.0, 3.0, -2.0],
                      [0.0, -1.0, 1.0]])
        Q = K.T @ P
        super(CubicDenseOutput, self).__init__(t_old, t, y_old, Q)
