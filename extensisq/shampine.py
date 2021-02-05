import numpy as np
from warnings import warn
from math import copysign, sqrt
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
from extensisq.common import h_start, LinearDenseOutput, NFS
from scipy.integrate._ivp.common import (
    validate_max_step, validate_tol, norm, warn_extraneous,
    validate_first_step)


class SWAG(OdeSolver):
    """Linear multistep Adams method for non-stiff problems by L.F. Shampine,
    H.A. Watts, and M.K. Gordon.

    This is a variable order, variable stepsize method. The predictor is an
    Adams Bashforth method of order k, and the corrector is a Adams Moulton
    method of order k+1. The maximum value of k is 12 (by default).

    The original Fortran code is DDEABM [1]_, a descendant of [2]_, with a
    smooth, C1 continuous interpolant for dense output [3]_. This method is
    similar to `ode113` in Matlab.

    Other characteristics are: local extrapolation, variable coefficients, PECE
    mode, scaled divided differences, functional iteration.

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
    k_max : int, optional
        The maximum k (order - 1) used in the solver. The default is 12.

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
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.

    References
    ----------
    .. [1] Slatec Fortran code ddeabm.f and dependencies, in particular
           dsteps.f and dintp.f.
           https://www.netlib.org/slatec/src/
    .. [2] L.F. Shampine and M.K. Gordon, "Computer solution of ordinary
           differential equations: The initial value problem", San Francisco,
           W.H. Freeman.
    .. [3] H.A. Watts and L.F. Shampine, "Smoother Interpolants for Adams
           Codes",  SIAM Journal on Scientific and Statistical Computing, 1986,
           Vol. 7, No. 1, pp. 334-345. ISSN 0196-5204
           https://doi.org/10.1137/0907022.
    """

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf, rtol=1e-3,
                 atol=1e-6, vectorized=False, first_step=None, k_max=12,
                 **extraneous):
        if not (isinstance(k_max, int) and k_max > 0 and k_max < 13):
            raise ValueError("`k_max` should be an integer between 1 and 12.")
        warn_extraneous(extraneous)
        super(SWAG, self).__init__(
            fun, t0, y0, t_bound, vectorized, support_complex=True)
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)

        # starting step size
        self.yp = self.fun(self.t, self.y)                 # initial evaluation
        if first_step is None:
            self.h = h_start(self.fun, self.t, self.t_bound, self.y, self.yp,
                             1, self.rtol, self.atol)
        else:
            h_abs = validate_first_step(first_step, t0, t_bound)
            self.h = copysign(h_abs, self.direction)

        # constants
        small = np.nextafter(np.finfo(self.y.dtype).epsneg, 1)
        self.twou = 2.0 * small
        self.fouru = 4.0 * small
        self.two = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0,
                    1024.0, 2048.0, 4096.0, 8192.0)
        self.gstr = (0.5, 0.0833, 0.0417, 0.0264, 0.0188, 0.0143, 0.0114,
                     0.00936, 0.00789, 0.00679, 0.00592, 0.00524, 0.00468)
        iq = np.arange(1, k_max + 2)
        self.iqq = 1.0 / (iq * (iq + 1))                                # added
        self.k_max = k_max                                              # added
        self.eps = 1.0                                       # tolerances in wt
        self.p5eps = 0.5                                     # tolerances in wt

        # allocate arrays
        self.phi = np.empty((self.n, k_max + 2), self.y.dtype, 'F')
        self.psi = np.empty(k_max)
        self.alpha = np.empty(k_max)
        self.beta = np.empty(k_max)
        self.sig = np.empty(k_max + 1)
        self.v = np.empty(k_max)
        self.w = np.empty(k_max)
        self.g = np.empty(k_max + 1)
        self.gi = np.empty(k_max - 1)
        self.iv = np.zeros(max(0, k_max - 2), np.short)

        # Tolerances are dealt with like in scipy: wt is like scipy's scale
        # and will be update each step.  This is only the initial value:
        self.wt = self.atol + self.rtol * np.maximum(
            np.abs(self.y), np.abs(self.y - self.h*self.yp))

        # initialization
        # from  *** block 0 ***  of dsteps.f, under IF START:
        _round = 0.0
        if self.y.size:                            # to pass scipy's unit tests
            _round = self.twou * norm(self.y / self.wt)
        if self.p5eps < 100.0 * _round:
            # The compensated summation of the original code that would be
            # executed if nornd == False has been removed.  Instead, this
            # warning is given to the user.
            warn("numerical rounding may limit accuracy at this tolerance.")
        self.phi[:, 0] = self.yp
        self.phi[:, 1] = 0.0
        self.sig[0] = 1.0
        self.g[0] = 1.0
        self.g[1] = 0.5
        self.hold = 0.0
        self.k = 1
        self.kold = 0
        self.kprev = 0
        self.phase1 = True
        self.ivc = 0
        self.kgi = 0
        self.ns = 0

        # from ddes.f, for stiffness detection
        self.stiff = False
        self.kle4 = 0

    def _step_impl(self):

        # current state
        x = self.t
        y = self.y.copy()
        self.y_old = self.y                           # added, for dense output

        # load variables (hold != self.step_size, rounding matters)
        (hold, h, wt, k, kold, phi, yp, psi, alpha, beta, sig, v, w, g,
         phase1, ns, kprev, ivc, iv, kgi, gi, gstr, iqq, eps, p5eps) = (
          self.hold, self.h, self.wt, self.k, self.kold, self.phi, self.yp,
          self.psi, self.alpha, self.beta, self.sig, self.v, self.w, self.g,
          self.phase1, self.ns, self.kprev, self.ivc, self.iv, self.kgi,
          self.gi, self.gstr, self.iqq, self.eps, self.p5eps)

        # from *** ddes.f ***
        min_step = self.fouru * abs(x)                                  # added

        # stiffness detection
        if kold > 4:
            self.kle4 = 0
        else:
            self.kle4 += 1
            if self.kle4 > 50 and not self.stiff and self.k_max > 4:
                # This warning is issued once, after 50 consequtive steps are
                # taken with order <= 4, while k_max > 4.
                self.stiff = True
                warn("problem appears to be stiff (for this tolerance).")

        # extrapolate if too close to t_bound
        d = self.t_bound - x
        if abs(d) <= min_step:
            self.kold = 0                                    # for dense output
            y[:] += d * yp
            # ouput
            self.t = self.t_bound
            self.y = y
            return True, None

        # don't allow to step over t_bound
        if self.direction * (h - d) > 0:
            h = d

        # limit h to max_step
        if self.max_step != np.inf:
            h = min(self.max_step, abs(h))
            h = copysign(h, self.direction)

        # (***first executable statement dsteps)
        if abs(h) < min_step:
            h = copysign(min_step, h)
            return False, self.TOO_SMALL_STEP

        # If error tolerance is too small, increase it to an acceptable value
        # or rather terminate the integration with an error
        _round = self.twou * norm(y / wt)
        if p5eps < _round:
            eps = 2.0 * _round * (1.0 + self.fouru)
            return False, ("tolerance too tight.\n"
                           f"suggested minimal increase factor: {eps}")

        ifail = 0

        # ***     begin block 1     ***
        # Compute coefficients of formulas for this step.  Avoid computing
        # those quantities not changed when step size is not changed.

        while True:
            kp1 = k + 1
            km1 = k - 1
            km2 = k - 2

            # ns is the number of dsteps taken with size h, including the
            # current one.  When k < ns, no coefficients change
            if h != hold:
                ns = 0
            if ns <= kold:
                ns += 1

            if k >= ns:
                # Compute those components of alpha(*), beta(*), psi(*), sig(*)
                # which are changed
                nsm1 = ns - 1                                           # added
                psi_old = psi[nsm1:km1].copy()                          # added
                psi[nsm1] = h * ns
                alpha[nsm1] = 1.0 / ns
                beta[nsm1] = 1.0
                sig[ns] = 1.0
                for i, temp2 in enumerate(psi_old, start=ns):
                    temp1 = h + temp2
                    alp = h / temp1                                     # added
                    psi[i] = temp1
                    alpha[i] = alp
                    beta[i] = beta[i-1] * psi[i-1] / temp2
                    sig[i+1] = (i + 1) * alp * sig[i]

                # compute coefficients g(*)

                # initialize v(*) and set w(*).
                if ns == 1:
                    w[:k] = v[:k] = iqq[:k]
                    ivc = kgi = 0
                    if k != 1:
                        kgi = 1
                        gi[0] = w[1]
                else:
                    # if order was raised, update diagonal part of v(*)
                    if k > kprev:
                        if ivc != 0:
                            ivc -= 1
                            jv = kp1 - iv[ivc]
                        else:
                            jv = 1
                            w[km1] = v[km1] = iqq[km1]
                            if k == 2:
                                kgi = 1
                                gi[0] = w[1]
                        for j, alp in enumerate(alpha[jv:nsm1], start=jv):
                            i = km1 - j
                            v[i] -= alp * v[i+1]
                            w[i] = v[i]
                        if k == ns and jv < nsm1:
                            kgi = nsm1
                            gi[kgi-1] = w[1]
                    # update v(*) and set w(*)
                    limit1 = kp1 - ns
                    v[:limit1] -= alpha[nsm1] * v[1:limit1+1]
                    w[:limit1+1] = v[:limit1+1]
                    g[ns] = w[0]
                    if limit1 != 1:
                        kgi = ns
                        gi[nsm1] = w[1]
                    if k < kold:
                        iv[ivc] = limit1 + 2
                        ivc += 1

                # compute the g(*) in the work vector w(*)
                kprev = k
                for i, alp in enumerate(alpha[ns:k], start=ns):
                    limit2 = k - i
                    w[:limit2] -= alp * w[1:limit2+1]
                    g[i+1] = w[0]

            # ***     end block 1     ***

            # ***     begin block 2     ***
            # Predict a solution p(*), evaluate derivatives using predicted
            # solution, estimate local error at order k and errors at orders
            # k, k-1, k-2 as if constant step size were used.

            # change phi to phi star
            phi[:, ns:k] *= beta[ns:k]

            # predict solution and differences
            phi[:, kp1] = phi[:, k]
            phi[:, k] = 0.0
            p = h * (phi[:, :k] @ g[:k]) + y
            for i in range(k, 0, -1):
                phi[:, i-1] += phi[:, i]
            xold = x
            x += h
            absh = abs(h)
            yp[:] = self.fun(x, p)                                   # evaluate

            # added update of wt:
            wt[:] = self.atol + self.rtol * np.maximum(np.abs(p), np.abs(y))

            # estimate errors at orders k, k-1, k-2
            temp3 = 1.0 / wt
            temp4 = yp - phi[:, 0]
            if k > 2:
                erkm2 = absh * norm((phi[:, km2] + temp4) * temp3)
                erkm2 *= sig[km2] * gstr[km2-1]
            if k > 1:
                erkm1 = absh * norm((phi[:, km1] + temp4) * temp3)
                erkm1 *= sig[km1] * gstr[km2]
            erk = absh * norm(temp4 * temp3)
            err = erk * (g[km1] - g[k])
            erk *= sig[k] * gstr[km1]

            # test if order should be lowered
            knew = k
            if k > 2 and max(erkm1, erkm2) < erk:
                knew = km1
            elif k == 2 and erkm1 < 0.5 * erk:
                knew = km1

            # test if step successful
            if err <= eps:
                # success
                break
            # else: failure

            # ***     end block 2     ***

            # ***     begin block 3     ***
            # The step is unsuccessful.  restore x, phi(*,*), psi(*). if third
            # consecutive failure, set order to one.  If step fails more than
            # three times, consider an optimal step size.  Double error
            # tolerance and return if estimated step size is too small for
            # machine precision.

            # restore x, phi(*,*) and psi(*)
            phase1 = False
            x = xold
            phi[:, :k] -= phi[:, 1:kp1]
            phi[:, :k] /= beta[:k]
            psi[:km1] = psi[1:k] - h

            # On third failure, set order to one.
            # Thereafter, use optimal step size.
            NFS[()] += 1
            ifail += 1
            temp2 = 0.5
            if ifail >= 4 and p5eps < 0.25 * erk:
                temp2 = sqrt(p5eps / erk)
            if ifail >= 3:
                knew = 1
            h *= temp2
            k = knew
            ns = 0
            if abs(h) < min_step:
                return False, self.TOO_SMALL_STEP

            # ***     end block 3     ***

        # end while loop

        # ***     begin block 4     ***
        # The step is successful.  Correct the predicted solution, evaluate the
        # derivatives using the corrected solution and update the differences.
        # Determine best order and step size for next step.

        kold = k
        hold = h

        # correct and evaluate
        y[:] = h * g[k] * (yp - phi[:, 0]) + p
        yp[:] = self.fun(x, y)                                       # evaluate
        # p does not need to store y_old for dense output anymore.

        # update differences for next step
        phi[:, k] = yp - phi[:, 0]
        phi[:, kp1] = phi[:, k] - phi[:, kp1]
        phi[:, :k] += phi[:, k, np.newaxis]

        # Estimate error at order k+1 unless:
        #   - in first phase when always raise order,
        #   - already decided to lower order,
        #   - step size not constant so estimate unreliable
        if knew == km1 or k == self.k_max:
            phase1 = False
        erkp1 = 0.0
        if phase1:
            # raise order
            k = kp1
            erk = erkp1
        elif knew == km1:
            # lower order, as already decided in block 2
            k = km1
            erk = erkm1
        elif k < ns:
            erkp1 = gstr[k] * absh * norm(phi[:, kp1] / wt)
            # Using estimated error at order k+1, determine appropriate order
            # for next step
            if k == 1:
                if erkp1 < 0.5 * erk and k < self.k_max:
                    # raise order
                    k = kp1
                    erk = erkp1
                # else: no order change
            elif erkm1 <= min(erk, erkp1):
                # lower order
                k = km1
                erk = erkm1
            elif not (erkp1 > erk or k == self.k_max):
                # Here erkp1 < erk < max(erkm1, erkm2) else order would
                # have been lowered in block 2.  Thus order is to be raised
                k = kp1
                erk = erkp1
            # else: no order change
        # else: no order change

        # With new order determine appropriate step size for next step
        if phase1 or p5eps >= erk * self.two[k]:
            hnew = h + h
        elif p5eps >= erk:
            # keep step size (double, or don't increase at all)
            hnew = h
        else:
            # calculate reduced step size
            r = (p5eps / erk) ** (1.0 / (k + 1))
            hnew = absh * max(0.5, min(0.9, r))
            hnew = copysign(max(hnew, min_step), h)
        h = hnew

        # ***     end block 4     ***

        # output
        self.t = x
        self.y = y

        # store the non-mutable variables for the next step:
        (self.h, self.hold, self.k, self.kold, self.phase1, self.ns,
         self.kprev, self.ivc, self.kgi) = (
            h, hold, k, kold, phase1, ns, kprev, ivc, kgi)
        return True, None

    def _dense_output_impl(self):
        x = self.t
        ox = self.t_old
        y = self.y
        oy = self.y_old
        kold = self.kold
        if kold:
            return SwagDenseOutput(
                x, y, kold, self.phi, self.ivc, self.iv, self.kgi, self.gi,
                self.alpha, self.g, self.w, ox, oy, self.iqq)
        else:
            # for the rare cases, in which the last step is tiny,
            # and therefore extrapolated.
            return LinearDenseOutput(ox, x, oy, y)


class SwagDenseOutput(DenseOutput):
    def __init__(self, x, y,
                 kold, phi, ivc, iv, kgi, gi, alpha, og, ow, ox, oy, iqq):
        super(SwagDenseOutput, self).__init__(ox, x)

        # compute the double integral term gdi
        if kold <= kgi:
            gdi = gi[kold-1]
        else:
            if ivc == 0:
                gdi = iqq[kold]
                m = 1
            else:
                iw = iv[ivc-1]
                gdi = ow[iw-1]
                m = kold - iw + 2
            for i in range(m, kold):
                gdi = ow[kold-i] - alpha[i] * gdi
        # and gdif, vector here, scalar in original code
        gdif = np.diff(og[:kold+1], prepend=0.0)                          # vec

        # store data
        (self.y, self.kold, self.phi, self.alpha, self.gdif, self.oy, self.iqq,
         self.gdi) = (y, kold, phi[:, :kold+1].copy(), alpha[1:kold].copy(),
                      gdif, oy, iqq[:kold+1], gdi)

    def _call_impl(self, t):
        # interpolation of derivative is deactivated, because it is unsupported
        # in scipy.  Nevertheless, it should work if all lines marked "prime"
        # are uncommented.

        # load data (phi, alpha and iqq were reduced)
        x, y, kold, phi, alpha, gdif, ox, oy, iqq, gdi = (
            self.t, self.y, self.kold, self.phi, self.alpha, self.gdif,
            self.t_old, self.oy, self.iqq, self.gdi)

        # work arrays
        g = np.empty(kold + 1)
        # c = np.empty(kold + 1)                                        # prime

        # interpolate point by point
        yout_array = np.empty((y.size, t.size), y.dtype, 'F')
        # ypout_array = np.empty_like(yout_array)                       # prime
        for it, xout in enumerate(np.atleast_1d(t)):

            # ***first executable statement dintp
            hi = xout - ox
            h = x - ox
            xi = hi / h
            xim1 = xi - 1.0

            # initialize w(*) for computing g(*)
            w = xi * (np.cumprod(np.full(kold + 1, xi)) * iqq)

            # compute g(*) and c(*)
            g[0] = xi
            g[1] = 0.5 * xi * xi
            # c[0] = 1.0                                                # prime
            # c[1] = xi                                                 # prime
            for i, alp in enumerate(alpha):
                lim = kold - i
                gamma = 1.0 + xim1 * alp
                w[:lim] = gamma * w[:lim] - alp * w[1:lim+1]
                g[i+2] = w[0]
                # c[i+2] = gamma * c[i+1]                               # prime

            # define interpolation parameters
            sigma = (w[1] - xim1 * w[0]) / gdi
            # rmu = xim1 * c[kold] / gdi                                # prime
            # hmu = rmu / h                                             # prime

            # interpolate for the solution -- yout
            g[:] = np.diff(g, prepend=0.0)
            yout = h * (phi @ (g - sigma * gdif))
            yout += sigma * y + (1.0 - sigma) * oy

            # and for the derivative of the solution -- ypout
            # c[:] = np.diff(c, prepend=0.0)                            # prime
            # ypout = phi @ (c + rmu * gdif)                            # prime
            # ypout += hmu * (oy - y)                                   # prime

            # store solution in output array
            yout_array[:, it] = yout
            # ypout_array[:, it] = ypout                                # prime

        # need this if to pass scipy's unit tests.
        if t.shape:
            return yout_array
        else:
            return yout_array[:, 0]
