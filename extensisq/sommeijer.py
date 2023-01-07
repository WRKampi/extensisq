import numpy as np
from math import sqrt, sinh, cosh, log
from warnings import warn
from scipy.integrate._ivp.common import (
    validate_max_step, validate_first_step, warn_extraneous)
from scipy.integrate._ivp.base import OdeSolver
from extensisq.common import (validate_tol, CubicDenseOutput, NFS, norm,
                              calculate_scale)


# global counters, several were removed
nrejct = NFS                              # nr of rejected steps
nfesig = np.array(0)                      # nr of fun evals for rho estimation
maxm = np.array(0)                        # max nr of stages used


class SSV2stab(OdeSolver):
    """Stabilized Runge Kutta Chebyshev method of Sommeijer, Shampine and
    Verwer [1].

    This is a translation of the Fortran code rkc.f [2]. It a variable step
    size, variable formula code to explicity and efficiently solve a class of
    large systems of mildly stiff ordinary differential equations. The number
    of stages in this method is adapted in each step to stretch the stability
    region along the real axis as much as neccesary.

    This method is particularly suited for initial value problems arising from
    semi-discretization of diffusion-dominated parabolic partial differential
    equations. The accuracy of such problems is limited by the spatial
    discretization. Therefore the low (second) order temporal convergence of
    this method is appropriate. Scince this is an explicit method, no
    costly solves of large matrix vector equations are needed.

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
    const_jac : bool, optional
        If your problem has a constant Jacobian, then the spectral radius needs
        to be estimated only once. Setting const_jac=True will inform the
        method, resulting in a slight efficiency increase. Default: False.
    rho_jac : None or callable, optional
        If the upper bound of the spectral radius of your problem can be given
        in a simple, fast to evaluate expression, then you can inform
        method using a function with signature: sprad = rho_jac(t, y). This is
        more efficient than using power iterations to find a spectral radius
        estimate, as is done by default: rho_jac=None.

    References
    ----------
    .. [1] B.P. Sommeijer, L.F. Shampine, J.G. Verwer, "RKC: An explicit solver
           for parabolic PDEs", Journal of Computational and Applied
           Mathematics, Vol. 88, No. 2, 1998, pp. 315-326.
           https://doi.org/10.1016/S0377-0427(97)00219-7
    .. [2] Fortran code rkc.f.
           http://www.netlib.no/netlib/ode/
    """
    # the main modifications are marked with "# mod"

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf, rtol=1e-3,
                 atol=1e-6, vectorized=False, first_step=None,
                 const_jac=False, rho_jac=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(
            fun, t0, y0, t_bound, vectorized, support_complex=False)
        if first_step is None:
            self.absh = None
        else:
            self.absh = validate_first_step(first_step, t0, t_bound)
        self.hold = None
        if not isinstance(const_jac, bool):
            raise TypeError('`const_jac` should be True or False')
        if rho_jac is not None:
            if not callable(rho_jac):
                raise TypeError(
                    '`rho_jac` should be None or a function: '
                    '`sprad = rho_jac(t, y)`')
            elif not isinstance(rho_jac(self.t, self.y), float):
                raise TypeError('`rho_jac` should return a float')
            elif rho_jac(self.t, self.y) <= 0:
                raise ValueError('`rho_jac` should return a positive float')
        self.const_jac = const_jac
        self.rho_jac = rho_jac
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.y)
        self.uround = np.nextafter(np.finfo(self.y.dtype).epsneg, 1)
        self.sqrtu = sqrt(self.uround)
        self.sqrtmin = sqrt(np.finfo(self.y.dtype).tiny)
        self.W = np.empty((4, self.n), self.y.dtype)
        self.V = None       # eigenvector for spectral radius estimation. It
        #                     was WORK(5) in Fortran. Init in self._rho().

        # reset counters
        nrejct[()] = 0
        nfesig[()] = 0
        maxm[()] = 0
        self.nstsig = 0
        self.mlim = 0                            # added, for stiffness warning

        # Initialize on the first call.
        mmax = int(round(sqrt(self.rtol/(10.0 * self.uround))))
        self.mmax = max(mmax, 2)
        self.newspc = True
        self.jacatt = False
        self.W[0] = self.y
        self.W[1] = self.fun(self.t, self.y)                         # evaluate
        max_step = min(self.max_step, abs(self.t_bound - self.t))
        self.max_step = min(max_step, sqrt(np.finfo(self.y.dtype).max))
        hmin = abs(self.t)
        if self.t_bound != np.inf:
            hmin = max(hmin, abs(self.max_step))
        self.hmin = max(self.sqrtmin, 10.0 * self.uround * hmin)

    def _init_step_size(self, t, yn, fn, vtemp1, vtemp2):
        absh = self.max_step
        if self.sprad * absh > 1.0:
            absh = 1.0 / self.sprad
        absh = max(absh, self.hmin)
        vtemp1[:] = yn + absh * fn
        vtemp2[:] = self.fun(t + absh, vtemp1)                       # evaluate
        wt = self.atol + self.rtol * np.abs(yn)
        est = absh * norm((vtemp2 - fn) / wt)
        if 0.1 * absh < self.max_step * sqrt(est):
            absh = max(0.1 * absh/sqrt(est), self.hmin)
        else:
            absh = self.max_step
        return absh

    def _step_impl(self):
        """original: subroutine RKCLOW in rkc.f"""
        t = self.t
        absh = self.absh
        y = self.y.copy()
        yn, fn, vtemp1, vtemp2 = self.W
        one3rd = 1/3
        two3rd = 2/3

        # Start of loop for taking one step.
        while True:
            # Estimate the spectral radius of the Jacobian when newspc=True.
            if self.newspc:
                if self.rho_jac is not None:
                    self.sprad = self.rho_jac(t, yn)
                else:
                    self.sprad = self._rho(t, yn, fn, vtemp1, vtemp2)
                    if self.sprad is None:
                        return False, (
                            "The method to estimate the spectral radius "
                            "of the Jacobian did not converge")
                self.jacatt = True

            # Compute an initial step size.
            if absh is None:
                absh = self._init_step_size(t, yn, fn, vtemp1, vtemp2)

            #  Adjust the step size and determine the number of stages m.
            if 1.1 * absh >= abs(self.t_bound - t):
                absh = abs(self.t_bound - t)
            m = 1 + int(sqrt(1.54 * absh * self.sprad + 1.0))

            # Limit m to mmax to control the growth of roundoff error.
            if m > self.mmax:
                m = self.mmax
                absh = (m**2 - 1) / (1.54 * self.sprad)
                # added stiffness warning:
                self.mlim += 1
                if self.mlim == 15:
                    warn('Your problem is too stiff for this method.')
            else:
                self.mlim = 0
            maxm[()] = max(m, maxm[()])

            # A tentative solution at t+h is returned in y and its slope is
            # evaluated in vtemp1(*). Mod: a factor 4/3*(m**2-1) and a lower
            # bound are added to the calculation of hmin.
            h = self.direction * absh
            # hmin = 10.0 * self.uround * max(abs(t) + abs(t+h))     # original
            hmin = max(self.sqrtmin,
                       13.3*self.uround*(abs(t) + absh)*(m**2 - 1))       # mod
            self._stages(t, yn, fn, h, m, y, vtemp1, vtemp2)           # stages
            vtemp1[:] = self.fun(t + h, y)                           # evaluate

            # Estimate the local error and compute its weighted RMS norm.
            # original:
            wt = calculate_scale(self.atol, self.rtol, y, yn)
            est = 0.8 * (yn - y) + 0.4 * h * (fn + vtemp1)
            err = norm(est / wt)

            if err < 1.0:
                # Step is accepted.
                break
            else:
                # Step is rejected.
                if np.isnan(err) or np.isinf(err):
                    return False, "Overflow or underflow encountered."
                nrejct[()] += 1
                absh = 0.8 * absh / err**one3rd
                if absh < hmin:
                    return False, self.TOO_SMALL_STEP
                else:
                    self.newspc = not self.jacatt
                    self.absh = absh

        # Step is accepted.
        t += h
        self.jacatt = self.const_jac
        self.nstsig = (self.nstsig + 1) % 25
        self.newspc = False
        if self.rho_jac is not None or self.nstsig == 0:
            self.newspc = not self.jacatt

        # Update the data for interpolation stored in W(*).
        ylast = yn.copy()
        yplast = fn.copy()
        yn[:] = y
        fn[:] = vtemp1
        vtemp1[:] = ylast
        vtemp2[:] = yplast
        fac = 10.0
        if self.hold is None:
            temp2 = err**one3rd
            if 0.8 < fac * temp2:
                fac = 0.8 / temp2
        else:
            # H220 dead-beat control (Soederlind's label)
            temp1 = 0.8 * absh * self.errold**one3rd
            temp2 = abs(self.hold) * err**two3rd
            if temp1 < fac * temp2:
                fac = temp1 / temp2
        absh = max(0.1, fac) * absh
        self.absh = max(hmin, min(self.max_step, absh))
        self.errold = err
        self.hold = h

        # output
        self.y = y
        self.t = t
        return True, None

    def _stages(self, t, yn, fn, h, m, y, yjm1, yjm2):
        """Take a step of size h from t to t+h to get y(*).

        original: subroutine STEP in rkc.f"""

        w0 = 1.0 + 2.0 / (13.0 * m**2)
        temp1 = w0**2 - 1.0
        temp2 = sqrt(temp1)
        arg = m * log(w0 + temp2)
        w1 = sinh(arg) * temp1 / (cosh(arg) * m * temp2 - w0 * sinh(arg))
        bjm1 = 1.0 / (2.0 * w0)**2
        bjm2 = bjm1

        # Evaluate the first stage.
        yjm2[:] = yn
        mus = w1 * bjm1
        yjm1[:] = yn + h * mus * fn
        thjm2 = 0.0
        thjm1 = mus
        zjm1 = w0
        zjm2 = 1.0
        dzjm1 = 1.0
        dzjm2 = 0.0
        d2zjm1 = 0.0
        d2zjm2 = 0.0

        # Evaluate stages j = 2,...,m.
        for j in range(2, m + 1):
            zj = 2.0 * w0 * zjm1 - zjm2
            dzj = 2.0 * w0 * dzjm1 - dzjm2 + 2.0 * zjm1
            d2zj = 2.0 * w0 * d2zjm1 - d2zjm2 + 4.0 * dzjm1
            bj = d2zj / dzj**2
            ajm1 = 1.0 - zjm1 * bjm1
            mu = 2.0 * w0 * bj / bjm1
            nu = -bj / bjm2
            mus = mu * w1/w0

            # Use the y array for temporary storage here.
            y[:] = self.fun(t + h * thjm1, yjm1)                     # evaluate
            y[:] = (mu * yjm1 + nu * yjm2 + (1.0 - mu - nu) * yn +
                    h * mus * (y - ajm1 * fn))
            thj = mu * thjm1 + nu * thjm2 + mus * (1.0 - ajm1)

            # Shift the data for the next stage.
            if j < m:
                yjm2[:] = yjm1
                yjm1[:] = y
                thjm2 = thjm1
                thjm1 = thj
                bjm2 = bjm1
                bjm1 = bj
                zjm2 = zjm1
                zjm1 = zj
                dzjm2 = dzjm1
                dzjm1 = dzj
                d2zjm2 = d2zjm1
                d2zjm1 = d2zj

    def _rho(self, t, yn, fn, v, fv):
        """_rho() attempts to compute a close upper bound, SPRAD, on the
        spectral radius of the Jacobian matrix using a nonlinear power method.
        A convergence failure is reported returning None.

        original: subroutine RKCRHO in rkc.f
        """

        # sprad smaller than small = 1/hmax are not interesting because
        # they do not constrain the step size.
        small = 1.0 / self.max_step

        # The initial slope is used as first guess and thereafter the last
        # computed eigenvector.  Some care is needed to deal with special
        # cases. Approximations to the eigenvector are normalized so that their
        # Euclidean norm has the constant value dynrm.
        if self.V is None:
            self.V = fn.copy()
        v[:] = self.V
        ynrm = np.linalg.norm(yn)
        vnrm = np.linalg.norm(v)
        if ynrm != 0.0 and vnrm != 0.0:
            dynrm = ynrm * self.sqrtu
            v[:] = yn + v * (dynrm/vnrm)
        elif ynrm != 0.0:
            dynrm = ynrm * self.sqrtu
            v[:] *= 1.0 + self.sqrtu
        elif vnrm != 0.0:
            dynrm = self.uround
            v[:] *= dynrm/vnrm
        else:
            dynrm = self.uround
            v[:] = dynrm

        # Now iterate with a nonlinear power method.
        sigma = 0.0
        itmax = 50
        for iter in range(itmax):
            # evaluation with fun_single does not increment the nfev counter of
            # scipy. This is a convention for Jacobian estimation, which is not
            # unlike the spectral radius estimation we are doing here.
            fv[:] = self.fun_single(t, v)                            # evaluate
            nfesig[()] += 1
            dfnrm = np.linalg.norm(fv - fn)
            sigmal = sigma
            sigma = dfnrm / dynrm

            # sprad is a little bigger than the estimate sigma of the
            # spectral radius, so is more likely to be an upper bound.
            sprad = 1.2 * sigma
            if iter and abs(sigma - sigmal) <= max(sigma, small) * 0.01:
                # converged
                self.V[:] = v - yn
                return sprad

            # The next v(*) is the change in f
            # scaled so that norm(v - yn) = dynrm.
            if dfnrm != 0.0:
                v[:] = yn + (fv - fn) * (dynrm/dfnrm)
            else:
                # The new v(*) degenerated to yn(*)--"randomly" perturb
                # current approximation to the eigenvector by changing
                # the sign of one component.
                index = iter % self.n
                v[index] = -v[index]

        # return None to report a convergence failure.
        return None

    def _dense_output_impl(self):
        """Cubic Hermite spline for C1 continuous dense output.

        instead of: subroutine RKCINT in rkc.f
        """
        y, f, y_old, f_old = self.W[:4].copy()
        return CubicDenseOutput(self.t_old, self.t, y_old, y, f_old, f)
