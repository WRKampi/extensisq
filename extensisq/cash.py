import numpy as np
from warnings import warn
from scipy.integrate._ivp.rk import norm, SAFETY
from extensisq.common import (
    RungeKutta, HornerDenseOutput, CubicDenseOutput, LinearDenseOutput, NFS)


class CK5(RungeKutta):
    """A 5th order method with 4th order error estimator that uses the
    coefficients of Cash and Karp [1]_.

    This is not the variable order method described [1]_. That method is
    available as `CKdisc`.

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
    sc_params : tuple of size 3, "standard", "G", "H", or "S"
        Parameters for the stepsize controller (k*b1, k*b2, a2). The
        controller is as defined in [2]_, with k the exponent of the standard
        controller, _n for new and _o for old:
            h_n = h * (tol/err)**b1 * (tol/err_o)**b2  * (h/h_o)**-a2
        Predefined coefficients are Gustafsson "G" (0.7,-0.4,0), Soederlind "S"
        (0.6,-0.2,0), Hairer "H" (1,-0.6,0), and "standard" (1,0,0). Standard
        is the default.

    References
    ----------
    .. [1] J. R. Cash, A. H. Karp, "A Variable Order Runge-Kutta Method for
           Initial Value Problems with Rapidly Varying Right-Hand Sides",
           ACM Trans. Math. Softw., Vol. 16, No. 3, 1990, pp. 201-222, ISSN
           0098-3500. https://doi.org/10.1145/79505.79507
    .. [2] G. Soederlind, "Digital Filters in Adaptive Time-Stepping", ACM
           Trans. Math. Softw. Vol 29, No. 1, 2003, pp. 1–26.
           https://doi.org/10.1145/641876.641877
    """

    n_stages = 6
    order = 5
    error_estimator_order = 4
    tanang = 2.4
    stbrad = 3.7

    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [3/10, -9/10, 6/5, 0, 0, 0],
        [-11/54, 5/2, -70/27, 35/27, 0, 0],
        [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096, 0]])

    B = np.array([37/378, 0, 250/621, 125/594, 0, 512/1771])

    C = np.array([0, 1/5, 3/10, 3/5, 1, 7/8])

    E = np.array(
        [277/64512, 0, -6925/370944, 6925/202752, 277/14336, -277/7084, 0])

    # fourth order, maximum ||T5|| over step is 1.52e-3
    P = np.array([
        [1, -10405/3843, 32357/11529, -855/854],
        [0, 0, 0, 0],
        [0, 308500/88389, -1424000/265167, 67250/29463],
        [0, 5875/24156, 12875/36234, -3125/8052],
        [0, 235/1708, -235/854, 235/1708],
        [0, -287744/108031, 700416/108031, -381440/108031],
        [0, 3/2, -4, 5/2]])


class CKdisc(RungeKutta):
    """Cash Karp variable order (5, 3, 2) Runge Kutta method with error
    estimators of order (4, 2, 1). This method is created to efficiently solve
    non-smooth problems [1]_; problems with discontinuous derivatives.
    Interpolants for dense output have been added.

    The method prefers order 5. Whether this high order can be successfully
    reached in the current step is predicted multiple times between the
    evaluations of the derivative function. After the first failed prediction,
    propagation with fallback solutions of reduced order and step size is
    assessed. These fallback solutions do not need extra derivative
    evaluations.

    Step size is expected to be irregular in this method. This can interfere
    with stiffness detection and non-standard stepsize control, which are
    therefore disabled.

    Can be applied in the complex domain.

    A fixed fifth order method with the Cash Karp parameters is available as
    CK5.

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

    References
    ----------
    .. [1] J. R. Cash, A. H. Karp, "A Variable Order Runge-Kutta Method for
           Initial Value Problems with Rapidly Varying Right-Hand Sides",
           ACM Trans. Math. Softw., Vol. 16, No. 3, 1990, pp. 201-222, ISSN
           0098-3500. https://doi.org/10.1145/79505.79507
    """
    # changes w.r.t. paper [1]_:
    # - loop reformulated to prevent code repetition.
    # - atol and rtol as in scipy, instead of only scalar atol.
    # - do not allow for an increase in step size directly after a failed step.
    # - incude interpolants for dense output.

    n_stages = 6                                              # for main method
    order = 5                                                 # for main method
    error_estimator_order = 4                                 # for main method
    max_factor = 5
    min_factor = 1/5

    # time step fractions
    C = np.array([0, 1/5, 3/10, 3/5, 1, 7/8])

    # coefficient matrix
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [3/10, -9/10, 6/5, 0, 0, 0],
        [-11/54, 5/2, -70/27, 35/27, 0, 0],
        [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096, 0]])

    # all embedded orders:
    B_all = np.array([
        6*[0],                                                        # order 0
        [1, 0, 0, 0, 0, 0],                                           # order 1
        [-3/2, 5/2, 0, 0, 0, 0],                                      # order 2
        [19/54, 0, -10/27, 55/54, 0, 0],                              # order 3
        [2825/27648, 0, 18575/48384, 13525/55296, 277/14336,
            1/4],                                                     # order 4
        [37/378, 0, 250/621, 125/594, 0, 512/1771]])                  # order 5

    # coefficients for main propagating method
    B = B_all[5, :]                                                   # order 5
    E = np.zeros(7)
    E[:-1] = B_all[5, :] - B_all[4, :]                             # order 4(5)

    # coefficients for convergence assessment
    B_assess = B_all[[2, 3], :]
    E_assess = np.array([
        B_all[2, :] - B_all[1, :],                                 # order 1(2)
        B_all[3, :] - B_all[2, :]])                                # order 2(3)

    # coefficients for fallback methods
    C_fallback = C[[1, 3]]
    B_fallback = np.array([
        [1/10, 1/10, 0, 0, 0, 0],                                     # order 2
        [1/10, 0, 2/5, 1/10, 0, 0]])                                  # order 3
    E_fallback = np.array([
        [-1/10, 1/10, 0, 0, 0, 0],                                 # order 1(2)
        [1/10, 0, -2/10, 1/10, 0, 0]])                             # order 2(3)

    # fourth order interpolator for fifth order solution
    # maximum ||T5|| in [0,1] is 1.52e-3
    P = np.array([
        [1, -10405/3843, 32357/11529, -855/854],
        [0, 0, 0, 0],
        [0, 308500/88389, -1424000/265167, 67250/29463],
        [0, 5875/24156, 12875/36234, -3125/8052],
        [0, 235/1708, -235/854, 235/1708],
        [0, -287744/108031, 700416/108031, -381440/108031],
        [0, 3/2, -4, 5/2]])

    def __init__(self, fun, t0, y0, t_bound, **extraneous):
        super(CKdisc, self).__init__(
            fun, t0, y0, t_bound, nfev_stiff_detect=0, **extraneous)
        # adaptive weighing factors:
        self.twiddle = [1.5, 1.1]                             # starting values
        self.quit = [100., 100.]                              # starting values

    def _step_impl(self):
        t = self.t
        y = self.y
        twiddle = self.twiddle
        quit = self.quit

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

        order_accepted = 0
        step_rejected = False
        while not order_accepted:

            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h

            # start the integration, two stages at a time
            self.K[0] = self.f                                        # stage 0
            self._rk_stage(h, 1)                                      # stage 1

            # first order error, second order solution, for assessment
            y_assess, err_assess, tol = self._comp_sol_err_tol(
                                    h, self.B_assess[0], self.E_assess[0], 2)
            E1 = norm(err_assess/tol) ** (1/2)
            esttol = E1 / self.quit[0]

            # assess if full step completion is likely
            if E1 < twiddle[0] * quit[0]:
                # green light -> continue with next two stages
                self._rk_stage(h, 2)                                  # stage 2
                self._rk_stage(h, 3)                                  # stage 3

                # second order error, third order solution, for assessment
                y_assess, err_assess, tol = self._comp_sol_err_tol(
                                    h, self.B_assess[1], self.E_assess[1], 4)
                E2 = norm(err_assess/tol) ** (1/3)
                esttol = E2 / self.quit[1]

                # assess if full step completion is likely
                if E2 < twiddle[1]*quit[1]:
                    # green light -> continue with last two stages
                    self._rk_stage(h, 4)                              # stage 4
                    self._rk_stage(h, 5)                              # stage 5

                    # second fourth error, fifth order solution, for output
                    y_new, err, tol = self._comp_sol_err_tol(h, self.B, self.E)
                    E4 = norm(err/tol) ** (1/5)
                    E4 = E4 or 1e-160                                # no div 0
                    esttol = E4

                    # assess final error
                    if E4 < 1:
                        # accept fifth order solution
                        order_accepted = 4                        # error order

                        # update h for next step
                        factor = min(self.max_factor, SAFETY/E4)
                        if step_rejected:                        # not in paper
                            factor = min(1.0, factor)
                        h_abs *= factor

                        # update quit factors
                        q = [E1/E4, E2/E4]
                        for j in (0, 1):
                            if q[j] > quit[j]:
                                q[j] = min(q[j], 10 * quit[j])
                            else:
                                q[j] = max(q[j], 2/3 * quit[j])
                            quit[j] = max(1., min(10000., q[j]))

                        break

                    # fifth order solution NOT accepted

                    # update twiddle factors
                    e = [E1, E2]
                    for i in (0, 1):
                        EQ = e[i] / quit[i]
                        if EQ < twiddle[i]:
                            twiddle[i] = max(1.1, EQ)

                    # assess propagation with third order fallback solution
                    if E2 < 1:
                        y_new, err, tol = self._comp_sol_err_tol(
                                h, self.B_fallback[1], self.E_fallback[1], 4)

                        # assess second order error
                        if norm(err/tol) < 1:
                            # accept third order fallback solution
                            order_accepted = 2                    # error order
                            h_abs *= self.C_fallback[1]      # reduce next step
                            h = h_abs * self.direction          # and THIS step
                            break

                # assess propagation with second order fallback solution
                if E1 < 1:
                    y_new, err, tol = self._comp_sol_err_tol(
                                h, self.B_fallback[0], self.E_fallback[0], 2)

                    # assess first order error
                    if norm(err/tol) < 1:
                        # accept second order fallback solution
                        order_accepted = 1
                        h_abs *= self.C_fallback[0]          # reduce next step
                        h = h_abs * self.direction              # and THIS step
                        break

                    else:
                        # non-smooth behavior detected retry step with h/5
                        step_rejected = True
                        h_abs *= self.C_fallback[0]
                        NFS[()] += 1
                        continue

            # just not accurate enough, retry with usual estimate for h
            step_rejected = True
            h_abs *= max(self.min_factor, SAFETY/esttol)
            NFS[()] += 1
            continue

        # end of main while loop

        # calculate the derivative of the accepted solution
        # for first stage of next step and for interpolation
        t_new = t + h
        f_new = self.fun(t_new, y_new)
        self.K[-1, :] = f_new

        # store for next step and interpolation
        self.order_accepted = order_accepted                      # error order
        self.h_previous = h
        self.y_old = y
        self.h_abs = h_abs
        self.f = f_new

        # output
        self.t = t_new
        self.y = y_new
        return True, None

    def _compute_error(self, h, E, i):
        return h * (self.K[:i, :].T @ E[:i])

    def _compute_solution(self, h, B, i):
        return h * (self.K[:i, :].T @ B[:i]) + self.y

    def _comp_sol_err_tol(self, h, B, E, i=6):
        sol = self._compute_solution(h, B, i)
        err = self._compute_error(h, E, i)
        tol = self.atol + self.rtol * 0.5*(np.abs(self.y) + np.abs(sol))
        return sol, err, tol

    def _dense_output_impl(self):
        if self.f is None:
            # output was extrapolated linearly
            return LinearDenseOutput(self.t_old, self.t, self.y_old, self.y)
        # select interpolator based on order of the accepted error (solution)
        if self.order_accepted == 4:
            # 4th order error estimate accepted (5th order solution)
            Q = self.K.T @ self.P
            return HornerDenseOutput(self.t_old, self.t, self.y_old, Q)
        # low order solution
        return CubicDenseOutput(self.t_old, self.t, self.y_old, self.y,
                                self.K[0, :], self.K[-1, :])


# old class names
class CK45_o(CK5):
    def __init__(self, *args, **kwargs):
        warn("This method will be replaced by 'CK5'.", FutureWarning)
        super(CK45_o, self).__init__(*args, **kwargs)


class CK45(CKdisc):
    def __init__(self, *args, **kwargs):
        warn("This method will be replaced by 'CKdisc'.", FutureWarning)
        super(CK45, self).__init__(*args, **kwargs)
