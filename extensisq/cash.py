import numpy as np
from scipy.integrate._ivp.rk import (
    RungeKutta, RkDenseOutput, norm, SAFETY, MIN_FACTOR, MAX_FACTOR)


class CK45(RungeKutta):
    """Cash Karp variable order (5, 3, 2) runge kutta method with error
    estimators of order (4, 2, 1). This method is created to efficiently solve
    non-smooth problems [1]_. Interpolants for dense output have been added.

    The method prefers order 5. Whether this high order can be successfuly
    reached in the current step is predicted multiple times between the
    evaluations of the derivative function. After the first failed prediction,
    propagation with fallback solutions of reduced order and step size is
    assessed. These fallback solutions do not need extra derivative
    evaluations.

    Can be applied in the complex domain.

    A fixed order (4,5) method with the Cash Karp parameters is available as
    CK45_o.

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
        Whether `fun` is implemented in a vectorized fashion. Default is False.

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
    .. [1] J. R. Cash, A. H. Karp, "A Variable Order Runge-Kutta Method for
           Initial Value Problems with Rapidly Varying Right-Hand Sides",
           ACM Trans. Math. Softw., Vol. 16, No. 3, 1990, pp. 201-222, ISSN
           0098-3500. https://doi.org/10.1145/79505.79507
    """
    # changes w.r.t. paper [1]_:
    # - loop reformulated to prevent code repetition.
    # - atol and rtol that may be arrays, instead of only scalar atol.
    # - do not allow for an increase in step size directly after a failed step.
    # - incude interpolants for dense output.

    n_stages = 6                                            # for main method
    order = 5                                               # for main method
    error_estimator_order = 4                               # for main method
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
        [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4],    # order 4
        [37/378, 0, 250/621, 125/594, 0, 512/1771]])                  # order 5

    # coefficients for main propagating method
    B = B_all[5, :]                                   # order 5
    E = B_all[5, :] - B_all[4, :]                     # order 4(5)

    # coefficients for convergence assessment
    B_assess = B_all[[2, 3], :]
    E_assess = np.array([
            B_all[2, :] - B_all[1, :],                # order 1(2)
            B_all[3, :] - B_all[2, :]])               # order 2(3)

    # coefficients for fallback methods
    C_fallback = C[[1, 3]]
    B_fallback = np.array([
        [1/10, 1/10, 0, 0, 0, 0],                     # order 2
        [1/10, 0, 2/5, 1/10, 0, 0]])                  # order 3
    E_fallback = np.array([
        [-1/10, 1/10, 0, 0, 0, 0],                    # order 1(2)
        [1/10, 0, -2/10, 1/10, 0, 0]])                # order 2(3)

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

    # cubic Hermite spline interpolators (C1) for fallback solutions
    P_fallback = np.zeros((2, n_stages+1, 3))
    P_fallback[:, :-1, :] = (
        B_fallback/C_fallback[:, np.newaxis]
        )[:, :, np.newaxis]*[0,  3, -2]               # value at end
    P_fallback[:,  0, :] += [1, -2,  1]               # derivative at start
    P_fallback[:, -1, :] += [0, -1,  1]               # derivative at end

    def __init__(self, fun, t0, y0, t_bound, **extraneous):
        super(CK45, self).__init__(fun, t0, y0, t_bound, **extraneous)
        self.K[:, :] = 0.   # set 0 for interpolator interpolator:
        # weighing factors, these are adaptively changed:
        self.twiddle = np.array([1.5, 1.1])           # starting values
        self.quit = np.array([100., 100.])            # starting values

    def _step_impl(self):
        t = self.t
        y = self.y
        twiddle = self.twiddle
        quit = self.quit

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > self.max_step:
            h_abs = self.max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        order_accepted = 0
        step_rejected = False
        while not order_accepted:

            # the usual tests
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h
            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
            # added look ahead to prevent too small last step
            elif abs(t_new - self.t_bound) <= min_step:
                t_new = t + h/2
            h = t_new - t
            h_abs = np.abs(h)

            # start the integration, two stages at a time
            self.K[0] = self.f                        # stage 0 (FSAL)
            self._rk_stage(h, 1)                      # stage 1

            # first order error, second order solution, for assessment
            y_assess, err_assess, tol = self._comp_sol_err_tol(
                                    h, self.B_assess[0], self.E_assess[0], 2)
            E1 = norm(err_assess/tol)**(1/2)
            esttol = E1/self.quit[0]

            # assess if full step completion is likely
            if E1 < twiddle[0]*quit[0]:
                # green light -> continue with next two stages
                self._rk_stage(h, 2)                  # stage 2
                self._rk_stage(h, 3)                  # stage 3

                # second order error, third order solution, for assessment
                y_assess, err_assess, tol = self._comp_sol_err_tol(
                                    h, self.B_assess[1], self.E_assess[1], 4)
                E2 = norm(err_assess/tol)**(1/3)
                esttol = E2/self.quit[1]

                # assess if full step completion is likely
                if E2 < twiddle[1]*quit[1]:
                    # green light -> continue with last two stages
                    self._rk_stage(h, 4)              # stage 4
                    self._rk_stage(h, 5)              # stage 5

                    # second fourth error, fifth order solution, for output
                    y_new, err, tol = self._comp_sol_err_tol(h, self.B, self.E)
                    E4 = norm(err/tol)**(1/5)
                    E4 = E4 or 1e-99                  # prevent div 0
                    esttol = E4

                    # assess final error
                    if E4 < 1:
                        # accept fifth order solution
                        order_accepted = 4            # error order

                        # update h for next step
                        factor = min(self.max_factor, SAFETY/E4)
                        if step_rejected:             # not in paper
                            factor = min(1.0, factor)
                        h_abs *= factor

                        # update quit factors
                        q = [E1/E4, E2/E4]
                        for j in (0, 1):
                            if q[j] > quit[j]:
                                q[j] = min(q[j], 10.*quit[j])
                            else:
                                q[j] = max(q[j], 2/3*quit[j])
                            quit[j] = max(1., min(10000., q[j]))

                        break

                    # fifth order solution NOT accepted

                    # update twiddle factors
                    e = [E1, E2]
                    for i in (0, 1):
                        EQ = e[i]/quit[i]
                        if EQ < twiddle[i]:
                            twiddle[i] = max(1.1, EQ)

                    # assess propagation with third order fallback solution
                    if E2 < 1:
                        y_new, err, tol = self._comp_sol_err_tol(
                                h, self.B_fallback[1], self.E_fallback[1], 4)

                        # assess second order error
                        if norm(err/tol) < 1:
                            # accept third order fallback solution
                            order_accepted = 2              # error order
                            h_abs *= self.C_fallback[1]     # reduce next step
                            h = h_abs * self.direction      # and THIS step
                            break

                # assess propagation with second order fallback solution
                if E1 < 1:
                    y_new, err, tol = self._comp_sol_err_tol(
                                h, self.B_fallback[0], self.E_fallback[0], 2)

                    # assess first order error
                    if norm(err/tol) < 1:
                        # accept second order fallback solution
                        order_accepted = 1
                        h_abs *= self.C_fallback[0]         # reduce next step
                        h = h_abs * self.direction          # and THIS step
                        break

                    else:
                        # non-smooth behavior detected retry step with h/5
                        step_rejected = True
                        h_abs *= self.C_fallback[0]
                        continue

            # just not accurate enough, retry with usual estimate for h
            step_rejected = True
            h_abs *= max(self.min_factor, SAFETY/esttol)
            continue

        # end of main while loop

        # calculate the derivative of the accepted solution
        # for first stage of next step and for interpolation
        t_new = t + h
        f_new = self.fun(t_new, y_new)
        self.K[-1, :] = f_new
        self.order_accepted = order_accepted                # error order

        # as usual:
        self.h_previous = h
        self.y_old = y
        self.t = t_new
        self.y = y_new
        self.h_abs = h_abs
        self.f = f_new
        return True, None

    def _rk_stage(self, h, i):
        dy = self.K[:i, :].T @ self.A[i, :i] * h
        self.K[i] = self.fun(self.t + self.C[i]*h, self.y + dy)

    def _compute_error(self, h, E, i):
        return self.K[:i, :].T @ E[:i] * h

    def _compute_solution(self, h, B, i):
        return self.K[:i, :].T @ B[:i] * h + self.y

    def _comp_sol_err_tol(self, h, B, E, i=6):
        sol = self._compute_solution(h, B, i)
        err = self._compute_error(h, E, i)
        tol = self.atol + self.rtol*np.maximum(np.abs(self.y), np.abs(sol))
        return sol, err, tol

    def _dense_output_impl(self):
        # select interpolator based on order of the accepted error (solution)
        P = self.P
        if self.order_accepted != 4:
            P = self.P_fallback[self.order_accepted-1]
        Q = self.K.T @ P
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)

    def _estimate_error(self, K, h):
        # only used for testing
        return self._compute_error(h, self.E, 6)

    def _estimate_error_norm(self, K, h, scale):
        # only used for testing
        sol, err, tol = self._comp_sol_err_tol(h, self.B, self.E, i=6)
        return norm(err/tol)


class CK45_o(CK45):
    """As CK45, but fixed at 5th order solution porpagator with 4th order error
    estimator. (suffix _o for order)

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
        Whether `fun` is implemented in a vectorized fashion. Default is False.

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
    .. [1] J. R. Cash, A. H. Karp, "A Variable Order Runge-Kutta Method for
           Initial Value Problems with Rapidly Varying Right-Hand Sides",
           ACM Trans. Math. Softw., Vol. 16, No. 3, 1990, pp. 201-222, ISSN
           0098-3500. https://doi.org/10.1145/79505.79507
    """

    order_accepted = 4  # for dense output

    def _step_impl(self):
        t = self.t
        y = self.y
        max_step = self.max_step

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h
            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
            # added look ahead to prevent too small last step
            elif abs(t_new - self.t_bound) <= min_step:
                t_new = t + h/2
            h = t_new - t
            h_abs = np.abs(h)

            # not FSAL, do last evaluation only after step is accepted
            # calculate the 6 stages
            self.K[0] = self.f                              # stage 0 (FSAL)
            for i in range(1, 6):
                self._rk_stage(h, i)                        # stages 1-5
            y_new, err, tol = self._comp_sol_err_tol(h, self.B, self.E)
            error_norm = norm(err/tol)
            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)
                if step_rejected:
                    factor = min(1, factor)
                h_abs *= factor
                step_accepted = True
                # now do the last evaluation
                f_new = self.fun(t_new, y_new)
                self.K[6] = f_new                           # stage 6 (FSAL)

            else:
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** self.error_exponent)
                step_rejected = True

        self.h_previous = h
        self.y_old = y
        self.t = t_new
        self.y = y_new
        self.h_abs = h_abs
        self.f = f_new
        return True, None


if __name__ == '__main__':
    """Construction of a free interpolant of the CK45 pair. The approach from
    "Runge-Kutta pairs of order 5(4) satisfying only the first column
    simplifying assumption" by Ch Tsitouras is followed."""
    import numpy as np
    import matplotlib.pyplot as plt
    import sympy
    from sympy.solvers.solveset import linsolve
    from sympy import Rational as R
    from pprint import pprint

    n_stages = 7         # including derivative evaluation at end of step
    order = 4            # of interpolation in t (not h)
    T5_method4 = 5e-4    # error of embedded fourth order method

    t = sympy.symbols('t', real=True)
    bi = sympy.symbols(f'bi0:{n_stages}', real=True)
    bi_vec = sympy.Matrix(bi)

    # Method
    A = sympy.Matrix([
            [0, 0, 0, 0, 0, 0, 0],
            [R(1, 5), 0, 0, 0, 0, 0, 0],
            [R(3, 40), R(9, 40), 0, 0, 0, 0, 0],
            [R(3, 10), R(-9, 10), R(6, 5), 0, 0, 0, 0],
            [R(-11, 54), R(5, 2), R(-70, 27), R(35, 27), 0, 0, 0],
            [R(1631, 55296), R(175, 512), R(575, 13824), R(44275, 110592),
             R(253, 4096), 0, 0],
            [R(37, 378), 0, R(250, 621), R(125, 594), 0, R(512, 1771), 0]
        ])     # output stage appended

    c = sympy.Matrix([0, R(1, 5), R(3, 10), R(3, 5), 1, R(7, 8), 1])
    e = sympy.Matrix([1, 1, 1, 1, 1, 1, 1])

    # error terms up to order 4
    c2 = c.multiply_elementwise(c)
    Ac = A*c

    c3 = c2.multiply_elementwise(c)
    cAc = c.multiply_elementwise(Ac)
    Ac2 = A*c2
    A2c = A*Ac

    T11 = bi_vec.dot(e) - t
    T21 = bi_vec.dot(c) - t**2/2

    T31 = bi_vec.dot(c2)/2 - t**3/6
    T32 = bi_vec.dot(Ac) - t**3/6

    T41 = bi_vec.dot(c3)/6 - t**4/24
    T42 = bi_vec.dot(cAc) - t**4/8
    T43 = bi_vec.dot(Ac2)/2 - t**4/24
    T44 = bi_vec.dot(A2c) - t**4/24

    # solve polynomials to let all terms up to order 4 vanish
    bi_vec_t = sympy.Matrix(
        linsolve([T11, T21, T31, T32, T41, T42, T43, T44], bi).args[0])
    i_free_poly = [i for i, (term, poly) in enumerate(zip(bi_vec_t, bi))
                   if poly == term]
    free_polys = [bi[i] for i in i_free_poly]
    print('free polynomials:', free_polys)      # poly bi5 is free to define

    # Make this free polynommial explicit in t
    parameters = sympy.symbols([f'bi{i}_0:{order+1}' for i in i_free_poly])
    polys = []
    for coefs in parameters:
        p = 0
        for i, coef in enumerate(coefs):
            p = p + coef * t**i
        polys.append(p)

    # substitute in bi_vec_t
    subs_dict = dict(zip(free_polys, polys))
    bi_vec_t = bi_vec_t.subs(subs_dict)

    # demand continuity at start and end of step
    d_bi_vec_t = sympy.diff(bi_vec_t, t)        # derivative
    # C0 at t=0
    C0_0 = [eq for eq in bi_vec_t.subs(t, 0)]
    # C0 at t=1
    C0_1 = [eq for eq in (bi_vec_t.subs(t, 1) - A[-1, :].T)]
    # C1 at t=0
    C1_0 = d_bi_vec_t.subs(t, 0)
    C1_0[0] = C1_0[0] - 1
    C1_0 = [eq for eq in C1_0]
    # C1 at t=1
    C1_1 = d_bi_vec_t.subs(t, 1)
    C1_1[-1] = C1_1[-1] - 1
    C1_1 = [eq for eq in C1_1]

    # combine equations in list
    eqns = C0_0
    eqns.extend(C0_1)
    eqns.extend(C1_0)
    eqns.extend(C1_1)

    # combine parameters in list
    params = []
    for p in parameters:
        params.extend(p)

    # solve continuity constraints
    sol1 = linsolve(eqns, params).args[0]

    # whych params are still free?
    free_params = [p for s, p in zip(sol1, params) if s == p]
    print('free parameters:', free_params)      # free parameter: bi5_4

    # update bi_vec_t
    subs_dict = dict(zip(params, sol1))
    bi_vec_t = bi_vec_t.subs(subs_dict)

    # find value for free parameter that minimizes the 5th order error terms

    # error terms of order 5
    c4 = c3.multiply_elementwise(c)
    c2Ac = c2.multiply_elementwise(Ac)
    Ac_Ac = Ac.multiply_elementwise(Ac)
    cAc2 = c.multiply_elementwise(Ac2)
    Ac3 = A*c3
    cA2c = c.multiply_elementwise(A2c)
    A_cAc = A*cAc
    A2c2 = A*Ac2
    A3c = A*A2c

    T51 = bi_vec_t.dot(c4)/24 - t**5/120
    T52 = bi_vec_t.dot(c2Ac)/2 - t**5/20
    T53 = bi_vec_t.dot(Ac_Ac)/2 - t**5/40
    T54 = bi_vec_t.dot(cAc2)/2 - t**5/30
    T55 = bi_vec_t.dot(Ac3)/6 - t**5/120
    T56 = bi_vec_t.dot(cA2c) - t**5/30
    T57 = bi_vec_t.dot(A_cAc) - t**5/40
    T58 = bi_vec_t.dot(A2c2)/2 - t**5/120
    T59 = bi_vec_t.dot(A3c) - t**5/120

    # error norm 5 (omitting square root for simplification)
    T5_norm_t = (T51**2 + T52**2 + T53**2 + T54**2 + T55**2 + T56**2 + T57**2 +
                 T58**2 + T59**2)
    T5_norm_i = sympy.integrate(T5_norm_t, (t, 0, 1))

    # minimize norm -> find root of derivative
    eqns = []
    for param in free_params:
        eqns.append(sympy.diff(T5_norm_i, param))
    if eqns:
        sol2 = linsolve(eqns, free_params).args[0]
    else:
        sol2 = []
    print('optimal value of free parameters:', sol2)

    # update bi_vec_t and norms
    subs_dict = dict(zip(free_params, sol2))
    bi_vec_t = bi_vec_t.subs(subs_dict)
    T5_norm_t = sympy.sqrt(T5_norm_t.subs(subs_dict))   # now take sqrt

    # create numerical function for plotting
    T5_fun = sympy.lambdify(t, T5_norm_t, 'numpy')
    t_ = np.linspace(0., 1., 101)
    T5_max = T5_fun(t_).max()
    print('T_5 max:', T5_max)
    print('T5 max interp/T5 method:', T5_max.max()/T5_method4)

    print('resulting interpolant:')
    pprint(bi_vec_t)

    # plot error
    plt.plot(t_, T5_fun(t_), label='free 4th order interpolant')
    plt.axhline(T5_method4, ls='--', label='embedded 4th order method')
    plt.tight_layout
    plt.xlim(0, 1)
    plt.ylim(ymin=0)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'error $\hat{T}_5$')
    plt.legend(loc=1)
    plt.title('free interpolant for CK45')
    plt.tight_layout()
    plt.savefig('free interpolant for CK45')
    plt.show()
