import numpy as np
from scipy.integrate._ivp.rk import RungeKutta, RkDenseOutput, norm, rk_step


SAFETY = 0.9
MIN_FACTOR = 1/5
MAX_FACTOR = 5



class CK45(RungeKutta):
    """Cash Karp variable order (5, 3, 2) runge kutta method with error 
    estimators of order (4, 2, 1). This method is created to efficiently solve 
    non-smooth problems [1]_. Interpolants for dense output have been added.
    
    Order 5 is preferred. Whether this high order can be successfuly reached in 
    the current step is predicted multiple times between the evaluations of the 
    derivative function. After the first failed prediction, propagation with 
    fallback solutions of reduced oredr and step size is assessed. These 
    fallback solutions do not need extra derivative evaluations.
    
    A fixed order (4,5) method with the Cash Karp parameters is available as
    CK45_o.
    
    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
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
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
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
    # - allow for atol and rtol that may be arrays, instead of only scalar atol.
    # - do not allow for an increase in step size directly after a failed step.
    # - incude interpolants for dense output.
    
    n_stages = 6                                            # for main method
    order = 5                                               # for main method
    error_estimator_order = 4                               # for main method
    
    # weighing factors, these are adaptively changed
    twiddle = np.array([1.5, 1.1])                          # starting values
    quit = np.array([100., 100.])                           # starting values
    
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
    B = B_all[5,:]                                    # order 5
    E = B_all[5,:] - B_all[4,:]                       # order 4(5)
    
    # coefficients for convergence assessment
    B_assess = B_all[[2,3],:]
    E_assess = np.array([
            B_all[2,:] - B_all[1,:],                  # order 1(2)
            B_all[3,:] - B_all[2,:]])                 # order 2(3)
    
    # coefficients for fallback methods
    C_fallback = C[[1,3]]
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
    
    # cubic spline interpolators (C1) for fallback solutions
    P_fallback = np.zeros((2, n_stages+1, 3))
    P_fallback[:,:-1,:] = (B_fallback/C_fallback[:,np.newaxis]
        )[:,:,np.newaxis]*[0,  3, -2]                 # value at end
    P_fallback[:,0,:]  += [1, -2,  1]                 # derivative at start
    P_fallback[:,-1,:] += [0, -1,  1]                 # derivative at end
    
    def __init__(self, fun, t0, y0, t_bound, **extraneous):
        super(CK45, self).__init__(fun, t0, y0, t_bound, **extraneous)
        # prevent possible nan or inf in K that may interfere with interpolator:
        self.K[:,:] = 0.
    
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
                    E4 = E4 or np.finfo(float).eps    # prevent div 0
                    esttol = E4
                    
                    # assess final error
                    if E4 < 1:
                        # accept fifth order solution
                        order_accepted = 4            # error order
                        
                        # update h for next step
                        factor = min(MAX_FACTOR, SAFETY/E4)
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
            h_abs *= max(MIN_FACTOR, SAFETY/esttol)
            continue
        
        # end of main while loop
        
        # calculate the derivative of the accepted solution
        # for first stage of next step and for interpolation
        t_new = t + h
        f_new = self.fun(t_new, y_new)
        self.K[-1,:] = f_new
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
        dy = np.dot(self.K[:i,:].T, self.A[i,:i]) * h
        self.K[i] = self.fun(self.t + self.C[i]*h, self.y + dy)
    
    def _compute_error(self, h, E, i):
        return h*np.dot(self.K[:i,:].T, E[:i])
    
    def _compute_solution(self, h, B, i):
        return np.dot(self.K[:i,:].T, B[:i]) * h + self.y
    
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
        Q = self.K.T.dot(P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)


class CK45_o(CK45):
    """As CK45, but fixed at 5th order solution porpagator with 4th order error 
    estimator. (suffix _o for order)

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
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
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
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
    
    max_factor = 10     # use scipy default
    order_accepted = 4  # for dense output
    
    def _step_impl(self):
        t = self.t
        y = self.y
        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol
        
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
                    factor = self.max_factor
                else:
                    factor = min(self.max_factor,
                                 SAFETY * error_norm ** self.error_exponent)
                if step_rejected:
                    factor = min(1, factor)
                h_abs *= factor
                step_accepted = True
                # now do the last evaluation
                f_new = self.fun(t_new, y_new)
                self.K[6] = f_new                               # stage 6 (FSAL)
                
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
    
