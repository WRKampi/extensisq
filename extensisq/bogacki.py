import numpy as np
from scipy.integrate._ivp.rk import (RungeKutta, RkDenseOutput, rk_step, norm,
    SAFETY, MAX_FACTOR, MIN_FACTOR)       # using scipy's values, not rksuite's


class BS45(RungeKutta):
    """Explicit Runge-Kutta method of order 5(4).

    This uses the Bogacki-Shampine pair of formulas [1]_. It is designed
    to be more efficient than the Dormand-Prince pair (RK45 in scipy).
    
    There are two independent fourth order estimates of the local error.
    The fifth order method is used to advance the solution (local 
    extrapolation). Coefficients from [2]_ are used.
    
    The interpolator for dense output is of fifth order and needs three 
    additional derivative function evaluations (when used). A free, fourth 
    order interpolator is also available; see dense_output_order below.
    
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
    dense_output_order : {'high', 'low'}, optional
        For 'high', the high quality fifth order interpolator from [1]_ is used 
        for dense output. This requires extra evaluations of `fun` when used. 
        For 'low', a free 4th order interpolator is used. It was constructed 
        following [3]_. This option can be useful when low quality interpolation 
        is acceptable (e.g. smooth plotting). The default setting is 'high'.

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
    .. [1] P. Bogacki, L.F. Shampine, "An efficient Runge-Kutta (4,5) pair",
           Computers & Mathematics with Applications, Vol. 32, No. 6, 1996, 
           pp. 15-28, ISSN 0898-1221.
           https://doi.org/10.1016/0898-1221(96)00141-1
    .. [2] RKSUITE: https://www.netlib.org/ode/rksuite/
    .. [3] Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the 
           first column simplifying assumption", Computers & Mathematics with 
           Applications, Vol. 62, No. 2, pp. 770 - 775, 2011.
           https://doi.org/10.1016/j.camwa.2011.06.002
    """
    
    order = 5
    error_estimator_order = 4
    n_stages = 7        # the effective nr (total nr of stages is 8)
    n_extra_stages = 3  # for dense output
    
    
    # time step fractions
    C = np.array([0, 1/6, 2/9, 3/7, 2/3, 3/4, 1, 1])
    C = C[:-1]    # last one removed to pass unit test and conform to scipy
    
    
    # coefficient matrix, including row of last stage
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/6, 0, 0, 0, 0, 0, 0],
        [2/27, 4/27, 0, 0, 0, 0, 0],
        [183/1372, -162/343, 1053/1372, 0, 0, 0, 0],
        [68/297, -4/11, 42/143, 1960/3861, 0, 0, 0],
        [597/22528, 81/352, 63099/585728, 58653/366080, 4617/20480, 0, 0],
        [174197/959244, -30942/79937, 8152137/19744439, 666106/1039181, 
                -29421/29068, 482048/414219, 0],
        [587/8064, 0, 4440339/15491840, 24353/124800, 387/44800, 2152/5985, 
                7267/94080]])
    
    # coefficients for propagating method
    B = A[-1,:].copy()
    
    # remove last row from A, conforming to scipy convention of size
    A = A[:-1,:].copy()
    
    # coefficients for first error estimation method
    E1 = np.array([-3/1280, 0, 6561/632320, -343/20800, 243/12800, -1/95, 0, 0])
    
    # coefficients for second error estimation method
    E2 = np.array([2479/34992, 0, 123/416, 612941/3411720, 43/1440, 2272/6561, 
                    79937/1113912, 3293/556956])
    E2[:-1] -= B   # convert to error coefficients
    E = E2
    
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
    
    # coefficients for interpolation (high order, default)
    P = np.array([
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
    
    # Bogacki published a free interpolant in his thesis, but I was not able to
    # find a copy of it. Instead, I constructed an interpolant using sympy and 
    # the approach in [3]_.
    # This free 4th order interpolant has a leading error term ||T5|| that has 
    # maximum in [0,1] of 5.47 e-4. This is higher than the corresponding term
    # of the embedded fourth order method: 1.06e-4.
    Pfree = np.array([
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
        [0, -38480331/36476731, 226874786/36476731, -374785310/36476731, 
            186390855/36476731]])
    
    
    def __init__(self, fun, t0, y0, t_bound, dense_output_order='high', 
            **extraneous):
        super(BS45, self).__init__(fun, t0, y0, t_bound, **extraneous)
        # custom initialization to create extended storage for dense output
        # and to make the interpolator selectable
        self.K_extended = np.zeros((self.n_stages+self.n_extra_stages+1, 
                self.n), dtype=self.y.dtype)
        self.K = self.K_extended[:self.n_stages+1]
        self.dense_output_order = dense_output_order
        
    
    def _step_impl(self):
        # modified to include two error estimators. This saves two function 
        # evaluations for most rejected steps. (The step can still be rejected 
        # by the second error estimator, but this will be rare.)
        
        t = self.t
        y = self.y
        y_old = self.y_old
        if y_old is None:                                   # None at start
            y_old = y + np.finfo(float).eps * self.direction * self.f
        rtol = self.rtol                                    
        atol = self.atol
        
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
            if self.direction*(t_new - self.t_bound) > 0:
                t_new = self.t_bound
            # added look ahead to prevent too small last step
            elif abs(t_new - self.t_bound) <= min_step:
                t_new = t + h/2
            h = t_new - t
            h_abs = np.abs(h)
            
            # calculate first 6 stages
            self.K[0] = self.f                              # stage 0 (FSAL)
            for i in range(1, 6):
                self._rk_stage(h, i)                        # stages 1-5
            
            # calculate the first error estimate
            # y_new is not available yet for scale, so use y_old instead
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_old))
            error_norm = self._estimate_error_norm(self.E1, h, scale)
            
            # reject step if needed
            if error_norm > 1:
                step_rejected = True
                h_abs *= max(MIN_FACTOR, SAFETY*error_norm**self.error_exponent)
                continue
            
            # calculate solution
            self._rk_stage(h, 6)                            # stage 6
            y_new = y + h * np.dot(self.K[:-1].T, self.B)
            
            # calculate second error estimate
            # now use y_new for scale
            f_new = self.fun(t_new, y_new)
            self.K[7] = f_new                               # stage 7 (FSAL)
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
            error_norm = self._estimate_error_norm(self.E2, h, scale)
            
            # continue as usual
            if error_norm < 1:
                step_accepted = True
                if error_norm == 0.0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, 
                                    SAFETY * error_norm**self.error_exponent)
                if step_rejected:
                    factor = min(1.0, factor)
                h_abs *= factor
            else:
                step_rejected = True
                h_abs *= max(MIN_FACTOR, 
                                    SAFETY * error_norm**self.error_exponent)
        
        # after sucessful step; as usual
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
    
    
    def _estimate_error(self, E, h):
        # pass E instead of K
        return np.dot(self.K.T, E) * h
    
    
    def _estimate_error_norm(self, E, h, scale):
        # pass E instead of K
        return norm(self._estimate_error(E, h) / scale)
    
    
    def _dense_output_impl(self):
        
        if self.dense_output_order=='high':                 # default
            h = self.h_previous
            K = self.K_extended
            
            # calculate the required extra stages
            for s, (a, c) in enumerate(zip(self.A_extra, self.C_extra),
                        start=self.n_stages+1):
                dy = np.dot(K[:s,:].T, a[:s]) * h
                K[s] = self.fun(self.t_old + c * h, self.y_old + dy)
            
            # form Q. Usually: Q = K.T.dot(self.P)
            # but rksuite recommends to group summations to mitigate roundoff:
            Q = np.empty((K.shape[1], self.P.shape[1]))
            Q[:,0] = K[7,:]                                 # term for t**1
            KP = K*self.P[:,1,np.newaxis]                   # term for t**2
            Q[:,1] = ( KP[4]  +  ((KP[5]+KP[7]) + KP[0]) 
                   + ((KP[2]+KP[8]) + KP[9])  +  ((KP[3]+KP[10]) + KP[6]) )
            KP = K*self.P[:,2,np.newaxis]                   # term for t**3
            Q[:,2] = ( KP[4]  +  KP[5] 
                   + ((KP[2]+KP[8]) + (KP[9]+KP[7]) + KP[0]) 
                   + ((KP[3]+KP[10]) + KP[6]) )
            KP = K*self.P[:,3,np.newaxis]                   # term for t**4
            Q[:,3] = ( ((KP[3]+KP[7]) + (KP[6]+KP[5]) + KP[4]) 
                   + ((KP[9]+KP[8]) + (KP[2]+KP[10]) + KP[0]) )
            KP = K*self.P[:,4,np.newaxis]                   # term for t**5
            Q[:,4] = ( (KP[9]+KP[8])  +  ((KP[6]+KP[5]) + KP[4]) 
                   + ((KP[3]+KP[7]) + (KP[2]+KP[10]) + KP[0]) )
            KP = K*self.P[:,5,np.newaxis]                   # term for t**6
            Q[:,5] = ( KP[4]  +  ((KP[9]+KP[7]) + (KP[6]+KP[5]))  
                   + ((KP[3]+KP[8]) + (KP[2]+KP[10]) + KP[0]) )
            # this is almost the same as Q usual
            
            # Rksuite uses horners rule to evaluate the polynomial. Moreover,
            # the polynomial definition is different: looking back from the end 
            # of the step instead of forward from the start. 
            # The call is modified accordingly:
            return HornerDenseOutput(self.t, self.t+h, self.y, Q)
        
        elif self.dense_output_order=='low':
            # as usual:
            Q = self.K.T.dot(self.Pfree)
            return RkDenseOutput(self.t_old, self.t, self.y_old, Q)
        
        else:
            raise ValueError("`dense_output_order` must be 'high' or 'low',"
                f" not{self.dense_output_order}.")


class HornerDenseOutput(RkDenseOutput):
    """use Horner's rule for the evaluation of the polynomials"""
    def _call_impl(self, t):
        
        # scaled time
        x = (t - self.t_old) / self.h
        
        # Horner's rule:
        y = np.zeros((self.Q.shape[0], x.size))
        for q in reversed(self.Q.T):
            y += q[:,np.newaxis]
            y *= x
        
        # finish:
        y *= self.h
        y += self.y_old[:,np.newaxis]
        return y
    
