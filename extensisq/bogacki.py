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
    order interpolator is also available as method BS45_i.
    
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
    .. [1] P. Bogacki, L.F. Shampine, "An efficient Runge-Kutta (4,5) pair",
           Computers & Mathematics with Applications, Vol. 32, No. 6, 1996, 
           pp. 15-28, ISSN 0898-1221.
           https://doi.org/10.1016/0898-1221(96)00141-1
    .. [2] RKSUITE: https://www.netlib.org/ode/rksuite/
    """
    
    order = 5
    error_estimator_order = 4
    n_stages = 7        # the effective nr (total nr of stages is 8)
    n_extra_stages = 3  # for dense output
    dense_output_order = 'high'
    
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
    E1 = np.array([-3/1280, 0, 6561/632320, -343/20800, 243/12800, -1/95])
    
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
    # the approach in [3]_ (docstring of BS45_i).
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
    
    def __init__(self, fun, t0, y0, t_bound, **extraneous):
        super(BS45, self).__init__(fun, t0, y0, t_bound, **extraneous)
        # custom initialization to create extended storage for dense output
        # and to make the interpolator selectable
        self.K_extended = np.zeros((self.n_stages+self.n_extra_stages+1, 
                self.n), dtype=self.y.dtype)
        self.K = self.K_extended[:self.n_stages+1]
        # y_old is used for first error assessment, it should not be None
        self.y_old = self.y - self.direction * self.h_abs * self.f
        
    
    def _step_impl(self):
        # modified to include two error estimators. This saves two function 
        # evaluations for most rejected steps. (The step can still be rejected 
        # by the second error estimator, but this will be rare.)
        
        t = self.t
        y = self.y
        rtol = self.rtol
        atol = self.atol
        y_old = self.y_old
        
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
            y_new = y + self.K[:-1].T @ self.B * h
            
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
        dy = self.K[:i,:].T @ self.A[i,:i] * h
        self.K[i] = self.fun(self.t + self.C[i]*h, self.y + dy)
    
    def _estimate_error(self, E, h):
        # pass E instead of K
        return self.K[:E.size,:].T @ E * h
    
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
                dy = K[:s,:].T @ a[:s] * h
                K[s] = self.fun(self.t_old + c * h, self.y_old + dy)
            
            # form Q. Usually: Q = K.T @ self.P
            # but rksuite recommends to group summations to mitigate roundoff:
            Q = np.empty((K.shape[1], self.P.shape[1]), dtype=K.dtype)
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
        
        else:                                   # self.dense_output_order=='low'
            # for BS45_i 
            # as usual:
            Q = self.K.T @ self.Pfree
            return RkDenseOutput(self.t_old, self.t, self.y_old, Q)
        
    

class BS45_i(BS45):
    """As BS45, but with free 4th order interpolant for dense output. Suffix _i
    for interpolant.
    
    The source [1]_ refers to the thesis of Bogacki for a free interpolant, but 
    this could not be found. Instead, the interpolant is constructed following 
    the steps in [3]_. 

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
    
    dense_output_order = 'low'


class HornerDenseOutput(RkDenseOutput):
    """use Horner's rule for the evaluation of the polynomials"""
    def _call_impl(self, t):
        
        # scaled time
        x = (t - self.t_old) / self.h
        
        # Horner's rule:
        y = np.zeros((self.Q.shape[0], x.size), dtype=self.Q.dtype)
        for q in reversed(self.Q.T):
            y += q[:,np.newaxis]
            y *= x
        
        # finish:
        y *= self.h
        y += self.y_old[:,np.newaxis]
        
        # need this `if` to pass scipy's unit tests. I'm not sure why.
        if t.shape:
            return y
        else:
            return y[:,0]

if __name__ == '__main__':
    """Construction of a free interpolant of the BS45 pair. The approach from 
    "Runge-Kutta pairs of order 5(4) satisfying only the first column 
    simplifying assumption" by Ch Tsitouras is followed.
    Bogacki has derived an interpolant for this method as well, but I was not 
    able to find a copy of his thesis that contains this interpolant.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import sympy 
    from sympy.solvers.solveset import linsolve
    from sympy import Rational as R
    from pprint import pprint
    
    n_stages = 8            # including derivative evaluation at end of step
    order = 5               # of interpolation in t (not h)
    T5_method4 = 1.06e-4    # error of embedded fourth order method
    
    t = sympy.symbols('t', real=True)
    bi = sympy.symbols(f'bi0:{n_stages}', real=True)
    bi_vec = sympy.Matrix(bi)
    
    # Method
    A = sympy.Matrix([      # full A matrix, including last line
            [0, 0, 0, 0, 0, 0, 0, 0],
            [R(1,6), 0, 0, 0, 0, 0, 0, 0],
            [R(2,27), R(4,27), 0, 0, 0, 0, 0, 0],
            [R(183,1372), R(-162,343), R(1053,1372), 0, 0, 0, 0, 0],
            [R(68,297), R(-4,11), R(42,143), R(1960,3861), 0, 0, 0, 0],
            [R(597,22528), R(81,352), R(63099,585728), R(58653,366080), 
                    R(4617,20480), 0, 0, 0],
            [R(174197,959244), R(-30942,79937), R(8152137,19744439), 
                    R(666106,1039181), R(-29421,29068), R(482048,414219), 0, 0],
            [R(587,8064), 0, R(4440339,15491840), R(24353,124800), R(387,44800), 
                    R(2152,5985), R(7267,94080), 0]])
    
    c = sympy.Matrix([0, R(1,6), R(2,9), R(3,7), R(2,3), R(3,4), 1, 1])
    e = sympy.Matrix([1, 1, 1, 1, 1, 1, 1, 1])

    # error terms up to order 4
    c2 = c.multiply_elementwise(c)
    Ac = A*c
    
    c3 = c2.multiply_elementwise(c)
    cAc = c.multiply_elementwise(Ac)
    Ac2 = A*c2
    A2c = A*Ac
    
    T11 = bi_vec.dot(e)     -  t
    T21 = bi_vec.dot(c)     -  t**2/2

    T31 = bi_vec.dot(c2)/2  -  t**3/6
    T32 = bi_vec.dot(Ac)    -  t**3/6

    T41 = bi_vec.dot(c3)/6  -  t**4/24
    T42 = bi_vec.dot(cAc)   -  t**4/8
    T43 = bi_vec.dot(Ac2)/2 -  t**4/24
    T44 = bi_vec.dot(A2c)   -  t**4/24
    
    # solve polynomials to let all terms up to order 4 vanish
    bi_vec_t = sympy.Matrix(linsolve([T11, T21, T31, T32, T41, T42, T43, T44], 
        bi).args[0])
    i_free_poly = [i for i, (term, poly) in enumerate(zip(bi_vec_t, bi)) 
                    if poly == term]
    free_polys = [bi[i] for i in i_free_poly]
    print('free polynomials:', free_polys)      # polys bi5 and bi_7 are free
    
    # Make these free polynommials explicit in t
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
    C0_0 = [eq for eq in bi_vec_t.subs(t,0)]
    # C0 at t=1
    C0_1 = [eq for eq in (bi_vec_t.subs(t,1) - A[-1,:].T)] 
    # C1 at t=0
    C1_0 = d_bi_vec_t.subs(t,0)
    C1_0[0] = C1_0[0] - 1
    C1_0 = [eq for eq in C1_0]
    # C1 at t=1
    C1_1 = d_bi_vec_t.subs(t,1)
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
    free_params = [p for s, p in zip(sol1, params) if s==p]
    print('free parameters:', free_params) 
    # remaining free parameters: bi5_4, bi5_5, bi7_4, bi7_5
    
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
    
    T51 = bi_vec_t.dot(c4)/24   -  t**5/120
    T52 = bi_vec_t.dot(c2Ac)/2  -  t**5/20
    T53 = bi_vec_t.dot(Ac_Ac)/2 -  t**5/40
    T54 = bi_vec_t.dot(cAc2)/2  -  t**5/30
    T55 = bi_vec_t.dot(Ac3)/6   -  t**5/120
    T56 = bi_vec_t.dot(cA2c)    -  t**5/30
    T57 = bi_vec_t.dot(A_cAc)   -  t**5/40
    T58 = bi_vec_t.dot(A2c2)/2  -  t**5/120
    T59 = bi_vec_t.dot(A3c)     -  t**5/120
    
    # error norm 5 (omitting square root for simplification)
    T5_norm_t = (T51**2 + T52**2 + T53**2 + T54**2 + T55**2 + T56**2 + T57**2 
                + T58**2 + T59**2)
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
    #~ T5_norm_i = sympy.integrate(T5_norm_t, (t, 0, 1))
    #~ print('optimal T5 integrated:', T5_norm_i.evalf())
    
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
    plt.xlim(0,1)
    plt.ylim((0, 6e-4))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'error $\hat{T}_5$')
    plt.legend(loc=1,ncol=2)
    plt.title('free interpolant for BS45_i')
    plt.tight_layout()
    plt.savefig('free interpolant for BS45_i')
    plt.show()