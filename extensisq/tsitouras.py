import numpy as np
from scipy.integrate._ivp.rk import RungeKutta


class Ts45(RungeKutta):
    """Explicit Runge-Kutta method of order 5, with an error estimate of order 4
    and a free interpolant of order 4.

    This method only differs from RK45 (scipy default) by the values of its 
    coefficients. These coefficients have been derived with fewer simplifying 
    assumptions [1]_. This results in an increased efficiency in most cases.

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
        Number of evaluations of the Jacobian. Is always 0 for this solver as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the 
           first column simplifying assumption", Computers & Mathematics with 
           Applications, Vol. 62, No. 2, pp. 770 - 775, 2011.
           https://doi.org/10.1016/j.camwa.2011.06.002
    """
    
    order = 5
    error_estimator_order = 4
    n_stages = 6        # effective nr
    
    # time step fractions
    C = np.array([0, 0.161, 0.327, 0.9, 0.9800255409045097, 1])
    
    # coefficient matrix
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0.3354806554923570 , 0, 0, 0, 0],
        [0, -6.359448489975075, 4.362295432869581, 0, 0, 0],
        [0, -11.74888356406283, 7.495539342889836, -0.09249506636175525, 0, 0],
        [0, -12.92096931784711, 8.159367898576159, -0.07158497328140100, 
                -0.02826905039406838, 0.0]])
    A[:,0] = C - A.sum(axis=1)
    
    # coefficients for propagating method
    B = np.array([0.09646076681806523, 0.01, 0.4798896504144996, 
        1.379008574103742, -3.290069515436081, 2.324710524099774 ])
    
    # coefficients for error estimation
    E = np.array([0.001780011052226, 0.000816434459657, -0.007880878010262, 
            0.144711007173263, -0.582357165452555, 0.458082105929187, 
            -1/66])             # last term corrected with a minus sign
    
    # coefficients for interpolation (dense output)
    P = np.array([
        [1,  -2.763706197274826,    2.9132554618219126, -1.0530884977290216],
        [0,   0.13169999999999998, -0.2234,              0.1017            ],
        [0,   3.930296236894751,   -5.941033872131505,   2.490627285651253 ],
        [0, -12.411077166933676,   30.338188630282318, -16.548102889244902 ],
        [0,  37.50931341651104,   -88.1789048947664,    47.37952196281928  ],
        [0, -27.896526289197286,   65.09189467479368,  -34.87065786149661  ],
        [0,   1.5,                 -4.0,                 2.5               ]])


if __name__=='__main__':
    from numpy.polynomial.polynomial import Polynomial
    
    # conversion of coefficient in P from the notation in the paper to the
    # array given above
    
    # formulation in [1]_, 
    p1 = (Polynomial((0, -1.0530884977290216))
        * Polynomial((-1.3299890189751412, 1))
        * Polynomial((0.7139816917074209, -1.4364028541716351, 1)))
    p2 = (Polynomial((0, 0, 0.1017))
        * Polynomial((1.2949852507374631, -2.1966568338249754, 1)))
    p3 = (Polynomial((0, 0, 2.490627285651252793))
        * Polynomial((1.57803468208092486, -2.38535645472061657, 1)))
    p4 = (Polynomial((0, 0, -16.54810288924490272))
        * Polynomial((-1.21712927295533244, 1))
        * Polynomial((-0.61620406037800089, 1)))
    p5 = (Polynomial((-1.203071208372362603, 1))
        * Polynomial((-0.658047292653547382, 1))
        * Polynomial((0, 0, 47.37952196281928122)))
    p6 = (Polynomial((-1.2, 1)) * Polynomial((-0.666666666666666667, 1))
        * Polynomial((0, 0, -34.87065786149660974)))
    p7 = Polynomial((-0.6, 1)) * Polynomial((-1, 1)) * Polynomial((0, 0, 2.5))
    
    # convert to single polynomial
    order = 5           # of polynomial
    n_stages = 6        # effective nr
    P = np.zeros((n_stages+1, order-1))
    for i, p in enumerate((p1, p2, p3, p4, p5, p6, p7)):
        P[i,:] = p.coef[1:]
    
    np.set_printoptions(floatmode='unique')
    #~ print(P)      # P[0,0] needs manual correction
    
    P[0,0] = 1
    print(P)

