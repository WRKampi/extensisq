import numpy as np
from extensisq.common import RungeKutta


class CFMR7osc(RungeKutta):
    """Explicit Runge-Kutta method of (algebraic) order 7, with an error
    estimate of order 5 and a free interpolant of order 5.

    This method by Calvo, Franco, Montijano and Randez is tuned to get a
    dispersion order of 10 and a dissipation order of 11. This is beneficial
    for problems with oscillating solutions. It can outperform methods of
    higher algebraic order, such as `DOP853`, for these problems.

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
    sc_params : tuple of size 3, "standard", "G", "H", "S", "C", or "H211b"
        Parameters for the stepsize controller (k*b1, k*b2, a2). The
        controller is as defined in [2]_, with k the exponent of the standard
        controller, _n for new and _o for old:
            h_n = h * (err/tol)**-b1 * (err_o/tol_o)**-b2  * (h/h_o)**-a2
        Predefined coefficients are Gustafsson "G" (0.7,-0.4,0), Soederlind "S"
        (0.6,-0.2,0), Hairer "H" (1,-0.6,0), central between these three "C"
        (0.7,-0.3,0), Soederlind's digital filter "H211b" (1/4,1/4,1/4) and
        "standard" (1,0,0). Standard is currently the default.

    References
    ----------
    .. [1] M. Calvo, J.M. Franco, J.I. Montijano, L. Randez, "Explicit
           Runge-Kutta methods for initial value problems with oscillating
           solutions", Journal of Computational and Applied Mathematics, Vol.
           76, No. 1–2, 1996, pp. 195-212.
           https://doi.org/10.1016/S0377-0427(96)00103-3
    .. [2] G. Soederlind, "Digital Filters in Adaptive Time-Stepping", ACM
           Trans. Math. Softw. Vol 29, No. 1, 2003, pp. 1–26.
           https://doi.org/10.1145/641876.641877
    """
    order = 7
    error_estimator_order = 5
    n_stages = 9
    tanang = 40
    stbrad = 4.7

    # time step fractions
    C = np.array([0, 4/63, 2/21, 1/7, 7/17, 13/24, 7/9, 91/100, 1])

    # coefficient matrix
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4/63, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/42, 1/14, 0, 0, 0, 0, 0, 0, 0],
        [1/28, 0, 3/28, 0, 0, 0, 0, 0, 0],
        [12551/19652, 0, -48363/19652, 10976/4913, 0, 0, 0, 0, 0],
        [-36616931/27869184, 0, 2370277/442368, -255519173/63700992,
         226798819/445906944, 0, 0, 0, 0],
        [-10401401/7164612, 0, 47383/8748, -4914455/1318761, -1498465/7302393,
         2785280/3739203, 0, 0, 0],
        [181002080831/17500000000, 0, -14827049601/400000000,
         23296401527134463/857600000000000, 2937811552328081/949760000000000,
         -243874470411/69355468750, 2857867601589/3200000000000, 0, 0],
        [-228380759/19257212, 0, 4828803/113948, -331062132205/10932626912,
         -12727101935/3720174304, 22627205314560/4940625496417,
         -268403949/461033608, 3600000000000/19176750553961, 0]])

    # coefficients for propagating method
    B = np.array([
        95/2366, 0, 0, 3822231133/16579123200, 555164087/2298419200,
        1279328256/9538891505, 5963949/25894400, 50000000000/599799373173,
        28487/712800])

    # coefficients for error estimation
    E = np.array([
        1689248233/50104356120, 0, 0, 1/4, 28320758959727/152103780259200,
        66180849792/341834007515, 31163653341/152322513280,
        36241511875000/394222326561063, 28487/712800, 0])
    E[:-1] -= B

    # coefficients for interpolation (dense output)
    # free 5th order interpolant
    P = np.array([
        [1, -5439025530946664/962612632480499,
            11749635326218168/962612632480499,
            -21678779457626505/1925225264960998,
            327710752109297/87510239316409],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 638179467680252558557/89936733884308337664,
            -14255534194656773634833/674525504132312532480,
            19906096271400751215461/899367338843083376640,
            -803769037740476397583/102200833959441292800],
        [0, -240888787178227123847/187023445194904693760,
            826261604014230263373/93511722597452346880,
            -4713020495142786403091/374046890389809387520,
            225011315479847521529/42505328453387430400],
        [0, -7385804770592231424/12829488640833179393,
            468492836426628415488/141124375049164973323,
            -3201166691676868608/754675802401951729,
            1151481881047441563648/705621875245824866615],
        [0, -64099669134317769/421407887582468864,
            -981682377584040009/1053519718956172160,
            860549546313049167/247886992695569920,
            -1033072437172556997/478872599525532800],
        [0, 223086733115140000000000/162686518334802682543323,
            -3270266198264210000000000/488059555004408047629969,
            92866234659910000000000/9569795196164863679019,
            -63469294997090000000000/14789683484982062049393],
        [0, 28487/23760, -199409/35640, 370331/47520, -199409/59400],
        [0, -2, 10, -15, 7]])
