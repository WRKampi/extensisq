import numpy as np
from warnings import warn
from extensisq.common import RungeKutta


class Pr7(RungeKutta):
    """Explicit Runge-Kutta method by Prince [1]_, developed around a
    continuous method of order 6. Discrete propagation is of order 7 and the
    solution for error estimation and stepsize selection is of order 5.

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
        is done if many steps fail and each time nfev exceeds integer multiple
        of `nfev_stiff_detect`. For the assessment itself, the problem is
        assessed as non-stiff if the predicted nfev to complete the integration
        is lower than `nfev_stiff_detect`. The default value is 5000.
    sc_params : tuple of size 3, "standard", "G", "H", "S", "C", or "H211b"
        Parameters for the stepsize controller (k*b1, k*b2, a2). The
        controller is as defined in [2]_, with k the exponent of the standard
        controller, _n for new and _o for old:
            h_n = h * (tol/err)**-b1 * (tol/err_o)**-b2  * (h/h_o)**-a2
        Predefined coefficients are Gustafsson "G" (0.7,-0.4,0), Soederlind "S"
        (0.6,-0.2,0), Hairer "H" (1,-0.6,0), central between these three "C"
        (0.7,-0.3,0), Soederlind's digital filter "H211b" (1/4,1/4,1/4) and
        "standard" (1,0,0). Standard is currently the default.

    References
    ----------
    .. [1] P.J. Prince, "Parallel Derivation of Efficient Continuous/Discrete
           Explicit Runge-Kutta Methods", Guisborough TS14 6NP U.K.,
           September 6 2018.
           http://www.peteprince.co.uk/parallel.pdf
    .. [2] G. Soederlind, "Digital Filters in Adaptive Time-Stepping", ACM
           Trans. Math. Softw. Vol 29, No. 1, 2003, pp. 1–26.
           https://doi.org/10.1145/641876.641877
    """

    order = 7
    error_estimator_order = 5
    n_stages = 10
    tanang = 7.5
    stbrad = 4.1

    C = np.array([0, 1/6, 1/4, 1/2, 1/2, 3/16, 3/16, 3/5, 6/7, 1])

    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/16, 3/16, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/4, -3/4, 1, 0, 0, 0, 0, 0, 0, 0],
        [1/12, 0, 1/3, 1/12, 0, 0, 0, 0, 0, 0],
        [9/128, 135/1024, -3/256, -57/4096, 45/4096, 0, 0, 0, 0, 0],
        [129/2048, 0, -117/1024, 207/16384, -135/16384, 15/64, 0, 0, 0, 0],
        [36/625, 0, 36/625, -288/15625, 4023/15625, -14592/78125,
            33792/78125, 0, 0, 0],
        [51237/285719, 0, 4580136/2000033, -419616/2000033,
            -2586870/2000033, -8901120/2000033,  3840000/1294139,
            30234375/22000363, 0, 0],
        [-1396/1647, 0, -19396/1281, 944/1281, 105447/10675, 494336/19215,
            -213085184/15852375, -2575625/380457, 530621/617625, 0],
    ])

    B = np.array([179/3240, 0, 0, 0, 88/375, 0, 2097152/7239375,
                  3125/21384, 285719/1215000, 61/1560])

    E = np.array([575/6912, 0, 1/2, 1/40, -1/5, -128/225, 45568/96525,
                  34625/76032, 31213/172800, 175/3328, 0])
    E[:-1] -= B

    P = np.array([
        [1, -67/12, 392/27, -4205/216, 587/45, -280/81],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, -432/25, 528/5, -5448/25, 23696/125, -896/15],
        [0, 0, 0, 0, 0, 0],
        [0, 524288/53625, -3670016/96525, 29360128/482625,
            -12058624/268125, 3670016/289575],
        [0, 3125/198, -59375/594, 1559375/7128, -59375/297, 175000/2673],
        [0, -16807/4500, 16807/675, -4823609/81000, 1025227/16875,
            -134456/6075],
        [0, 19764/302107, -160857/604214, -2003423/2416856,
            3669821/1510535, -1233176/906321],
        [0, 45225/46478, -157380/23239, 844295/46478, -485844/23239,
            198464/23239],
    ])


class Pr8(RungeKutta):
    """Explicit Runge-Kutta method by Prince [1]_, developed around a
    continuous method of order 7. Discrete propagation is of order 8 and the
    solution for error estimation and stepsize selection is of order 6.

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
            h_n = h * (tol/err)**-b1 * (tol/err_o)**-b2  * (h/h_o)**-a2
        Predefined coefficients are Gustafsson "G" (0.7,-0.4,0), Soederlind "S"
        (0.6,-0.2,0), Hairer "H" (1,-0.6,0), central between these three "C"
        (0.7,-0.3,0), Soederlind's digital filter "H211b" (1/4,1/4,1/4) and
        "standard" (1,0,0). Standard is currently the default.

    References
    ----------
    .. [1] P.J. Prince, "Parallel Derivation of Efficient Continuous/Discrete
           Explicit Runge-Kutta Methods", Guisborough TS14 6NP U.K.,
           September 6 2018.
           http://www.peteprince.co.uk/parallel.pdf
    .. [2] G. Soederlind, "Digital Filters in Adaptive Time-Stepping", ACM
           Trans. Math. Softw. Vol 29, No. 1, 2003, pp. 1–26.
           https://doi.org/10.1145/641876.641877
    """

    order = 8
    error_estimator_order = 6
    n_stages = 13
    tanang = 6.0
    stbrad = 4.5

    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [9.33333333333333333333333333333E-2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3.5E-2, 1.05E-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.4E-1, -4.2E-1, 5.6E-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4.66666666666666666666666666667E-2, 0,
         1.86666666666666666666666666667E-1,
         4.66666666666666666666666666667E-2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5.51743197278911564625850340136E-1,
         -1.68355389030612244897959183674E0,
         1.28544855442176870748299319728E0,
         -1.02147251199586977648202137998E0,
         1.3428346506013119533527696793E0, 0, 0, 0, 0, 0, 0, 0, 0],
        [6.24295812074829931972789115646E-2, 0,
         1.1561591754239009036450401056E-1,
         -2.48173453254069484936831875607E-1,
         4.76004820175838192419825072886E-1,
         6.9123134328358208955223880597E-2, 0, 0, 0, 0, 0, 0, 0],
        [5.31754385964912280701754385965E-2, 0,
         1.07960199004975124378109452736E-1,
         -4.55448717948717948717948717949E-2,
         2.13141025641025641025641025641E-2,
         1.80954510721537509608660353975E-2,
         -1.50003194428508726399201574995E-2, 0, 0, 0, 0, 0, 0],
        [4.66308628714643752237737200143E-2, 0,
         5.89926174413351732228314575591E-2,
         -3.85681391925321394464275175145E-2,
         2.31207294594081206081229649243E-2,
         1.33009364423326992630144289062E-2,
         -1.12383747568330718937876881162E-2,
         -1.22386322651751569775273657731E-2, 0, 0, 0, 0, 0],
        [5.44217687074829931972789115646E-2, 0,
         5.40333156097836021965644146011E-1,
         1.13688418211922452259942034977E-1,
         7.13352588752466984012721183007E-2,
         5.33742685221169022273976869374E-3,
         1.41741045931070379958169549861E-1,
         -3.93523741342436902671713196074E-1, 0, 0, 0, 0, 0],
        [-2.5940248582906658528380592843E0, 0,
         -1.77182396161676339315977974886E1,
         -7.20291569558395826074056304001E0,
         1.49881506312855377166637034571E1,
         6.91875347236946677726180862128E-1,
         -7.1579553906147432099663790337E0,
         2.26593581547052770034800605954E0,
         1.31756939077540917088647769884E1,
         4.35147985890989745154013147945E0, 0, 0, 0],
        [2.13650273135026061952015557422E0, 0,
         1.22726517463837019805397194412E1,
         4.57653009750266308294439689668E0,
         -1.02693748349496721494231277368E1,
         -3.7629577005440390399992522631E-1,
         5.13017034156407434494186004831E0,
         9.62489338117485142433512784366E-2,
         -1.03442766767566431388674762321E1,
         -2.59772774629732852847083885323E0,
         2.55571177445599178571884809579E-1, 0, 0],
        [-1.35769285555186849895417607979E0, 0,
         -5.1848880697594401985846769119E0,
         -9.44146275842569180247520769719E-1,
         5.7895230679047775815740126675E0,
         -8.78331322343203825844350417575E-2,
         -5.3269980917879320756691248069E0,
         -3.29419214246872762413611365341E0,
         7.12498180803033672915227750425E0,
         4.23294185993291744523268528268E0,
         -4.04195634685480312241204160092E-1,
         4.52499466462306516458275969143E-1, 0],
    ])

    C = np.array([0, 7/75, 7/50, 7/25, 7/25, 19/40, 19/40, 7/50, 2/25, 8/15,
                  4/5, 22/25, 1])

    B = np.array([
        1.01794429109647960346134352579E-2, 0, 0, 0,
        3.42368385444252768240392979897E-1, 0,
        -2.93579170210195254952533184107E-1,
        -9.95353542504925310796528261447E-2,
        2.07437934572206346408597479637E-1,
        5.43021201240774613165496369804E-1,
        1.07235531414217311653209089107E-1,
        1.44540738194287577003626386342E-1,
        3.8331290683984373526250270207E-2])

    E = np.array([
        -5.91293258004837089685158123242E-2, 0, 0, 0,
        -1.14901053624678659268733720599E0, 0,
        3.77974861322618360761792589719E0,
        7.48570483449616205367366755746E-1,
        1.74577267785426015996335159381E-2,
        -3.00845790812659142174173541204E0,
        9.70820946719519308812662261486E-1, -4.0E-1, 1.0E-1, 0])
    E[:-1] -= B

    P = np.array([
        [1, -1.41239093743887554093438640359E1,
            8.4016766195696458031461586846E1,
            -2.51083997467010984848468599308E2,
            3.86512292414528095658571249453E2,
            -2.91148854238488538278621434142E2,
            8.48378819125746896424356746212E1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 5.10665844292158893795819184009E0,
            -4.00124279090280806878089311459E1,
            1.84440467426588995836723836551E2,
            -3.78973592698221239000827838379E2,
            3.41538781236289637930611683348E2,
            -1.11757518113106650248416549234E2],
        [0, 0, 0, 0, 0, 0, 0],
        [0, -2.0751485626953971018794503634E0,
            -9.92180312887292232371883540967E0,
            9.25598164917894535323732388042E1,
            -2.15979497620388810544292166846E2,
            2.02287446902906568511069478446E2,
            -6.71643932529490873285047978155E1],
        [0, -2.14183863541985721594046947605E1,
            2.2543825665248598292363025894E2,
            -8.84829314979143198157611115581E2,
            1.57460856803292263565777875922E3,
            -1.29008703344712019545777930343E3,
            3.96188374740802854662306442791E2],
        [0, 3.15180535325053411373578467654E1,
            -2.71413850918964647969857885217E2,
            9.52348654179583905315594876659E2,
            -1.59274235351047950380282579202E3,
            1.25795594603754462227636944346E3,
            -3.7745901138561751061022989217E2],
        [0, 9.65701535222266902447206365602E-1,
            1.29274299904241623819239369537E1,
            -1.00480323438077279181257364969E2,
            2.45834235483979601485491021852E2,
            -2.42964579882849927174983253853E2,
            8.42605575125419501995439500197E1],
        [0, 8.35557991425411575752552757343E-2,
            -3.81989088470777655385269085515E-1,
            1.70967961712293483156233961215E0,
            -5.28883927213581506454906269727E0,
            7.30946577097275164964837014524E0,
            -3.32463729521741760719842416123E0],
        [0, -6.33380888284421342020961449909E-2,
            -1.09526815610176510133599623253E0,
            8.61906649612399098614496017804E0,
            -2.30612339267248796358151679261E1,
            2.59748166009870704285753056776E1,
            -1.02295021872616869663633791656E1],
        [0, -2.84608939125612693786991431373E-1,
            4.29888914910786433803628796472E-3,
            5.20337256488446600050553460014E0,
            -1.46885666621411381932249338734E1,
            1.54411838034484010111998278429E1,
            -5.63734836553123961550522315598E0],
        [0, 2.91422009445041363278596489354E-1,
            4.38587473682482536753098062593E-1,
            -8.48742089186228431556770654766E0,
            2.37789877586610534396939312211E1,
            -2.63071727836903908960901174964E1,
            1.0285596433764097871932198271E1],
    ])


class Pr9(RungeKutta):
    """Explicit Runge-Kutta method by Prince [1]_, developed around a
    continuous method of order 8. Discrete propagation is of order 9 and the
    solution for error estimation and stepsize selection is of order 7.

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
            h_n = h * (tol/err)**-b1 * (tol/err_o)**-b2  * (h/h_o)**-a2
        Predefined coefficients are Gustafsson "G" (0.7,-0.4,0), Soederlind "S"
        (0.6,-0.2,0), Hairer "H" (1,-0.6,0), central between these three "C"
        (0.7,-0.3,0), Soederlind's digital filter "H211b" (1/4,1/4,1/4) and
        "standard" (1,0,0). Standard is currently the default.

    References
    ----------
    .. [1] P.J. Prince, "Parallel Derivation of Efficient Continuous/Discrete
           Explicit Runge-Kutta Methods", Guisborough TS14 6NP U.K.,
           September 6 2018.
           http://www.peteprince.co.uk/parallel.pdf
    .. [2] G. Soederlind, "Digital Filters in Adaptive Time-Stepping", ACM
           Trans. Math. Softw. Vol 29, No. 1, 2003, pp. 1–26.
           https://doi.org/10.1145/641876.641877
    """

    order = 9
    error_estimator_order = 7
    n_stages = 17
    tanang = 7.5
    stbrad = 4.6

    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.03703703703703703703703703704E-1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3.88888888888888888888888888889E-2,
            1.16666666666666666666666666667E-1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.55555555555555555555555555556E-1,
            -4.66666666666666666666666666667E-1,
            6.22222222222222222222222222222E-1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5.18518518518518518518518518519E-2,
            0,
            2.07407407407407407407407407407E-1,
            5.18518518518518518518518518519E-2,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.43904006046863189720332577476E0,
            -4.29081632653061224489795918367E0,
            2.79697656840513983371126228269E0,
            -3.79589677140697548860814166937E0,
            4.49514091350826044703595724004E0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.15627362055933484504913076342E-1,
            0,
            -4.18968597540026111454682883254E-2,
            -1.02048779829392074290033473707E0,
            1.48135325558794946550048590865E0,
            1.09848484848484848484848484849E-1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [6.01213282247765006385696040869E-2,
            0,
            1.16077441077441077441077441078E-1,
            -4.08333333333333333333333333333E-2,
            1.87962962962962962962962962963E-2,
            7.59713798041186071308940908624E-3,
            -6.20331469003684620014386165793E-3,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [7.51874092848346235045742434905E-2,
            0,
            3.08577739962730314518376303984E0,
            1.02777024339893729511937200674E0,
            -7.05175775924447499201004659796E-1,
            -2.78778538145998406742566524436E-2,
            2.85805153481527740360777028619E-1,
            -2.99148657605355546429322500644E0,
            0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.78548047149894440534834623505E-2,
            0,
            5.16426091974317668823733047791E-3,
            -3.43681317296491684696604004926E-3,
            2.32372133623022296941501964109E-3,
            5.14201281710644115584844336227E-4,
            -4.33506924343872955710236185757E-4,
            -1.98666815536469802404438057068E-3,
            0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1.18757288815909505564677978471E0,
            0,
            1.13239724969350324395658487817E-1,
            4.15212324887617856117414455329E-2,
            6.55529796136042083177053423077E-1,
            -1.43685337376884729223927086404E-3,
            5.0120866241769657823310092071E-3,
            -7.07903479324387044990828600376E-1,
            2.87573843923931191319441255834E-1,
            1.66070320338165526431028870115E0,
            0, 0, 0, 0, 0, 0, 0],
        [2.9326637612934925469041347559E-2,
            0,
            2.27648379357749274981640086384E-1,
            6.42188982905375335427504387651E-2,
            4.54015730358590118030630636463E-2,
            -7.94683657147007110737301019097E-4,
            2.4861235148309752048563545005E-3,
            -3.8188748663301078106113860815E-2,
            -6.35233112974332282906394142683E-4,
            3.05370536215106964984062651216E-2,
            0, 0, 0, 0, 0, 0, 0],
        [-6.0881392683555120739135947005E-2,
            0,
            -4.43438802229705377088167804351E-3,
            4.85920262412870550959705396755E-2,
            -1.23883926446739678566391896451E-1,
            -5.40541062978150844511169211286E-3,
            2.22380334943109355360348696484E-3,
            3.9784387690238541442182630348E-2,
            4.00102754975348157308270345129E-4,
            1.39498241761820581940505774544E-1,
            1.11825158329569318169361499236E-4,
            5.9232826064386410109019245474E-2,
            0, 0, 0, 0, 0],
        [-1.08294884409577446574523573542E1,
            0,
            -4.58555384056239216218818778683E1,
            -1.32193447471870660838992789085E1,
            3.05914618343960445376558308999E1,
            1.90922815696523943781555810831E-1,
            -1.88814404888067339077866227303E0,
            2.120320047452681450999642799E1,
            1.58374092516843429808128896804E0,
            1.01434070966925954939865647766E1,
            -3.75623558814887119657201098616E-1,
            -8.5871515129714592091127087542E0,
            1.75314464568442281881693067004E1,
            0, 0, 0, 0],
        [1.23847409176679080375217053279E0,
            0,
            1.09333898278372589827844780651E0,
            3.31536159352838127791365049203E-1,
            -2.09936149938613061202956506086E0,
            -6.32995160818675610148594925667E-3,
            8.53402652892084128305708186158E-2,
            -7.13566595489746263585587501118E-2,
            -2.39880075764678496228135351925E-2,
            -1.65181846577159146362072750765E0,
            6.6245576312291875410038570643E-3,
            1.58274616357353025449630677489E0,
            1.41814358816879890444609368724E-1,
            9.34364104078509623504023164074E-3,
            0, 0, 0],
        [-1.72854883427149906480834787355E1,
            0,
            -8.51259795469259804679437273731E0,
            -2.80551782495675369923611987322E0,
            3.51919004458313930717487691522E1,
            7.36701829077372132621075400047E-2,
            -1.2458454706806936083664523748E0,
            6.34431710970183708662917963423E-1,
            -2.99382194728825930237778557253E0,
            2.58384480400235896878583385841E1,
            7.60451062344283617975442262104E-1,
            -2.72531371870126046405092063507E1,
            -7.66905838615669874875007702991E0,
            -4.30717080246156504921654666097E-2,
            6.242970712783360378435415972E0,
            0, 0],
        [2.38352607101136790911153507492E1,
            0,
            7.7738756188986200797865735259E0,
            2.57266456224187218683628421487E0,
            -4.27301289332539628209504430154E1,
            -6.84309506233548912384135730021E-2,
            1.30451149660509620846896301402E0,
            3.34387872500134107823468680391E0,
            2.90875579291685107583165783656E0,
            -3.51867845271413839756884852364E1,
            -3.99562978143036844261012184204E-1,
            3.35175018689652853293607378584E1,
            9.93231523768212904432842366324E0,
            8.37781070877954481643354905414E-2,
            -5.98649811509162584079618360502E0,
            9.88633847406948308075244574403E-2,
            0],
    ])

    C = np.array([0, 14/135, 7/45, 14/45, 14/45, 29/45, 29/45, 7/45, 3/4, 1/50,
                  13/15, 0.36, 2/21,  22/45, 7/11, 14/15, 1])

    B = np.array([
        3.07799270475412360737919532866E-2,
        0, 0, 0, 0, 0, 0,
        9.27611792928056326667152622109E-2,
        1.22968704959599330688571533231E-2,
        0,
        1.54432519665121535509579924022E-1,
        2.62981836369090817847088871306E-1,
        1.08761305196919820431285749084E-1,
        5.88466616840491098220532404464E-3,
        2.60270801998885728778978732780E-1,
        4.29012449776118791099335675107E-2,
        2.89296487876585055315634624318E-2])

    E = np.array([
        -1.27690537577646353299981537083E-1,
        0, 0, 0, -9.5E-1, 0, 2.5E-2,
        4.20967933030044959322296039676E-1,
        4.73337930270267175222242790533E-1,
        2.5E-1,
        -4.31855604704258756750593362625E-2,
        1.18421357803228938463464105438E0,
        -1.52346033048153512791475335877E-1,
        2.79212408462991738161601055159E-2,
        -2.13862748653591616107674163303E-1,
        5.43489879984277053495778555136E-2,
        5.12952095724889595292725269046E-2,
        0])
    E[:-1] -= B

    P = np.array([
        [1, -2.77735116893550678899410577323E1,
            2.5165978861779362829892054496E2,
            -1.08626391924281399602886402221E3,
            2.53356539924885542940296045855E3,
            -3.24230250171399947308061311167E3,
            2.13055284911623123780143489376E3,
            -5.60407324409664217267823913703E2],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1.18697364879199052960842505705E1,
            1.02459915482709922665427835603E2,
            -1.16095838290346893556824162957E2,
            -6.70639976556572401329871695175E2,
            1.94483721571669034924983554886E3,
            -1.85445230708830610183339303496E3,
            6.057607272237450301009097592E2],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2.65383553931689365018564794762E-1,
            -3.84957223533893082531676747188E0,
            1.98583488086398818369789716893E1,
            -4.89425851614840465067296601841E1,
            6.25680078034413170103241113952E1,
            -4.00860955038855039119026804046E1,
            1.01865127346955930316274601813E1],
        [0, 4.10881688594931446561283260961E-1,
            3.12020001032662815102399649694E1,
            -2.49707701312674989659598434886E2,
            7.78195589210511825395485214743E2,
            -1.17201077920519207960027474654E3,
            8.50532394815990090483253786076E2,
            -2.38529624121203253943000352364E2],
        [0, 4.14906689983933914089354123183E0,
            -6.0201083012212029122532262984E1,
            3.1340256450693611124013315901E2,
            -7.81911886672181247720231054571E2,
            1.01249775080717639184821281449E3,
            -6.56660710999561695264414176554E2,
            1.6873659534049908981100683653E2],
        [0, 2.70087958974258880514796095306E1,
            -3.08300187466966862891517091984E2,
            1.44655177948042279851428814073E3,
            -3.51700174110387571456172732356E3,
            4.60841657049002751224578802649E3,
            -3.07258687363984008871483884247E3,
            8.15911656342806467356527481271E2],
        [0, -2.15215229602329501142575462752E0,
            3.14196745583973862967039208679E1,
            -1.6532985104192405065217131984E2,
            4.18414305050309583418887171955E2,
            -5.53657581775559173675697808708E2,
            3.70441653709359610572530943729E2,
            -9.89816156848949394133175734526E1],
        [0, 1.15334997254946792039603096315E1,
            -1.18814083869203384405600072643E2,
            3.08134567500691900387880121103E2,
            2.33436352026189303075135007582E1,
            -9.66772385163294883072158319407E2,
            1.15794887039996694701727086852E3,
            -4.15111121959905098621019319093E2],
        [0, 2.15636542120343295525422617798E0,
            2.23094376381637869726414340883E1,
            -2.14860691636078809627896996708E2,
            6.77621572331989459966523793888E2,
            -1.0140266973350173253494403702E3,
            7.31016153941917335279615104043E2,
            -2.04107379056980960376265905544E2],
        [0, 5.08899586642482571660412314383E-2,
            -7.42202087910459104358965739767E-1,
            4.42595830228471928946003912407E0,
            -1.24465247279828531500574820871E1,
            1.80205784909925668387118635985E1,
            -1.29526679902614061184352135296E1,
            3.64985272038158889849592272644E0],
        [0, -4.43001472145292266932794194954E0,
            6.21578560270965455283632655849E1,
            -3.07676880364428105577295947394E2,
            7.15128713595346851864396276583E2,
            -8.41901562801894856570402038232E2,
            4.86997480884687474061182189924E2,
            -1.10015321817356100908136825784E2],
        [0, 6.39746391350199829505667086882E-1,
            -8.92447331136320993314728274381E0,
            4.39375977273632452580211911017E1,
            -1.00055397522875981592003636938E2,
            1.13223500868691469540904428159E2,
            -6.09041019096722296163008301721E1,
            1.21260290014841183921303970742E1],
        [0, -3.42936518565914843445423340005E-2,
            1.02263992789510082522682584494E0,
            -7.67553082732108617644153808883E0,
            2.87682707734889980917298956223E1,
            -5.45304700130068082381237392245E1,
            4.87822504767962797310095764161E1,
            -1.63039370372082342435249147736E1],
        [0, 4.5079310103374101284304267947E-2,
            -1.39971037232777581505134835254E0,
            1.12995963892492747523307993272E1,
            -4.40393736681488335868754595851E1,
            8.56383538309449928529333409909E1,
            -7.86288962134219494870125843802E1,
            2.70849507236009171823909477318E1],
    ])


# old class names
class Pri6(Pr7):
    def __init__(self, *args, **kwargs):
        warn("This method will be replaced by 'Pr7'.", FutureWarning)
        super(Pri6, self).__init__(*args, **kwargs)


class Pri7(Pr8):
    def __init__(self, *args, **kwargs):
        warn("This method will be replaced by 'Pr8'.", FutureWarning)
        super(Pri7, self).__init__(*args, **kwargs)


class Pri8(Pr9):
    def __init__(self, *args, **kwargs):
        warn("This method will be replaced by 'Pr9'.", FutureWarning)
        super(Pri8, self).__init__(*args, **kwargs)
