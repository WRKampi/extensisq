import numpy as np
from extensisq.common import (
    RungeKuttaNystrom, QuinticHermiteDenseOutput, HornerDenseOutputNystrom)


class Fi4N(RungeKuttaNystrom):
    """Explicit Runge-Kutta Nystrom (general) method by Fine [1]_ of order 4.
    This method is applicable to second order initial value problems only.

    The second order problem should be recast in first order form as
    u = [x, v], du = [v, a], with x, v, a variables like, position,
    velocity, acceleration. The derivative function du = f(t, u) should
    calculate only a and pass through v. (The order in u and du matters.)

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
        two options is determined by `vectorized` argument (see below). For
        this second order problem, y should contain all solution components
        first followed by an equal number of first derivative components of the
        solution. Likewise, the returned array should contain the first
        derivatives first followed by the second derivatives. (The first
        derivatives are identical those in the input and the second derivatives
        are calculated.)
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
    sc_params : tuple of size 4, "standard", "G", "H" or "W", optional
        Parameters for the stepsize controller (k*b1, k*b2, a2, g). The step
        size controller is, with k the exponent of the standard controller,
        _n for new and _o for old:
            h_n = h * g**(k*b1 + k*b2) * (h/h_o)**-a2
                * (err/tol)**-b1 * (err_o/tol_o)**-b2
        Predefined parameters are [2]_:
            Gustafsson "G" (0.7, -0.4, 0, 0.9),
            Soederlind "S" (0.6, -0.2, 0, 0.9),
            and "standard" (1, 0, 0, 0.9).
        The default for this method is "G".

    References
    ----------
    .. [1] J.M. Fine, "Low order practical Runge-Kutta-Nyström methods",
           Computing, Vol. 38, 1987, pp. 281-297.
           https://doi.org/10.1007/BF02278707
    .. [2] G.Söderlind, "Automatic Control and Adaptive Time-Stepping",
           Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310.
           https://doi.org/10.1023/A:1021160023092
    """
    n_stages = 5
    order = 4
    order_secondary = 3
    sc_params = "G"

    tanang = 40.
    stbre = 1.5
    stbim = 4.

    C = np.array([0, 2/9, 1/3, 3/4, 1])
    A = np.array([[0, 0, 0, 0, 0],
                  [2/81, 0, 0, 0, 0],
                  [1/36, 1/36, 0, 0, 0],
                  [9/128, 0, 27/128, 0, 0],
                  [11/60, -3/20, 9/25, 8/75, 0]])
    Ap = np.array([[0, 0, 0, 0, 0],
                   [2/9, 0, 0, 0, 0],
                   [1/12, 1/4, 0, 0, 0],
                   [69/128, -243/128, 135/64, 0, 0],
                   [-17/12, 27/4, -27/5, 16/15, 0]])
    B = np.array([19/180, 0, 63/200, 16/225, 1/120])
    Bp = np.array([1/9, 0, 9/20, 16/45, 1/12])
    E = np.array([25/1116, 0, -63/1240, 64/1395, -13/744, 0])
    Ep = np.array([2/125, 0, -27/625, 32/625, -3/125, 0])


class Fi5N(RungeKuttaNystrom):
    """Explicit Runge-Kutta Nystrom (general) method by Fine [1]_ of order 5.
    This method is applicable to second order initial value problems only.

    The second order problem should be recast in first order form as
    u = [x, v], du = [v, a], with x, v, a variables like, position,
    velocity, acceleration. The derivative function du = f(t, u) should
    calculate only a and pass through v. (The order in u and du matters.)

    This method can use one of 4 interpolants; see [2]_ for two options.

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
        two options is determined by `vectorized` argument (see below). For
        this second order problem, y should contain all solution components
        first followed by an equal number of first derivative components of the
        solution. Likewise, the returned array should contain the first
        derivatives first followed by the second derivatives. (The first
        derivatives are identical those in the input and the second derivatives
        are calculated.)
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
    sc_params : tuple of size 4, "standard", "G", "H" or "W", optional
        Parameters for the stepsize controller (k*b1, k*b2, a2, g). The step
        size controller is, with k the exponent of the standard controller,
        _n for new and _o for old:
            h_n = h * g**(k*b1 + k*b2) * (h/h_o)**-a2
                * (err/tol)**-b1 * (err_o/tol_o)**-b2
        Predefined parameters are [3]_:
            Gustafsson "G" (0.7, -0.4, 0, 0.9),
            Soederlind "S" (0.6, -0.2, 0, 0.9),
            and "standard" (1, 0, 0, 0.9).
        The default for this method is "G".
    interpolant : int: 0, 1, 2 or 3
        Select the interpolant for dense output.
        Option 0 is for the 4th order accurate interpolant that is the 5th
        order hermite polynomial that satisfies C2 continuity at the solution
        points. This free interpolant satisfies the RKN order conditions up to
        order 4.
        Option 1 is for the 5th order interpolant in [2]_ that needs two extra
        function evaluations (at c = 2/5 and 1/5). This interpolant changes the
        solution at the output point, which is uncommon practice.
        Option 2 is for the other 5th order interpolant in [2]_ that also nees
        two extra function evaluations (both at c = 1/2). This interpolant
        improves the interpolated velocity (1st derivative). The interpolated
        displacement (solution) is the same as for interpolant 0 as proposed in
        the paper. However, a modified interpolant is implemented in extensisq,
        which uses the extra evalutions to slightly improve the position
        interpolant.
        Option 3 is a 4th order accurate interpolant that needs 1 extra
        evaluation and is slightly more accurate than interpolant 0.
        Default: 0.

    References
    ----------
    .. [1] J.M. Fine, "Low order practical Runge-Kutta-Nyström methods",
           Computing, Vol. 38, 1987, pp. 281-297.
           https://doi.org/10.1007/BF02278707
    .. [2] J.M. Fine, "Interpolants for Runge-Kutta-Nyström Methods",
           Computing, Vol. 39, 1987, pp. 27-42.
           https://doi.org/10.1007/BF02307711
    .. [3] G.Söderlind, "Automatic Control and Adaptive Time-Stepping",
           Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310.
           https://doi.org/10.1023/A:1021160023092
    """
    n_stages = 6
    order = 5
    order_secondary = 4
    sc_params = "G"

    tanang = 15.
    stbre = 2.0
    stbim = 4.0

    C = np.array([0, 8/39, 4/13, 5/6, 43/47, 1])
    A = np.array([[0, 0, 0, 0, 0, 0],
                  [32/1521, 0, 0, 0, 0, 0],
                  [4/169, 4/169, 0, 0, 0, 0],
                  [175/5184, 0, 1625/5184, 0, 0, 0],
                  [-342497279/5618900760, 6827067/46824173,
                   35048741/102161832, -2201514/234120865, 0, 0],
                  [-7079/52152, 767/2173, 14027/52152, 30/2173, 0, 0]])
    Ap = np.array([[0, 0, 0, 0, 0, 0],
                   [8/39, 0, 0, 0, 0, 0],
                   [1/13, 3/13, 0, 0, 0, 0],
                   [7385/6912, -9425/2304, 13325/3456, 0, 0, 0],
                   [223324757/91364240, -174255393/18272848,
                    382840094/46824173, -39627252/234120865, 0, 0],
                   [108475/36464, -9633/848, 7624604/806183,
                    8100/49979, -4568212/19446707, 0]])
    B = np.array([4817/51600, 0, 388869/1216880,
                  3276/23575, -1142053/22015140, 0])
    Bp = np.array([4817/51600, 0, 1685099/3650640,
                   19656/23575, -53676491/88060560, 53/240])
    E = np.array([8151/2633750, 0, -1377519/186334750,
                  586872/28879375, -36011118/2247378875, 0, 0])
    Ep = np.array([8151/2633750, 0, -5969249/559004250,
                   3521232/28879375, -846261273/4494757750, 4187/36750, -1/25])

    # first higher order interpolant of fine that replaces endpoint,
    # needs two extra evaluations
    C_extra1 = (2/5, 1/5)
    A_extra1 = np.array([
        [3166724675977/89400687626250, 12182/175275,
         -2308196389073/59333908961250, 113223739712/2656609580625,
         -4985173058548/281912811350625, 0, -108322/9884875, 0, 0],
        [13703589067379/1021293449694000, -16588/178955775,
         393366167467741/44058124399590000, -129051960428/18967820851875,
         286445641484101/88563993965842500, 0, 185739/141153250, 0, 0]])
    Ap_extra1 = np.array([
        [-153602563/3630543750, 4/5, -15809974379/36693821250,
         25076304/131584375, -20756397983/250144464375, 0, -251893/7317375,
         0, 0],
        [549292232911/4942380225000, 0, 38673447228481/349667653965000,
         -14447155986/237691990625, 239970566676929/8434666091985000, 0,
         48691361/4597563000, 0, 0]])
    P1 = np.array([
        [1/2, -65425102193/34422618000, 246336178993/68845236000,
         -17271401477/5737103000, 2646330829/2868551500],
        [0, 0, 0, 0, 0],
        [0, -103408733716249/21918241774800, 36216248499769/2087451597600,
         -34348334365943/1826520147900, 11946681497647/1826520147900],
        [0, 10008729576/15727000375, -45252884088/15727000375,
         67501517184/15727000375, -28901603736/15727000375],
        [0, -42869319978551/58745639878800, 26279956317109/8392234268400,
         -5270387298308/1223867497475, 103642379853661/58745639878800],
        [0, 10265285443/27377989200, -14253109853/9125996400,
         1166320544/570374775, -2474620297/3041998800],
        [0, -102497539/608399760, 451530737/608399760, -659633561/608399760,
         899768737/1825199280],
        [0, 39325/31704, -96525/21136, 53625/10568, -7150/3963],
        [0, 25525/4848, -25525/1616, 25525/1616, -25525/4848]])
    Pp1 = P1 * np.arange(2, 7)    # derivative of P1
    Bi = np.array([
        20342293/227212000, 0, 159248338847/434024589600, 33225336/155712875,
        -27213980937/193879999600, 12057023/271069200, -19822/1129455,
        -3575/63408, 0])

    # second higher order interpolant of fine where the velocity intepolant is
    # not the derivative of the displacement interpolant, needs two extra
    # evaluations. I modified the displacement interpolant to slightly increase
    # accuracy, making use of the extra evaluations.
    C_extra2 = (1/2, 1/2)
    A_extra2 = np.array([
        [43312501780291/1275109236086400, 544043/4374864,
         -39526111133/929968205760, -72268815551/18945421466400,
         15249672887173/919058778293760, 0, -936943/260282880, 0, 0],
        [78787/1651200, 0, 10240217/116820480, -5733/94300, 65097021/939312640,
         -53/1536, 1/64, 0, 0]])
    Ap_extra2 = np.array([
        [-10074474119/31414368960, 2, -2024048255/1461229504,
         262779483/503599720, -27782604453665/107223524134272, 0,
         -16606877/292226688, 0, 0],
        [70201889/791750400, 0, 86794851169/168046260480, 26319519/45216850,
         -895699605317/1351201232640, 867730151/2728776960, -192743/1624272,
         -429/1918, 0]])
    P2 = np.array([
        [1/2, -2121/1720, 4213/2580, -3783/3440, 1261/4300],
        [0, 0, 0, 0, 0],
        [0, 388869/121688, -648115/91266, 7388511/1216880, -1685099/912660],
        [0, 6552/4715, -29484/4715, 39312/4715, -78624/23575],
        [0, -1142053/2201514, 67381127/17612112, -170165897/29353520,
         53676491/22015140],
        [0, 0, -53/48, 159/80, -53/60],
        [0, -1/6, 1, -3/2, 2/3],
        [0, 0, 0, 0, 0],
        [0, -8/3, 8, -8, 8/3]])
    Pp2 = np.array([
        [1, -6478933/1649480, 4322879/618555, -18893117/3298960, 3783/2150],
        [0, 0, 0, 0, 0],
        [0, 12073105993/1050289128, -4230283954/131286141,
         67778480393/2100578256, -1685099/152110],
        [0, 68501376/4521685, -212403168/4521685, 238152312/4521685,
         -471744/23575],
        [0, -10466915745/703750642, 177078743809/4222503852,
         -238162590567/5630005136, 53676491/3669190],
        [0, 28284245/4263714, -150799787/8527428, 565239223/34109712, -53/10],
        [0, -294260/101517, 893071/101517, -1004879/101517, 4],
        [0, -3432/959, 6864/959, -3432/959, 0],
        [0, -8, 32, -40, 16]])

    # more accurate interpolant, not higher order, one extra evaluation
    C_extra3 = (7/13, )
    A_extra3 = np.array([
        [0.0514642952839635, 0, 0.103871371189972, -0.0689278735806428,
         0.0829881092428669, -0.0427358611434873, 0.0183103732085115]])
    Ap_extra3 = np.array([
        [0.0990632691053544, 0, 0.421051250650231, -0.0409728359730535,
         0.114675787267570, -0.0729606750492478, 0.0176047424606845]])
    P3 = np.array([
        [0.5, -1.17031483236872, 1.34773382152111, -0.621021772676567,
         -0.0597823106655287, 0.0967378073680061],
        [0, 0, 0, 0, 0, 0],
        [0, 2.65626120756873, -4.91705525506241, 2.75486717403821,
         0.391732827495055, -0.566243630720938],
        [0, 1.36554439688042, -4.36630995367658, 2.82125202693705,
         2.13320898315464, -1.81473468977485],
        [0, -0.514916544259984, 2.22831627015447, -1.0275974440311,
         -2.32367769527755, 1.58599961339654],
        [0, -0.00551650449428157, -0.398188141421234, -0.0973365487706088,
         1.19047020644871, -0.689429011762588],
        [0, -0.1230728332372, 0.484182482444333, -0.214110447909797,
         -0.532035218564604, 0.385036017267268],
        [0, -2.20798489008896, 5.62132077604031, -3.61605298758719,
         -0.799916792590728, 1.00263389422656]])
    Pp3 = P3 * np.arange(2, 8)    # derivative of P3

    def __init__(self, fun, t0, y0, t_bound,
                 sc_params=None, interpolant=0, **extraneous):
        super().__init__(fun, t0, y0, t_bound, **extraneous)
        # custom initialization to create extended storage for dense output
        if interpolant not in range(4):
            raise ValueError(
                "interpolant should be one of: 0, 1, 2, 3")
        self.interpolant = interpolant
        if self.interpolant == 3:
            self.K_extended = np.zeros((self.n_stages + 2,
                                        self.n), dtype=self.y.dtype)
            self.K = self.K_extended[:self.n_stages+1]
        elif self.interpolant != 0:
            self.K_extended = np.zeros((self.n_stages + 3,
                                        self.n), dtype=self.y.dtype)
            self.K = self.K_extended[:self.n_stages+1]

    def _dense_output_impl(self):
        if self.interpolant == 0:
            return QuinticHermiteDenseOutput(
                self.t_old, self.t, self.y_old, self.y, self.f_old, self.f)

        h = self.h_previous
        K = self.K_extended
        if self.interpolant == 1:
            C_extra = self.C_extra1
            A_extra, Ap_extra = self.A_extra1, self.Ap_extra1
            P, Pp = self.P1, self.Pp1
        elif self.interpolant == 2:
            C_extra = self.C_extra2
            A_extra, Ap_extra = self.A_extra2, self.Ap_extra2
            P, Pp = self.P2, self.Pp2
        else:
            C_extra = self.C_extra3
            A_extra, Ap_extra = self.A_extra3, self.Ap_extra3
            P, Pp = self.P3, self.Pp3
        for s, (a, ap, c) in enumerate(zip(A_extra, Ap_extra, C_extra),
                                       start=self.n_stages+1):
            dt = c * h
            du = (K[:s, :].T @ a[:s]) * h**2 + dt * self.y_old[self.n:]
            dv = (K[:s, :].T @ ap[:s]) * h
            dy = np.concatenate((du, dv))
            K[s] = self.fun(self.t_old + dt, self.y_old + dy)
        Q = K.T @ P
        Qp = K.T @ Pp
        if self.interpolant == 1:
            # replace position at end of step (not the velocity)
            # (The derivative function is not evaluated at this updated point)
            du = (K.T @ self.Bi) * h**2 + h * self.y_old[self.n:]
            self.y[:self.n] = self.y_old[:self.n] + du
        return HornerDenseOutputNystrom(self.t_old, self.t, self.y_old, Q, Qp)
