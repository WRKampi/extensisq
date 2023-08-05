import numpy as np
from extensisq.common import (
    RungeKuttaNystrom, QuinticHermiteDenseOutput, HornerDenseOutputNyquist)


class Mu5Nmb(RungeKuttaNystrom):
    """Explicit Runge-Kutta Nystrom (general) method by Murua [1]_ of order 5.
    This method is applicable to second order initial value problems only. The
    idea is to repeat the postion across several stage (but vary velocity).
    This allows for efficient integration of multibody equations.

    The second order problem should be recast in first order form as
    u = [x, v], du = [v, a], with x, v, a derivatives like, position,
    velocity, acceleration. The derivative function du = f(t, u) should
    calculate only a and pass through v. (The order in u and du matters.)

    This method can use a free interpolant or a better one.

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
    sc_params : tuple of size 4, "standard", "G", "H" or "W", optional
        Parameters for the stepsize controller (k*b1, k*b2, a2, g). The step
        size controller is, with k the exponent of the standard controller,
        _n for new and _o for old:
            h_n = h * g**(k*b1 + k*b2) * (h/h_o)**-a2
                * (err/tol)**-b1 * (err_o/tol_o)**-b2
        Predefined parameters are:
            Gustafsson "G" (0.7, -0.4, 0, 0.9),  Watts "W" (2, -1, -1, 0.8),
            Soederlind "S" (0.6, -0.2, 0, 0.9),  and "standard" (1, 0, 0, 0.9).
        The default for this method is "G".
    interpolant : "free" or "better"
        Select the interpolant for dense output.
        Option "free" is for the 4th order accurate interpolant that is the 5th
        order hermite polynomial that satisfies C2 continuity at the solution
        points. This free interpolant satisfies the RKN order conditions up to
        order 4 and requires no extra function evaluations.
        Option "better" is for the 5th order accurate  interpolant that needs 1
        extra function evaluation.
        Default: "free".

    References
    ----------
    .. [1] A. Murua, "Runge-Kutta-Nyström methods for general second order ODEs
           with application to multi-body systems", Applied Numerical
           Mathematics, Vol. 28, 1998, pp. 387-399.
           https://doi.org/10.1016/S0168-9274(98)00055-5
    """
    n_stages = 9
    order = 5
    error_estimator_order = 4
    sc_params = "G"

    C = np.array([0, 771/3847, 771/3847, 3051/6788, 4331/6516, 4331/6516,
                  10463/11400, 10463/11400, 1])
    Ap = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [771/3847, 0, 0, 0, 0, 0, 0, 0, 0],
        [771/7694, 771/7694, 0, 0, 0, 0, 0, 0, 0],
        [-264272222/4845505509, -9458865980/12714902623,
         17133165419/13729279360, 0, 0, 0, 0, 0, 0],
        [1943604853/18116134489, -2470367896/7636570485, 1733951147/3918733571,
         4613437932/10523350595, 0, 0, 0, 0, 0],
        [369952551/2046485744, 281630106828/143708239525,
         -9868262031/5606899429, 208606720/5597531799, 792516107/3209667255,
         0, 0, 0, 0],
        [-2089737154/15083636501, -39924138556/8175090533,
         72922890855/14010113917, 9484193351/15493195043,
         -17895265139/12412283353, 278232/177835, 0, 0, 0],
        [-1762013041/13188190032, -22636373880/4795132451,
         30527401913/6048941340, 11564353310/19632283007,
         -50677425731/36595197965, 12408/8167, 10722067/5782709432, 0, 0],
        [8034174097/12261534992, 72032427203/6782716235,
         -90566218637/8185393121, 18770105843/41171085325,
         28010344030/6199889941, -21917292279/4540377286,
         -236637914115/8183370127, 71217630373/2409299224, 0]])
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [594441/29598818, 0, 0, 0, 0, 0, 0, 0, 0],
        [594441/29598818, 0, 0, 0, 0, 0, 0, 0, 0],
        [-311625081/28869248936, 128/8219, 1015645542/10554116159,
         0, 0, 0, 0, 0, 0],
        [1852480471/26299626569, -247/14069, 648800762/5897141541,
         519849979/8963946221, 0, 0, 0, 0, 0],
        [1852480471/26299626569, -247/14069, 648800762/5897141541,
         519849979/8963946221, 0, 0, 0, 0, 0],
        [229929851/7158517178, 113395809/8665398238, 4865737279/19748497543,
         340133672/10137556453, 738/11587, 509108839/15737542787, 0, 0, 0],
        [229929851/7158517178, 113395809/8665398238, 4865737279/19748497543,
         340133672/10137556453, 738/11587, 509108839/15737542787, 0, 0, 0],
        [164505448/2653157365, 0, 9357192/40412735, 736403089/7677655029,
         960089/17896194, 482653907/11393392643, -47281957/150822000,
         6715245221/20471724521, 0]])
    Bp = np.array(
        [164505448/2653157365, 0, 3042/10505, 1586146904/9104113535,
         4394/27465, 2081836558/16479128289, -50461/13230,
         13928550541/3490062596, 91464477/8242174145])
    B = np.array(
        [164505448/2653157365, 0, 9357192/40412735, 736403089/7677655029,
         960089/17896194, 482653907/11393392643, -47281957/150822000,
         6715245221/20471724521, 0])
    Ep = np.array(
        [53757362/127184461, 0, -138687950/204047369, 161961633/188152853,
         36242723/103243418, 1/2, 1147554103/9981952, -2395015001/20532034,
         1, 23/100])
    Ep[:-1] -= Bp
    E = np.array(
        [53757362/127184461, 0, -426604134200/784970228543,
         605250622521/1277181566164, 79190349755/672734111688, 2185/13032,
         1075258194511/113794252800, -2244129055937/234065187600, 0, 0])
    E[:-1] -= B

    # better interpolant
    C_extra = 1/2
    A_extra = np.array([
        6272277221/169802071360, 0, 45601101/646603760,
        105407530693976029/5083508586660343092, 261443/143169552,
        4331771911506493999/3004050864175132445232, 264970711/603288000,
        -1050721229560409919849/2286323200843459728512, -91464477/52749914528,
        1/64])
    Ap_extra = np.array([
        4914093243/84901035680, 0, 9036261/29391080,
        1755813615360948893/16945028622201143640, 7300631/238615920,
        18144300078070688533/751012716043783111308, 86944303/80438400,
        -1292891362501846999547/1143161600421729864256,
        -640251339/131874786320, 1/32])
    P_better = np.array([
        [1/2, -4924143773/3183788838, 2398376727/1061262946,
         -1666124677/1061262946, 3332249354/7959472095],
        [0, 0, 0, 0, 0],
        [0, 18714384/8082547, -39774150/8082547, 161466318/40412735,
         -12168/10505],
        [0, 7364030890/7677655029, -978508507380924517/423625715555028591,
         4540205538386050898/2118128577775142955, -6344587616/9104113535],
        [0, 4800445/8948097, -9572329/5965398, 26273923/14913495,
         -17576/27465],
        [0, 4826539070/11393392643,
         -237901641408829340815/187753179010945777827,
         87064975276817078628/62584393003648592609, -8327346232/16479128289],
        [0, -47281957/15082200, 79677919/3351600, -910165057/25137000,
         100922/6615],
        [0, 67152452210/20471724521,
         -1777256640792585385045/71447600026358116516,
         2706892803882276765045/71447600026358116516, -13928550541/872515649],
        [0, 0, -91464477/1648434829, 823180293/8242174145,
         -365857908/8242174145],
        [0, -1/6, 1, -3/2, 2/3],
        [0, -8/3, 8, -8, 8/3]])
    Pp_better = P_better * np.arange(2, 7)    # derivative of P_low

    def __init__(self, fun, t0, y0, t_bound,
                 interpolant='better', **extraneous):
        super().__init__(fun, t0, y0, t_bound, **extraneous)
        # custom initialization to create extended storage for dense output
        if interpolant not in ('better', 'free'):
            raise ValueError(
                "interpolant should be one of: 'free', 'better'")
        self.interpolant = interpolant
        if interpolant == 'better':
            self.K_extended = np.zeros((
                self.n_stages + 2, self.n), dtype=self.y.dtype)
            self.K = self.K_extended[:self.n_stages+1]

    def _dense_output_impl(self):
        if self.interpolant == 'free':
            return QuinticHermiteDenseOutput(
                self.t_old, self.t, self.y_old, self.y, self.f_old, self.f)
        # else:
        h = self.h_previous
        K = self.K_extended

        # extra stage
        s = self.n_stages + 1
        dt = self.C_extra * h
        du = (self.K.T @ self.A_extra) * h**2 + dt * self.y_old[self.n:]
        dv = (self.K.T @ self.Ap_extra) * h
        dy = np.concatenate((du, dv))
        K[s] = self.fun(self.t_old + dt, self.y_old + dy)

        Q = K.T @ self.P_better
        Qp = K.T @ self.Pp_better
        return HornerDenseOutputNyquist(self.t_old, self.t, self.y_old, Q, Qp)
