import numpy as np
from extensisq.common import RungeKutta, RungeKuttaNystrom, QuinticHermiteDenseOutput, HornerDenseOutputNystrom, calculate_scale, norm, MAX_FACTOR, NFS
from extensisq import Fi4N
from math import sqrt, sin, cos
# from extensisq.dormand_nystrom import DPM6NN


class MR4N(RungeKuttaNystrom):
    n_stages = 6
    order = 4
    error_estimator_order = 3
    sc_params = "G"

    C = np.array([ 0, 1/77, 1/3, 2/3, 13/15, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/11858, 0, 0, 0, 0, 0],
        [-7189/17118, 4070/8559, 0, 0, 0, 0],
        [4007/2403, -589655/355644, 25217/118548, 0, 0, 0],
        [-4477057/843750, 13331783894/2357015625, -281996/5203125,
         563992/7078125, 0, 0],
        [17265/2002, -1886451746/212088107, 22401/31339, 2964/127897,
         178125/5428423, 0]])
    Ap = np.array([
        [0, 0, 0, 0, 0, 0],
        [              1/77,                         0,                  0,                 0,               0, 0],
        [       -34184/8559,                37037/8559,                  0,                 0,               0, 0],
        [       122041/4806,            -2348654/88911,        33371/19758,                 0,               0, 0],
        [-500331884/8015625, 2929175570321/44783296875, -93391948/32953125, 31942456/44828125,               0, 0],
        [     1176953/11011,      -3383067836/30298301,       188163/31339,     -92916/127897, 2671875/5428423, 0]])
    B = np.array(
        [-341/780, 386683451/661053840, 2853/11840, 267/3020, 9375/410176, 0])
    Bp = np.array([-341/780, 29774625727/50240091840, 8559/23680, 801/3020,
                   140625/820352, 847/18240])
    E = np.array([
        [                      105497/64740],
        [-4570712363895139/2903421842256240],
        [          52818381449/156007782720],
        [                      74693/842580],
        [                 4082175/233390144],
        [                                 0],
        [                                 0],
        ]).squeeze()
    Ep = np.array([
        [                       20299/21580],
        [-1440439768795731/1690869072700480],
        [     10261328923479/24485268734080],
        [        947960620681/3122698968620],
        [   124768847515625/848249120630912],
        [                      10587/255040],
        [                                 0],
        ]).squeeze()
    E[:-1] -= B
    Ep[:-1] -= Bp


class N5stage(RungeKuttaNystrom):
    n_stages = 5
    order = 4
    error_estimator_order = 3
    sc_params = "G"

    C = np.array([0, 1/4, 3/8, 12/13, 1])
    A = np.array([
        [       0,         0,    0,    0, 0],
        [    1/32,         0,    0,    0, 0],
        [   9/256,     9/256,    0,    0, 0],
        [413/4394, -231/4394, 5/13,    0, 0],
        [  10/117,    -8/195,  4/9, 1/90, 0]])
    Ap = np.array([
        [        0,          0,         0,         0, 0],
        [      1/4,          0,         0,         0, 0],
        [     3/32,       9/32,         0,         0, 0],
        [1932/2197, -7200/2197, 7296/2197,         0, 0],
        [  439/216,         -8,  3680/513, -845/4104, 0]])
    B = np.array([253/2160, 0, 4352/12825, 2197/41040, -1/100])
    Bp = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5])
    E = np.array([53/495, 0, 896/2475, 0, 17/550, 0])
    E[:-1] -= B
    Ep = np.array([   29719/258120, 0, 1687552/3065175, 2599051/4904280, -1169/5975, 0])
    Ep[:-1] -= Bp
    # print(E.sum())
    # print(Ep.sum())
    # print(B.sum())
    # print(Bp.sum())
    # print(C - Ap.sum(axis=1))
    # print(C**2/2 - A.sum(axis=1))


class Fi4N_alt(Fi4N):
    A = np.array([
            [     0,       0,       0, 0, 0],
            [  2/81,       0,       0, 0, 0],
            [  1/36,    1/36,       0, 0, 0],
            [63/640, -27/320, 171/640, 0, 0],
            [11/100, -33/100,   18/25, 0, 0]])
    Ap = np.array([
            [     0,        0,      0,     0, 0],
            [   2/9,        0,      0,     0, 0],
            [  1/12,      1/4,      0,     0, 0],
            [69/128, -243/128, 135/64,     0, 0],
            [-17/12,     27/4,  -27/5, 16/15, 0]])
    B = np.array([
            [19/180],
            [     0],
            [63/200],
            [16/225],
            [ 1/120]]).squeeze()
    Bp = np.array([
            [  1/9],
            [    0],
            [ 9/20],
            [16/45],
            [ 1/12]]).squeeze()
    E = np.array([
            [343/15660],
            [        0],
            [-219/5800],
            [256/19575],
            [    1/360],
            [0]]).squeeze()
    Ep = np.array([
            [-25/2448],
            [       0],
            [  15/544],
            [  -5/153],
            [ 25/1632],
            [0]]).squeeze()


class Pr4N(RungeKuttaNystrom):
    n_stages = 5
    order = 4
    error_estimator_order = 3
    sc_params = "G"

    C = np.array([0, 4/15, 2/5, 24/25, 1])
    A = np.array([
        [        0,          0,         0, 0, 0],
        [    8/225,          0,         0, 0, 0],
        [    7/200,      9/200,         0, 0, 0],
        [1533/6250, -2349/6250, 1848/3125, 0, 0],
        [     5/16,      -9/16,       3/4, 0, 0]])
    Ap = np.array([
        [      0,         0,        0,         0, 0],
        [   4/15,         0,        0,         0, 0],
        [   1/10,      3/10,        0,         0, 0],
        [528/625, -1944/625, 2016/625,         0, 0],
        [337/272,  -645/136, 2175/476, -125/1904, 0]])
    B = np.array([1/8, 0, 25/72, 0, 1/36])
    Bp = np.array([73/576, 0, 575/1008, 3125/4032, -17/36])
    E = np.array([73/684, 0, 775/2052, 0, 8/513, 0])
    E[:-1] -= B
    Ep = np.array([2357/19584, 0, 6725/11424, 90625/137088, -151/408, 0])
    Ep[:-1] -= Bp
    # scale
    E *= 0.15
    Ep *= 0.15
    # print(C - Ap.sum(axis=1))
    # print(C**2/2 - A.sum(axis=1))
    # print(B.sum())
    # print(Bp.sum())
    # print(E.sum())
    # print(Ep.sum())
    # stop


class Me4N(RungeKuttaNystrom):
    n_stages = 5
    order = 4
    error_estimator_order = 3
    sc_params = "G"

    C = np.array([0, 1/3, 1/3, 1/2, 1])
    A = np.array([
        [0, 0, 0, 0, 0],
        [1/18, 0, 0, 0, 0],
        [1/18, 0, 0, 0, 0],
        [1/16, 1/40, 3/80, 0, 0],
        [1/6, 0, 0, 1/3, 0]])
    Ap = np.array([
        [0, 0, 0, 0, 0],
        [1/3, 0, 0, 0, 0],
        [1/6, 1/6, 0, 0, 0],
        [1/8, 0, 3/8, 0, 0],
        [1/2, 0, -3/2, 2, 0]])
    B = np.array([1/6, 0, 0, 1/3, 0])
    Bp = np.array([1/6, 0, 0, 2/3, 1/6])
    E = np.array([1669/4860, 0, -299/480, 7877/9720, -1201/38880, 0])
    E[:-1] -= B
    Ep = np.array([1448/4515, 0, -4173/6020, 5792/4515, 1619/18060, 0])
    Ep[:-1] -= Bp


class Me4Nalt(Me4N):
    """different E, Ep: B, Bp, C, and Cp optimized with weighted norm.
    This does not seem to improve the method."""
    E = np.array([  12243/19540, 0, -25857/15632, 94907/58620, -21449/234480, 0])
    E[:-1] -= np.array([1/6, 0, 0, 1/3, 0])
    Ep = np.array([12808/53315, 0, -70599/213260, 51232/53315, 27699/213260, 0])
    Ep[:-1] -= np.array([1/6, 0, 0, 2/3, 1/6])


class BS5N(RungeKuttaNystrom):
    n_stages = 7
    order = 5
    error_estimator_order = 4
    sc_params = "G"

    C = np.array([0, 1/6, 2/9, 3/7, 2/3, 3/4, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/72, 0, 0, 0, 0, 0, 0],
        [10/729, 8/729, 0, 0, 0, 0, 0],
        [603/19208, 27/4802, 1053/19208, 0, 0, 0, 0],
        [367/6237, -116/2079, 171/1001, 560/11583, 0, 0, 0],
        [1107/22528, -189/5632, 63099/292864, 8379/366080, 1539/56320, 0, 0],
        [14681/147576, 17191/159874, -6453/755768, 18130/72501, -9351/159874,
         26368/239811, 0]])
    Ap = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/6, 0, 0, 0, 0, 0, 0],
        [2/27, 4/27, 0, 0, 0, 0, 0],
        [183/1372, -162/343, 1053/1372, 0, 0, 0, 0],
        [68/297, -4/11, 42/143, 1960/3861, 0, 0, 0],
        [597/22528, 81/352, 63099/585728, 58653/366080, 4617/20480, 0, 0],
        [174197/959244, -30942/79937, 8152137/19744439, 666106/1039181,
         -29421/29068, 482048/414219, 0]])
    B = np.array([79/1080, 0, 2187/9880, 2401/21060, 0, 704/7695, 0])
    Bp = np.array([587/8064, 0, 4440339/15491840, 24353/124800,
                  387/44800, 2152/5985, 7267/94080])
    E = np.array([6059/80640, 0, 951021/4426240, 3773/31200,
                  -309/89600, 443/4788, 0, 0])
    E[:-1] -= B
    Ep_pre = np.array([
        -3/1280, 0, 6561/632320, -343/20800, 243/12800, -1/95])
    Ep = np.array([2479/34992, 0, 123/416, 612941/3411720, 43/1440, 2272/6561,
                   79937/1113912, 3293/556956])
    Ep[:-1] -= Bp
    B_scale_pre = np.array([19/360, 0, 81/260, -343/11700, 33/200, 0])
    Bp_scale_pre = np.array([19/48, 0, -189/416, 343/780, 99/160, 0])

    # for low cost, higher order interpolant
    C_extra = 1/2
    A_extra = np.array([
        158023/3870720, 0, 32665761/495738880, 571781/21565440, -387/286720,
        -2249/215460, -7267/602112, 1/64])
    Ap_extra = np.array([
        455/6144, 0, 10256301/35409920, 2307361/17971200, -387/102400, 73/5130,
        -7267/215040, 1/32])
    P_low = np.array([
        [1/2, -155/108, 16441/8064, -56689/40320, 757/2016],
        [0, 0, 0, 0, 0],
        [0, 2187/988, -14727987/3098368, 60538347/15491840, -4440339/3872960],
        [0, 2401/2106, -603337/224640, 2740913/1123200, -24353/31200],
        [0, 0, -387/8960, 3483/44800, -387/11200],
        [0, 1408/1539, -11384/3591, 13592/3591, -8608/5985],
        [0, 0, -7267/18816, 21801/31360, -7267/23520],
        [0, -1/6, 1, -3/2, 2/3],
        [0, -8/3, 8, -8, 8/3]])
    Pp_low = P_low * np.arange(2, 7)    # derivative of P_low

    def __init__(self, fun, t0, y0, t_bound,
                 sc_params=(1, 0, 0, 0.85), interpolant='low', **extraneous):
        super().__init__(fun, t0, y0, t_bound, **extraneous)
        # custom initialization to create extended storage for dense output
        if interpolant not in ('low', 'free'):
            raise ValueError(
                "interpolant should be one of: 'low', 'free'")
        self.interpolant = interpolant
        if self.interpolant == 'low':
            self.K_extended = np.zeros((self.n_stages + 2,
                                        self.n), dtype=self.y.dtype)
            self.K = self.K_extended[:self.n_stages+1]

    def _step_impl(self):
        # redefine for two error estimates

        # mostly follows the scipy implementation of RungeKutta
        t = self.t
        y = self.y

        h_abs, min_step = self._reassess_stepsize(t, y)

        # loop until step accepted
        step_accepted = False
        step_rejected = False
        while not step_accepted:

            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            # calculate stages, except last two
            self.K[0] = self.f
            for i in range(1, self.n_stages - 1):
                self._rk_stage(h, i)

            # calculate pre_error_norm
            error_norm_pre = self._estimate_error_norm_pre(y, h)

            # reject step if pre_error too large
            if error_norm_pre > 1:
                step_rejected = True
                h_abs *= max(
                    self.min_factor,
                    self.safety * error_norm_pre ** self.error_exponent)

                NFS[()] += 1
                if self.nfev_stiff_detect:
                    self.jflstp += 1                  # for stiffness detection
                continue

            # calculate next stage
            self._rk_stage(h, self.n_stages - 1)

            # calculate error_norm and solution
            y_new, error_norm = self._comp_sol_err(y, h)

            # and evaluate
            if error_norm < 1:
                step_accepted = True

                if error_norm < self.tiny_err:
                    factor = self.max_factor
                    self.standard_sc = True

                elif self.standard_sc:
                    factor = self.safety * error_norm ** self.error_exponent
                    self.standard_sc = False

                else:
                    # use second order SC controller
                    h_ratio = h / self.h_previous
                    factor = self.safety_sc * (
                        error_norm ** self.minbeta1 *
                        self.error_norm_old ** self.minbeta2 *
                        h_ratio ** self.minalpha)
                    factor = min(self.max_factor, max(self.min_factor, factor))

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                if factor < MAX_FACTOR:
                    # reduce max_factor when on scale.
                    self.max_factor = MAX_FACTOR

            else:
                if np.isnan(error_norm) or np.isinf(error_norm):
                    return False, "Overflow or underflow encountered."

                step_rejected = True
                h_abs *= max(self.min_factor,
                             self.safety * error_norm ** self.error_exponent)

                NFS[()] += 1
                self.jflstp += 1                      # for stiffness detection

        # store for next step and interpolation
        self.h_previous = h
        self.y_old = y
        self.h_abs = h_abs
        self.f_old = self.f
        self.f = self.K[self.n_stages].copy()
        self.error_norm_old = error_norm

        # output
        self.t = t_new
        self.y = y_new

        # stiffness detection
        self._diagnose_stiffness()

        return True, None

    def _estimate_error_norm_pre(self, y, h):
        # first error estimate
        # y_new is not available yet for scale, so use y_pre instead
        du = (self.K[:6].T @ self.B_scale_pre) * h**2 \
            + h*self.y[self.n:]
        dv = (self.K[:6].T @ self.Bp_scale_pre) * h
        dy = np.concatenate((du, dv))
        y_pre = y + dy
        scale = calculate_scale(self.atol, self.rtol, y, y_pre)
        # error
        eu = (self.K[:6, :].T @ self.E[:6]) * h**2
        ev = (self.K[:6, :].T @ self.Ep_pre) * h
        err = np.concatenate((eu, ev))
        return norm(err / scale)

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

        Q = K.T @ self.P_low
        Qp = K.T @ self.Pp_low
        return HornerDenseOutputNystrom(self.t_old, self.t, self.y_old, Q, Qp)


class SS4(RungeKutta):
    n_stages = 4
    order = 4
    error_estimator_order = 3
    sc_params = "G"
    
    C = np.array([0, 2/5, 3/5, 1])

    # runge kutta coefficient matrix
    A = np.array([
        [0, 0, 0, 0],
        [2/5, 0, 0, 0],
        [-3/20, 3/4, 0, 0],
        [19/44, -15/44, 10/11, 0]])

    # output coefficients (weights)
    B = np.array([11/72, 25/72, 25/72, 11/72])

    # error coefficients (weights Bh - B)
    E = np.array([1251515/8970912, 3710105/8970912, 2519695/8970912, 61105/8970912, 119041/747576])        # B_hat
    E[:-1] -= B
    # print(E.sum())


class SS3(RungeKutta):
    n_stages = 3
    order = 3
    error_estimator_order = 2
    sc_params = "G"
    
    C = np.array([0, 1/2, 1])

    # runge kutta coefficient matrix
    A = np.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [-1, 2, 0]])

    # output coefficients (weights)
    B = np.array([1/6, 2/3, 1/6])

    # error coefficients (weights Bh - B)
    r = np.sqrt(82)
    E = np.array([11/36 - r/72, r/36 + 7/18, -1/36 + r/144, 1/3 - r/48])        # B_hat
    E[:-1] -= B
    # print(E.sum())


class SS3N(RungeKuttaNystrom):
    n_stages = 3
    order = 3
    error_estimator_order = 2
    sc_params = "G"
    
    C = np.array([0, 1/2, 1])

    # runge kutta coefficient matrix
    Ap = np.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [-1, 2, 0]])
    A = np.array([
        [0, 0, 0],
        [1/8, 0, 0],
        [1/8, 3/8, 0]])

    # output coefficients (weights)
    Bp = np.array([1/6, 2/3, 1/6])
    B = np.array([5/24, 1/4, 1/24])

    # error coefficients (weights Bh - B)
    r = np.sqrt(82)
    Ep = np.array([11/36 - r/72, r/36 + 7/18, -1/36 + r/144, 1/3 - r/48])        # B_hat
    Ep[:-1] -= Bp
    # print(E.sum())
    # E = np.array([7/30, 1/5, 1/60, 1/20])        # B_hat
    E = np.array([11/64, 5/16, 13/192, -5/96])        # B_hat
    E[:-1] -= B


class BS3NnoFSAL(RungeKuttaNystrom):
    n_stages = 3
    order = 3
    error_estimator_order = 2
    sc_params = "G"

    C = np.array([0, 1/2, 1])

    # runge kutta coefficient matrix
    Ap = np.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [-1, 2, 0]])
    A = np.array([
        [0, 0, 0],
        [1/8, 0, 0],
        [1/8, 3/8, 0]])

    # output coefficients (weights)
    Bp = np.array([1/6, 2/3, 1/6])
    B = np.array([5/24, 1/4, 1/24])

    # error coefficients (weights Bh - B)
    Ep = np.array([1/4, 1/4, 1/2, 0])        # Bp_hat
    Ep[:-1] -= Bp
    # print(Ep.sum())
    E = np.array([119/384, 29/384, 11/96, 0])        # B_hat
    E[:-1] -= B
    # print(E.sum())


class SS4N(RungeKuttaNystrom):
    n_stages = 4
    order = 4
    error_estimator_order = 3
    sc_params = "G"

    C = np.array([0, 2/5, 3/5, 1])

    A = np.array([
        [0, 0, 0, 0],
        [2/25, 0, 0, 0],
        [3/100, 3/20, 0, 0],
        [1/4, 3/44, 2/11, 0]])

    # runge kutta coefficient matrix
    Ap = np.array([
        [0, 0, 0, 0],
        [2/5, 0, 0, 0],
        [-3/20, 3/4, 0, 0],
        [19/44, -15/44, 10/11, 0]])

    B = np.array([11/72, 5/24, 5/36, 0])

    # output coefficients (weights)
    Bp = np.array([11/72, 25/72, 25/72, 11/72])

    E = np.array([1251515/8970912, 742021/2990304, 503939/4485456, 0, 0])
    E[:-1] -= B

    # error coefficients (weights Bh - B)
    Ep = np.array([1251515/8970912, 3710105/8970912, 2519695/8970912,
                   61105/8970912, 119041/747576])
    Ep[:-1] -= Bp
    # print(E.sum())


class CMR7(RungeKutta):
    order = 7
    error_estimator_order = 5
    n_stages = 9       # effective nr
    n_extra_stages = 1
    
    tanang = 30
    stbrad = 4.9
    
    # time step fractions
    C = np.array([0, 1/18, 1/12, 1/8, 141151442/351495483, 3/5, 7/9, 91/100, 1])
    
    # coefficient matrix
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/18, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/48, 1/16, 0, 0, 0, 0, 0, 0, 0],
        [1/32, 0, 3/32, 0, 0, 0, 0, 0, 0],
        [869438775/1009532624, 0, -950731448/286874265,
         1264053625/442836138, 0, 0, 0, 0, 0],
        [-5275567997/1151793423, 0, 1471930333/82266519, -728519255/53110846,
         246123667/244887087, 0, 0, 0, 0],
        [-645025827/336785728, 0, 2434795058/338513483, -612892207/118428380, 291558131/1044507815,
         60119121/151628548, 0, 0, 0],
        [8295229381/419686309, 0, -1182714313/16094409,
         3951798187/70936984, -119548150/350906669,
         -235440881/147135711, 332228193/384969575, 0, 0],
        [-1280686197/53447645, 0, 3079014121/34582924, -1675387903/25062046,
         79857679/86519700, 832647762/368676149,
         -116509745/197679253, 642808259/3451600411, 0],
        ])
    # print(A.sum(axis=1) - C)
    #~ print(A[-2,:])
    
    # coefficients for propagating method
    B = np.array([
        52152085/1622722103, 0, 0, 176651997/822132167, 221088691/735240925,
        95965011/875274922, 469539417/2151541436, 45088067/537536234,
        37565173/926723012
        ])
    # print(B.sum() - 1)

    # coefficients for error estimation
    E = np.array([
        29785215/1630176049, 0, 0, 1/4, 80587936/349510325, 547273798/2676284737,
        111968947/734737436, 74972764/722724891, 37565173/926723012, 0
        ])
    E[:-1] -= B
    # print(E)
    # print(E.sum())


class semiRKN(RungeKuttaNystrom):
    def __init__(self, *args, jac, **kwargs):
        self.jac = jac
        super().__init__(*args, **kwargs)
    
    def _rk_stage(self, h, i):
        """compute a single RK stage"""
        dt = self.C[i] * h
        du = (self.K[:i, :].T @ self.A[i, :i])*h**2 + dt*self.y[self.n:]
        
        if self.Ap[i, i] == 0.:
            dv = (self.K[:i, :].T @ self.Ap[i, :i])*h
            dy = np.concatenate((du, dv))
            self.K[i] = self.fun(self.t + dt, self.y + dy)
        else:
            dv = (self.K[:i, :].T @ self.ap[i, :i])*h
            dy = np.concatenate((du, dv))
            self.K[i] = self.fun(self.t + dt, self.y + dy)
            res = dv - h*(self.K[:i+1, :].T @ self.Ap[i, :i+1])
            j = 0
            while True:     
                if j and norm(res) < self.atol/h/100:   # for now
                    break
                jacv = np.eye(self.n) - h*self.Ap[i, i]*self.jac(
                    self.t + dt, self.y + dy)[self.n:, self.n:]
                dv -= np.linalg.solve(jacv, res)
                dy = np.concatenate((du, dv))
                self.K[i] = self.fun_single(self.t + dt, self.y + dy)[self.n:]
                res = dv - (self.K[:i+1, :].T @ self.Ap[i, :i+1])*h
                j += 1
            # print(j)


class DMP4NSalt(semiRKN):
    n_stages = 4
    order = 4
    error_estimator_order = 3
    sc_params = "G"
    
    C = np.array([0, 1/4, 7/10, 1])
    A = np.array([
        [     0,       0,      0, 0],
        [  1/32,       0,      0, 0],
        [7/1000, 119/500,      0, 0],
        [  1/14,    8/27, 25/189, 0]])
    Ap = np.array([
        [     0,     0,    0, 0],
        [   1/8,   1/8,    0, 0],
        [-1/100, 14/25, 3/20, 0],
        [   2/7,     0,  5/7, 0]])
    ap = np.array([     # for prediction of v
        [     0,     0,    0, 0],
        [   1/4,   0,    0, 0],
        [-7/25, 49/50, 0, 0],
        [0, 0, 0, 0]])
    B = np.array([1/14, 8/27, 25/189, 0])
    Bp = np.array([1/14, 32/81, 250/567, 5/54])
    E = np.array([31/1575, -203/8100, -67/22680, 1/120, 0])
    Ep = np.array([-23/672, 23/324, -575/9072, 23/864, 0])


class DMP4NS(semiRKN):
    n_stages = 4
    order = 4
    error_estimator_order = 3
    sc_params = "G"
    
    C = np.array([0, 1/4, 7/10, 1])
    A = np.array([
        [     0,       0,      0, 0],
        [  1/32,       0,      0, 0],
        [7/1000, 119/500,      0, 0],
        [  1/14,    8/27, 25/189, 0]])
    Ap = np.array([
        [     0,     0,    0, 0],
        [   1/8,   1/8,    0, 0],
        [1/100, 119/225,  29/180, 0],
        [1/14,   32/81, 250/567, 5/54]])
    ap = np.array([     # for prediction of v
        [     0,     0,    0, 0],
        [   1/4,   0,    0, 0],
        [-7/25, 49/50, 0, 0],
        [4/21, 4/27, 125/189, 0]])
    B = np.array([1/14, 8/27, 25/189, 0])
    Bp = np.array([1/14, 32/81, 250/567, 5/54])
    E = np.array([31/1575, -203/8100, -67/22680, 1/120, 0])
    Ep = np.array([-23/672, 23/324, -575/9072, 23/864, 0])


class Oz4NS(semiRKN):
    n_stages = 3
    order = 4
    error_estimator_order = 3
    sc_params = "G"
    
    C = np.array([0, 1/3, 5/6])
    A = np.array([
        [0, 0, 0],
        [1/18, 0, 0],
        [5/144, 5/16, 0]])
    Ap = np.array([
        [0, 0, 0],
        [1/6, 1/6, 0],
        [1/24, 5/8, 1/6]])
    ap = np.array([     # for prediction of v
        [0, 0, 0],
        [1/3, 0, 0],
        [-5/24, 25/24, 0]])
    B = np.array([1/10, 1/3, 1/15])
    Bp = np.array([1/10, 1/2, 2/5])
    E = np.array([1/20, -1/12, 1/30, 0])
    Ep = np.array([1/20, -1/8, 1/5, -1/8])


class MR6NS(semiRKN):
    n_stages = 6
    order = 6
    error_estimator_order = 4
    sc_params = "G"
    
    C = np.array([0, 1/77, 1/3, 2/3, 13/15, 1])

    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/11858, 0, 0, 0, 0, 0],
        [-7189/17118, 4070/8559, 0, 0, 0, 0],
        [4007/2403, -589655/355644, 25217/118548, 0, 0, 0],
        [-4477057/843750, 13331783894/2357015625, -281996/5203125,
         563992/7078125, 0, 0],
        [17265/2002, -1886451746/212088107, 22401/31339, 2964/127897,
         178125/5428423, 0]])
    Ap = np.array([
        [            0,                    0,             0,              0,                0, 0],
        [        1/154,                1/154,             0,              0,                0, 0],
        [   -7189/5706,            4235/2853,        69/634,              0,                0, 0],
        [    4007/1602,       -300685/118548,   25217/39516,         17/267,                0, 0],
        [-344389/56250, 1041123083/157134375, -70499/693750, 563992/1415625,             1/15, 0],
        [   17265/2002,    -49643467/5508782,   67203/62678,    8892/127897, 2671875/10856846, 0]])
    ap = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/77, 0, 0, 0, 0, 0],
        [-71/18, 77/18, 0, 0, 0, 0],
        [154/27, -5929/999, 299/333, 0, 0, 0],
        [-1552685888/635326875, 39130486738343/14198285002500,
         938973971/10447597500, 14860477918/31978119375, 0, 0],
        [0, 0, 0, 0, 0, 0]])
    B = np.array([-341/780, 386683451/661053840, 2853/11840, 267/3020,
                  9375/410176, 0])

    Bp = np.array([-341/780, 29774625727/50240091840, 8559/23680, 801/3020,
                   140625/820352, 847/18240])

    E = np.array([-95/39, 89332243/33052692, 317/3552, 623/5436, 54125/1845792,
                  0, 0])
    E[:-1] -= B

    Ep = np.array([-95/39, 362030669/132210768, 317/2368, 623/1812,
                   270625/1230528, 0, 0])
    Ep[:-1] -= Bp


if __name__ == '__main__':
    from scipy.integrate import solve_ivp, RK45, RK23, DOP853
    from extensisq import NFS, Me4, CFMR7osc, Mu5Nmb, Fi4N, Fi5N, SWAG, BS5, MR6NN
    import matplotlib.pyplot as plt
    from math import sin

    # method = Fi4N
    # print(method.A.sum(axis=1) - 0.5*method.C**2)
    # print(method.B.sum())
    # print(method.Bp.sum())

    print(Me4N.A.min())
    print(Me4N.Ap.min())
    print(Fi4N.A.min())
    print(Fi4N.Ap.min())
    print(SS4N.A.min())
    print(SS4N.Ap.min())
    # stop

    def fun(t, y):
        return [y[1], -y[0]]
    
    def jac(t, y):
        return np.array([[0, 1], [-1, 0]])

    res = solve_ivp(fun, (0, 2*np.pi), [0, 1], method=Mu5Nmb)
    print(res)

    def ref(t):
        x = np.cos(t)
        y = -np.sin(t)
        return np.stack((x, y))

    def fun2(t, y):
        return [y[1], -sin(y[0])]


    t_span = [0, 30]
    y0 = [1., 0.]

    # s1 = solve_ivp(fun, t_span, y0, atol=1e-3,
    #                method=DMP4NSalt, jac=jac)
    # refs = solve_ivp(fun, t_span, y0, atol=1e-3,
    #                method=Fi4N)

    # plt.plot(refs.t, refs.y.T)
    # plt.plot(s1.t, s1.y.T, ':')
    # plt.grid()
    # plt.show()
    # stop

    ref2 = solve_ivp(fun2, t_span, y0, 
                     method=CFMR7osc,
                     # method=Mu5Nmb,
                     atol=1e-12, rtol=1e-12, 
                     dense_output=True,
                     # interpolant="better"
                     )
    print(ref2.nfev)
    ref2 = ref2.sol
    t = np.linspace(*t_span, 1000)
    # plt.plot(t, ref2(t).T)
    # plt.show()

    # stop

    E2 = {'fun': lambda x, y: [y[1], (1-y[0]**2)*y[1] - y[0]],
          'y0': [2., 0.], 't_span': (0, 20.)}
    refE2 = solve_ivp(**E2, method=CFMR7osc,
                      atol=1e-12, rtol=1e-12, dense_output=True)
    refE2 = refE2.sol
    # plt.plot(t, refE2(t).T) 
    # plt.show()

    y0D = [3., 6.]

    def funD(t, y):
        # damped pendulum
        return [y[1], - sin(y[0]) - abs(y[1])*y[1]/4]
    
    def jacD(t, y):
        return np.array([[0, 1], [-cos(y[0]), -abs(y[1])/2]])
    
    refD = solve_ivp(funD, t_span, y0D, 
                     method=CFMR7osc,
                     atol=1e-12, rtol=1e-12, 
                     dense_output=True,
                     )
    refD = refD.sol
    
    # plt.plot(t, refD(t).T)
    # plt.show()
    # plt.plot(*refD(t))
    # plt.grid()
    # plt.axvline(1*np.pi)
    # plt.axvline(2*np.pi)
    # plt.axvline(3*np.pi)
    # plt.axvline(4*np.pi)
    # plt.show()
    # stop
    
    y0F = [0., 1.]

    def funF(t, y):
        # forced duffing
        return [y[1], cos(t) - y[0]*(1 + y[0]**2) - y[1]/2]
    
    def jacF(t, y):
        return np.array([[0, 1], [-1-3*y[0]**2, -1/2]])
    
    refF = solve_ivp(funF, t_span, y0F, 
                     method=CFMR7osc,
                     atol=1e-12, rtol=1e-12, 
                     dense_output=True,
                     )
    refF = refF.sol
    
    # plt.plot(t, refF(t).T)
    # plt.show()
    # stop
    
    y0G = [1, 0, 0, 0]
    
    def funG(t, y):
        # gyroscopic
        g = 1/3
        return [y[2], y[3], g*y[3]-y[0], -g*y[2]-y[1]]
    
    def jacG(t, y):
        g = 1/3
        return np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, g], [0, -1, -g, 0]])
    
    refG = solve_ivp(funG, t_span, y0G, 
                     method=CFMR7osc,
                     atol=1e-12, rtol=1e-12, 
                     dense_output=True,
                     )
    refG = refG.sol
    
    # plt.plot(t, refG(t).T)
    # plt.show()
    # stop


    for method in [#RK45,
                   Fi4N,
                   # N5stage,
                   # Fi4N_alt,
                   # Fi5N,
                   # Mu5Nmb,
                   # CFMR7osc,
                   # CMR7,
                   # DOP853
                   #Me4,
                   Me4N,
                   # Me4Nalt,   # seems worse
                   #DPM6NN,
                   #Me4Nalt,
                   #MR6NN,
                   # Me4Nalt,
                   # BS5,
                   #BS5N,
                   #SS4N,
                   #SS4,
                   # SS3,
                   # SS3N,
                   # RK23,
                   # BS3NnoFSAL,
                   # SWAG,
                   # MR4N,
                   # Pr4N,
                   # DMP4NSalt,
                   # DMP4NS,
                   Oz4NS,
                   MR6NS
                   ]:
        es = []
        ns = []
        nfs = []
        nfe = []
        for tol in np.logspace(-7, -2, 21):
            sol = solve_ivp(#**E2,
                            funF, t_span, y0F, jac=jacF,
                            method=method, atol=tol, rtol=1e-12,
                            dense_output=True,
                            scale_embedded=False
                            )
            es.append(np.linalg.norm(sol.y - refF(sol.t)))
            ns.append(sol.nfev)
            nfs.append(NFS[()])
            nfe.append(sol.nfev)
        print(method, np.sum(nfs), np.sum(nfe))
        plt.loglog(ns, es, '.-', label=method.__name__)
        if method in (Me4N, Me4Nalt):
            plt.loglog(np.array(ns)/5*3, es, '.--', label=method.__name__)
        if method in (Mu5Nmb, ):
            plt.loglog(np.array(ns)/9*5, es, '.--', label=method.__name__)

    plt.xlabel('nfev')
    plt.ylabel('err')
    plt.legend()
    plt.grid()
    plt.show()
