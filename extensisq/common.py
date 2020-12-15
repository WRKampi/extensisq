import numpy as np
from scipy.integrate._ivp.rk import (
    RungeKutta, RkDenseOutput, SAFETY, MIN_FACTOR, MAX_FACTOR)


class RungeKuttaConv(RungeKutta):
    """Modified RungeKutta class for conventional explicit methods.

    The RungeKutta class from scipy is created for FSAL pairs which need the
    evaluation of the output stage for the error estimate. Conventional pairs
    can evaluate the error before this evaluation. Therefore, one evaluation
    can be saved for all rejected steps. This is the main reason to create this
    modified class.

    The coefficient arrays are defined identicaly to those in the default
    class. This makes the two classes compatible. The RungeKuttaConv class
    may be used if the last number in the E array is 0.

    The dense output is calculated with Horner's method.
    """

    def __init__(self, *args, **kwargs):
        super(RungeKuttaConv, self).__init__(*args, **kwargs)
        assert self.E.size == self.n_stages + 1, \
            'The lenght of the array E should equal n_stages + 1'
        assert self.E[self.n_stages] == 0., \
            'The class RungeKuttaConv can only be applied if E[-1] == 0.'

    def _step_impl(self):
        # mostly follows the scipy implementation of RungeKutta
        t = self.t
        y = self.y
        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > self.max_step:
            h_abs = self.max_step
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

            # calculate stages, except last
            self.K[0] = self.f
            for i in range(1, self.n_stages):
                self._rk_stage(h, i)

            # calculate error_norm and solution
            y_new, error_norm = self._comp_sol_err(y, h)

            # and evaluate
            if error_norm < 1:
                step_accepted = True
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm**self.error_exponent)
                if step_rejected:
                    factor = min(1, factor)
                h_abs *= factor

                # now calculate last stage
                f_new = self.fun(t + h, y_new)
                self.K[-1] = f_new
            else:
                step_rejected = True
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm**self.error_exponent)

        self.h_previous = h
        self.y_old = y
        self.t = t_new
        self.y = y_new
        self.h_abs = h_abs
        self.f = f_new
        return True, None

    def _estimate_error(self, K, h):
        # exclude K[-1]. It could contain nan or inf
        return (K[:self.n_stages].T @ self.E[:self.n_stages]) * h

    def _comp_sol_err(self, y, h):
        # compute solution and error norm of step
        y_new = y + h * (self.K[:-1].T @ self.B)
        scale = self.atol + np.maximum(np.abs(y), np.abs(y_new)) * self.rtol
        error_norm = self._estimate_error_norm(self.K, h, scale)
        return y_new, error_norm

    def _rk_stage(self, h, i):
        # compute a single RK stage
        dy = self.K[:i, :].T @ self.A[i, :i] * h
        self.K[i] = self.fun(self.t + self.C[i]*h, self.y + dy)

    def _dense_output_impl(self):
        # P[:,0] just selects K[0].
        # Therefore, P[:,0] is ignored and K[0] is passed directly.
        Q = self.K.T @ self.P[:, 1:]
        return HornerDenseOutput(self.t_old, self.t, self.y_old, self.K[0], Q)


class HornerDenseOutput(RkDenseOutput):
    """use Horner's rule for the evaluation of the dense output polynomials.
    """

    def __init__(self, t_old, t, y_old, yp_old, Q):
        super(HornerDenseOutput, self).__init__(t_old, t, y_old, Q)
        self.order += 1       # because Q is 1 smaller than usual
        self.yp_old = yp_old.copy()

    def _call_impl(self, t):

        # scaled time
        x = (t - self.t_old) / self.h

        # Horner's rule:
        y = self.Q.T[-1, :, np.newaxis] * x
        for q in reversed(self.Q.T[:-1]):
            y += q[:, np.newaxis]
            y *= x
        y += self.yp_old[:, np.newaxis]
        y *= x * self.h
        y += self.y_old[:, np.newaxis]

        # need this `if` to pass scipy's unit tests. I'm not sure why.
        if t.shape:
            return y
        else:
            return y[:, 0]
