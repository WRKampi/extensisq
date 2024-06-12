import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from extensisq import Fi4N


errors = []


def calculate_scale(atol, rtol, y, y_new, h, _mean=False):
    """calculate a scaling vector for the error estimate"""
    N = y.size
    _atol = np.full(N, atol)
    _atol[N//2:] /= h
    return _atol + rtol * np.maximum(np.abs(y), np.abs(y_new))


class Fi4N_alt(Fi4N):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_exponent_u = -1 / (self.error_estimator_order + 1)
        self.error_exponent_v = -1 / (self.error_estimator_order + 2)
        self.error_exponent = self.error_exponent_v 
        print(self.error_exponent)
    
    def _estimate_error_norm(self, K, h, scale):
        err = self._estimate_error(K, h)
        N = scale.size//2
        norm_u = norm(err[:N]/scale[:N])
        norm_v = norm(err[N:]/scale[N:])
        # print(norm_u, norm_v)
        errors.append([norm_u, norm_v])
        return (0*norm_u + norm_v)

    def _comp_sol_err(self, y, h):
        du = (self.K[:self.n_stages, :].T @ self.B) * h**2 \
            + h * self.y[self.n:]
        dv = (self.K[:self.n_stages, :].T @ self.Bp) * h
        dy = np.concatenate((du, dv))
        y_new = y + dy
        scale = calculate_scale(self.atol, self.rtol, y, y_new, h)

        if self.FSAL:
            # do FSAL evaluation if needed for error estimate
            self.K[self.n_stages, :] = self.fun(self.t + h, y_new)

        error_norm = self._estimate_error_norm(self.K, h, scale)
        return y_new, error_norm


omega = 3.


def fun(t, y, omega=omega):
    return np.concatenate(
        [y[3:], -omega**2*y[:3]])


y0 = np.concatenate([
    np.sin([-2/3*np.pi, 0, 2/3*np.pi]),
    omega*np.cos([-2/3*np.pi, 0, 2/3*np.pi])])
tspan = (0., 20/omega)

sol = solve_ivp(fun, tspan, y0, method=Fi4N_alt, sc_params="standard",
                atol=1e-3, rtol=1e-12)
errors = np.asarray(errors)
print(errors.shape)

# plt.plot(sol.t, sol.y[:3, :].T)
# plt.show()

# plt.plot(sol.t, sol.y[3:, :].T)
# plt.show()

fig, axs = plt.subplots(2)
axs[0].plot(np.diff(sol.t), ':.')
axs[1].plot(errors[1:-3])
print(errors[:, 0]/errors[:, 1])
plt.show()
