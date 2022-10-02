import numpy as np
from scipy.integrate import solve_ivp
from extensisq import BS5, Ts5, CK5, CKdisc, SWAG, Pr7, Pr8, Pr9
from extensisq.common import NFS
from math import cos, sin
import matplotlib.pyplot as plt
import warnings


warnings.simplefilter("always")

# import logging
# logging.captureWarnings(True)
# logging.basicConfig(level=logging.WARNING)


def fun2(t, y):
    # t = 0
    #~ print(np.abs(np.linalg.eigvals(dfun2(t, y))))
    return [-2000*( y[0]*cos(t) + y[1]*sin(t) + 1), 
            -2000*(-y[0]*sin(t) + y[1]*cos(t) + 1)]


y02 = [1, 0]
t_span2 = [0, 1.57]


methods = [BS5, Ts5, CK5, Pr7, Pr8, Pr9]
for method in methods:
    print(f'{method}')
    sol = solve_ivp(fun2, t_span2, y02, rtol=1e-10, atol=1, method=method)
    print('nfev', sol.nfev)
    print('nfs', NFS)

    print(f'no detect {method}')
    sol = solve_ivp(fun2, t_span2, y02, rtol=1e-10, atol=1, method=method,
                    nfev_stiff_detect=0)
    print('nfev', sol.nfev)
    print('nfs', NFS)

    print(f'Central coefs {method}')
    sol = solve_ivp(fun2, t_span2, y02, rtol=1e-10, atol=1, method=method,
                    sc_params="G")
    print('nfev', sol.nfev)
    print('nfs', NFS)


#~ plt.plot(sol.t[:-1], np.diff(sol.t))
#~ plt.show()

#~ plt.plot(sol.t, sol.y.T)
#~ plt.show()