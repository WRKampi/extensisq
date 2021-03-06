# extensisq
This package extends scipy.integrate with various methods (OdeSolver classes) for the solve_ivp function.

![python:3](https://img.shields.io/pypi/pyversions/extensisq?style=flat-square)
![platform:noarch](https://img.shields.io/conda/pn/conda-forge/extensisq?style=flat-square)
[![license:MIT](https://img.shields.io/github/license/WRKampi/extensisq?style=flat-square)](https://github.com/WRKampi/extensisq/blob/main/LICENSE)
[![downloads pypi](https://img.shields.io/pypi/dm/extensisq?label=PyPI%20downloads&style=flat-square)](https://pypistats.org/packages/extensisq)
[![downloads conda](https://img.shields.io/conda/dn/conda-forge/extensisq?label=conda%20downloads&style=flat-square)](https://anaconda.org/conda-forge/extensisq)
[![release-date](https://img.shields.io/github/release-date/WRKampi/extensisq?style=flat-square)](https://github.com/WRKampi/extensisq/releases)


Currently, several explicit methods (for non-stiff problems) are provided.

One multistep method is implemented:
* `SWAG`: the variable order Adams-Bashforth-Moulton predictor-corrector method of Shampine, Gordon and Watts [5-7]. This is a translation of the Fortran code `DDEABM`. Matlab's method `ode113` is related.

Three explicit Runge Kutta methods of order 5 are implemented:
* `BS5`: efficient fifth order method by Bogacki and Shampine [1]. Three interpolants are included: the original accurate fifth order interpolant, a lower cost fifth order one, and a 'free' fourth order one.
* `CK5`: fifth order method with the coefficients from [2], for general use.
* `Ts5`: relatively new solver (2011) by Tsitouras, optimized with fewer simplifying assumptions [3].

Three higher order explicit Runge Kutta methods by Prince [4] are implemented:
* `Pr7`: a seventh order discrete method with fifth order error estimate, derived from a sixth order continuous method.
* `Pr8`: an eighth order discrete method with sixth order error estimate, derived from a seventh order continuous method.
* `Pr9`: a ninth order discrete method with seventh order error estimate, derived from an eighth order continuous method.

The numbers in the names refer to the discrete methods, while the orders in [4] refer to the continuous methods. These  methods are relatively efficient when dense output is needed, because the interpolants are free. (Other high-order methods typically need several additional function evaluations for dense output.)

One method for a specific type of problem is available:
* `CKdisc`: variable order solver by Cash and Karp, tailored to solve non-smooth problems efficiently [2].

## Other features
The initial step size, when not supplied by you, is estimated using the method of Watts [7]. This method analyzes your problem with a few (3 to 4) evaluations and carefully estimates a safe stepsize to start the integration with.

Most of extensisq's Runge Kutta methods have stiffness detection. If many steps fail, or if the integration needs a lot of steps, the power iteration method of Shampine [8] is used to test your problem for stiffness. You will get a warning if your problem is diagnosed as stiff. The kind of roots (real, complex or nearly imaginary) is also reported, such that you can select a stiff solver that better suits your problem.

Second order stepsize controllers [9, 10] can be enabled for most of extensisq's Runge Kutta methods. You can set your own coefficients, or select one of the default values.

## Installation

You can install extensisq from [PyPI](https://pypi.org/project/extensisq/):

    pip install extensisq

Or, if you'd rather use [conda](https://anaconda.org/conda-forge/extensisq):

    conda install -c conda-forge extensisq

## Example
Borrowed from the the scipy documentation:

    from scipy.integrate import solve_ivp
    from extensisq import BS5
    
    def exponential_decay(t, y): return -0.5 * y
    sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8], method=BS5)
    
    print(sol.t)
    print(sol.y)

Notice that the class `BS5` is passed to `solve_ivp`, not the string `"BS5"`. The other methods (`CK5`, `CKdisc`, `Ts5`, `Pr7`, `Pr8`, `Pr9` and `SWAG`) can be used in a similar way.

More examples are available as notebooks:
1. [Duffing's equation, Bogacki Shampine method](https://github.com/WRKampi/extensisq/blob/main/docs/Bogacki_Shampine.ipynb)
2. [Non-smooth problem, Cash Karp method](https://github.com/WRKampi/extensisq/blob/main/docs/Cash_Karp.ipynb)
3. [Lotka Volterra equation, all fifth order methods](https://github.com/WRKampi/extensisq/blob/main/docs/all_methods.ipynb)
4. [Riccati equation, higher order Prince methods](https://github.com/WRKampi/extensisq/blob/main/docs/Prince.ipynb)
5. [Van der Pol's equation, Shampine Gordon Watts method](https://github.com/WRKampi/extensisq/blob/main/docs/Shampine_Gordon_Watts.ipynb)

## References
[1] P. Bogacki, L.F. Shampine, "An efficient Runge-Kutta (4,5) pair", Computers & Mathematics with Applications, Vol. 32, No. 6, 1996, pp. 15-28. https://doi.org/10.1016/0898-1221(96)00141-1

[2] J. R. Cash, A. H. Karp, "A Variable Order Runge-Kutta Method for Initial Value Problems with Rapidly Varying Right-Hand Sides", ACM Trans. Math. Softw., Vol. 16, No. 3, 1990, pp. 201-222. https://doi.org/10.1145/79505.79507

[3] Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the first column simplifying assumption", Computers & Mathematics with Applications, Vol. 62, No. 2, 2011, pp. 770 - 775. https://doi.org/10.1016/j.camwa.2011.06.002

[4] P.J. Prince, "Parallel Derivation of Efficient Continuous/Discrete Explicit Runge-Kutta Methods", Guisborough TS14 6NP U.K., September 6 2018. http://www.peteprince.co.uk/parallel.pdf

[5] L.F. Shampine and M.K. Gordon, "Computer solution of ordinary differential equations: The initial value problem", San Francisco, W.H. Freeman, 1975.

[6] H.A. Watts and L.F. Shampine, "Smoother Interpolants for Adams Codes",  SIAM Journal on Scientific and Statistical Computing, Vol. 7, No. 1, 1986, pp. 334-345. https://doi.org/10.1137/0907022

[7] H.A. Watts, "Starting step size for an ODE solver", Journal of Computational and Applied Mathematics, Vol. 9, No. 2, 1983, pp. 177-191. https://doi.org/10.1016/0377-0427(83)90040-7

[8] L.F. Shampine, "Diagnosing Stiffness for Runge–Kutta Methods", SIAM Journal on Scientific and Statistical Computing, Vol. 12, No. 2, 1991, pp. 260-272. https://doi.org/10.1137/0912015

[9] G. Söderlind, "Digital Filters in Adaptive Time-Stepping", ACM Trans. Math. Softw. Vol 29, No. 1, 2003, pp. 1–26. https://doi.org/10.1145/641876.641877

[10] K. Gustafsson, "Control-Theoretic Techniques for Stepsize Selection in Implicit Runge-Kutta Methods", ACM Trans. Math. Softw., Vol. 20, No. 4, 1994, pp. 496–517. https://doi.org/10.1145/198429.198437
