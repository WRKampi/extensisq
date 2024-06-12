# extensisq
This package extends scipy.integrate with various methods (OdeSolver classes) for the solve_ivp function.

![python:3](https://img.shields.io/pypi/pyversions/extensisq?style=flat-square)
![platform:noarch](https://img.shields.io/conda/pn/conda-forge/extensisq?style=flat-square)
[![license:MIT](https://img.shields.io/github/license/WRKampi/extensisq?style=flat-square)](https://github.com/WRKampi/extensisq/blob/main/LICENSE)
[![downloads pypi](https://img.shields.io/pypi/dm/extensisq?label=PyPI%20downloads&style=flat-square)](https://pypistats.org/packages/extensisq)
[![downloads conda](https://img.shields.io/conda/dn/conda-forge/extensisq?label=conda%20downloads&style=flat-square)](https://anaconda.org/conda-forge/extensisq)
[![release-date](https://img.shields.io/github/release-date/WRKampi/extensisq?style=flat-square)](https://github.com/WRKampi/extensisq/releases)


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

Notice that the class `BS5` is passed to `solve_ivp`, not the string `"BS5"`. The other methods (`SWAG`, `CK5`, `Ts5`, `Me4`, `Pr7`, `Pr8`, `Pr9`, `CKdisc`, `CFMR7osc`, `SSV2stab`, `Fi4N`, `Fi5N`, `Mu5Nmb`, and `MR6NN`) can be used in a similar way.

More examples are available as notebooks (update needed):
1. [Integration with Scipy's `solve_ivp` function](https://github.com/WRKampi/extensisq/blob/main/docs/Demo_solve_ivp.ipynb)
2. [About `BS5` and its interpolants](https://github.com/WRKampi/extensisq/blob/main/docs/Demo_BS5.ipynb)
3. [Higher order Prince methods `Pr7`, `Pr8` and `Pr9`](https://github.com/WRKampi/extensisq/blob/main/docs/Prince.ipynb)
4. [Special method `CKdisc` for non-smooth problems](https://github.com/WRKampi/extensisq/blob/main/docs/Demo_CKdisc.ipynb)
5. [Special method `CFMR7osc` for oscillatory problems](https://github.com/WRKampi/extensisq/blob/main/docs/Demo_CFMR7osc.ipynb)
6. [Special method `SSV2stab` for large, mildly stiff problems](https://github.com/WRKampi/extensisq/blob/main/docs/Demo_SSV2stab.ipynb)
7. [Fifth order methods compared](https://github.com/WRKampi/extensisq/blob/main/docs/all_methods.ipynb)
8. [Van der Pol's equation, Shampine Gordon Watts method](https://github.com/WRKampi/extensisq/blob/main/docs/Shampine_Gordon_Watts.ipynb)
9. [Runge Kutta Nyström methods for second order equations](docs/Demo_Nystrom.ipynb)
10. [Sensitivity analysis](https://github.com/WRKampi/extensisq/blob/main/docs/Demo_sensitivity.ipynb)
11. [How to implement other explicit Runge Kutta methods](https://github.com/WRKampi/extensisq/blob/main/docs/Demo_own_RK.ipynb)


## Methods

Currently, several explicit methods (for non-stiff problems) are provided.

One multistep method is implemented:
* `SWAG`: the variable order Adams-Bashforth-Moulton predictor-corrector method of Shampine, Gordon and Watts [5-7]. This is a translation of the Fortran code `DDEABM`. Matlab's method `ode113` is related.

Three explicit Runge Kutta methods of order 5 are implemented:
* `BS5`: efficient fifth order method by Bogacki and Shampine [1,A]. Three interpolants are included: the original accurate fifth order interpolant, a lower cost fifth order one, and a 'free' fourth order one.
* `CK5`: fifth order method with the coefficients from [2], for general use.
* `Ts5`: relatively new solver (2011) by Tsitouras, optimized with fewer simplifying assumptions [3].

One fourth order method:
* `Me4`: Merson's method, the first embedded RK method [14]. The embedded method for error estimation is 5th order for linear problems and 3rd order for general problems. A 3rd order interpolant is added. This method has a large stability region. It may be useful as alternative to 'RK23' for solving problems to lower accuracy.

Three higher order explicit Runge Kutta methods by Prince [4] are implemented:
* `Pr7`: a seventh order discrete method with fifth order error estimate, derived from a sixth order continuous method.
* `Pr8`: an eighth order discrete method with sixth order error estimate, derived from a seventh order continuous method.
* `Pr9`: a ninth order discrete method with seventh order error estimate, derived from an eighth order continuous method.

The numbers in the names refer to the discrete methods, while the orders in [4] refer to the continuous methods. These  methods are relatively efficient when dense output is needed, because the interpolants are free. (Other high-order methods typically need several additional function evaluations for dense output.)

Three methods for specific types of problems are available:
* `CKdisc`: variable order solver by Cash and Karp, tailored to solve non-smooth problems efficiently [2].
* `CFMR7osc`: explicit Runge Kutta method, with algebraic order 7, dispersion order 10 and dissipation order 9, to efficiently and accurately solve problems with oscillating solutions [12]. A free 5th order interpolant for dense output is added.
* `SSV2stab`: second order stabilized Runge Kutta Chebyshev method [13,C], to explicity and efficiently solve large systems of mildly stiff ordinary differential equations up to low to moderate accuracy. Equations arising from semi-discretization of parabolic PDEs are a typical use case.

Several Nystrom methods are added. These are for second order initial value problems. Three methods are for general problems and one is for the strict problem in which the second derivative should not depend on the first derivative. The [demo](docs/Demo_Nystrom.ipynb) shows how to use these methods.
* `Fi4N`: 4th order general Nystrom method of Fine [16].
* `Fi5N`: 5th order general Nystrom method of Fine [16, 17].
* `Mu5Nmb`: 5th order general Nystrom method of Murua for integration of multibody equations. This is method "RKN5459" in the paper [18]. I added two interpolants.
* `MR6NN`: 6th order strict Nystrom method of El-Mikkawy and Rahmo [19]. I couldn't find the interpolant that the paper refers to as future work. However, I created a free C2-continuous sixth order interpolant and added it to this method.

## Sensitivity analysis
Three methods for sensitiviy analysis are available; see [15] and Example 9 above. These can be used with any of the solvers.
* `sens_forward`: to calculate the sensitivity of all solution components to (a few) parameters.
* `sens_adjoint_end`: to calculate the sensitivity of a scalar function of the solution to (many) parameters.
* `sens_adjoint_int`: to calculate the sensitivity of a scalar integral of the solution to (many) parameters.

## Other features
The initial step size, when not supplied by you, is estimated using the method of Watts [7,B]. This method analyzes your problem with a few (3 to 4) evaluations and carefully estimates a safe stepsize to start the integration with.

Most of extensisq's Runge Kutta methods have stiffness detection. If many steps fail, or if the integration needs a lot of steps, the power iteration method of Shampine [8,A] is used to test your problem for stiffness. You will get a warning if your problem is diagnosed as stiff. The kind of roots (real, complex or nearly imaginary) is also reported, such that you can select a stiff solver that better suits your problem.

Second order stepsize controllers [9-11] can be enabled for most of extensisq's Runge Kutta methods. You can set your own coefficients, or select one of the default values.

## References
[1] P. Bogacki, L.F. Shampine, "An efficient Runge-Kutta (4,5) pair", Computers & Mathematics with Applications, Vol. 32, No. 6, 1996, pp. 15-28. https://doi.org/10.1016/0898-1221(96)00141-1

[2] J. R. Cash, A. H. Karp, "A Variable Order Runge-Kutta Method for Initial Value Problems with Rapidly Varying Right-Hand Sides", ACM Trans. Math. Softw., Vol. 16, No. 3, 1990, pp. 201-222. https://doi.org/10.1145/79505.79507

[3] Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the first column simplifying assumption", Computers & Mathematics with Applications, Vol. 62, No. 2, 2011, pp. 770 - 775. https://doi.org/10.1016/j.camwa.2011.06.002

[4] P.J. Prince, "Parallel Derivation of Efficient Continuous/Discrete Explicit Runge-Kutta Methods", Guisborough TS14 6NP U.K., September 6 2018. http://www.peteprince.co.uk/parallel.pdf

[5] L.F. Shampine and M.K. Gordon, "Computer solution of ordinary differential equations: The initial value problem", San Francisco, W.H. Freeman, 1975.

[6] H.A. Watts and L.F. Shampine, "Smoother Interpolants for Adams Codes",  SIAM Journal on Scientific and Statistical Computing, Vol. 7, No. 1, 1986, pp. 334-345. https://doi.org/10.1137/0907022

[7] H.A. Watts, "Starting step size for an ODE solver", Journal of Computational and Applied Mathematics, Vol. 9, No. 2, 1983, pp. 177-191. https://doi.org/10.1016/0377-0427(83)90040-7

[8] L.F. Shampine, "Diagnosing Stiffness for Runge–Kutta Methods", SIAM Journal on Scientific and Statistical Computing, Vol. 12, No. 2, 1991, pp. 260-272. https://doi.org/10.1137/0912015

[9] K. Gustafsson, "Control Theoretic Techniques for Stepsize Selection in Explicit Runge-Kutta Methods", ACM Trans. Math. Softw., Vol. 17, No. 4, 1991, pp. 533–554. https://doi.org/10.1145/210232.210242

[10] G.Söderlind, "Automatic Control and Adaptive Time-Stepping", Numerical Algorithms, Vol. 31, No. 1, 2002, pp. 281-310. https://doi.org/10.1023/A:1021160023092

[11] G. Söderlind, "Digital Filters in Adaptive Time-Stepping", ACM Trans. Math. Softw., Vol. 29, No. 1, 2003, pp. 1–26. https://doi.org/10.1145/641876.641877

[12] M. Calvo, J.M. Franco, J.I. Montijano, L. Rández, "Explicit Runge-Kutta methods for initial value problems with oscillating solutions", Journal of Computational and Applied Mathematics, Vol. 76, No. 1–2, 1996, pp. 195-212. https://doi.org/10.1016/S0377-0427(96)00103-3

[13] B.P. Sommeijer, L.F. Shampine, J.G. Verwer, "RKC: An explicit solver for parabolic PDEs", Journal of Computational and Applied Mathematics, Vol. 88, No. 2, 1998, pp. 315-326. https://doi.org/10.1016/S0377-0427(97)00219-7

[14] E. Hairer, G. Wanner, S.P. Norsett, "Solving Ordinary Differential Equations I", Springer Berlin, Heidelberg, 1993, https://doi.org/10.1007/978-3-540-78862-1

[15] R.Serban, A.C. Hindmarsh, "CVODES: The Sensitivity-Enabled ODE Solver in SUNDIALS", 5th International Conference on Multibody Systems Nonlinear Dynamics and Control, Vol. 6, 2005, https://doi.org/10.1115/DETC2005-85597

[16] J.M. Fine, "Low order practical Runge-Kutta-Nyström methods", Computing, Vol. 38, 1987, pp. 281–297, https://doi.org/10.1007/BF02278707

[17] J.M. Fine, "Interpolants for Runge-Kutta-Nyström methods", Computing, Vol. 39, 1987, pp. 27–42, https://doi.org/10.1007/BF02307711

[18] A. Murua, "Runge-Kutta-Nyström methods for general second order ODEs with application to multi-body systems", Applied Numerical Mathematics, Vol. 28, Issues 2–4, 1998, pp. 387-399, https://doi.org/10.1016/S0168-9274(98)00055-5

[19] M. El-Mikkawy, E.D. Rahmo, "A new optimized non-FSAL embedded Runge–Kutta–Nystrom algorithm of orders 6 and 4 in six stages", Applied Mathematics and Computation, Vol. 145, Issue 1, 2003, pp. 33-43, https://doi.org/10.1016/S0096-3003(02)00436-8


## Original source codes (Fortran)

[A] RKSuite, R.W. Brankin,  I. Gladwell,  L.F. Shampine. https://www.netlib.org/ode/rksuite/

[B] DDEABM, L.F. Shampine, H.A. Watts, M.K. Gordon. https://www.netlib.org/slatec/src/

[C] RKC, B.P. Sommeijer, L.F. Shampine, J.G. Verwer. https://www.netlib.org/ode/
