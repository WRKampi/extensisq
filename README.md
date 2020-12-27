# extensisq
This package extends scipy.integrate with various methods (OdeSolver classes) for the solve_ivp function.

![python:3](https://img.shields.io/pypi/pyversions/extensisq?style=flat-square)
![platform:noarch](https://img.shields.io/conda/pn/conda-forge/extensisq?style=flat-square)
[![license:MIT](https://img.shields.io/github/license/WRKampi/extensisq?style=flat-square)](https://github.com/WRKampi/extensisq/blob/main/LICENSE)
[![downloads pypi](https://img.shields.io/pypi/dm/extensisq?label=PyPI%20downloads&style=flat-square)](https://pypistats.org/packages/extensisq)
[![downloads conda](https://img.shields.io/conda/dn/conda-forge/extensisq?label=conda%20downloads&style=flat-square)](https://anaconda.org/conda-forge/extensisq)
[![release-date](https://img.shields.io/github/release-date/WRKampi/extensisq?style=flat-square)](https://github.com/WRKampi/extensisq/releases)


Currently, several explicit Runge Kutta methods (for non-stiff problems) are provided.

Three explicit Runge Kutta methods of order 5 and two variants are implemented:
* `BS45`: efficient solver with an accurate high order interpolant by Bogacki and Shampine [1]. The variant `BS45_i` has a free, lower order interpolant.
* `CK45`: variable order solver by Cash and Karp, tailored to solve non-smooth problems efficiently [2]. The variant `CK45_o` is a fixed (fifth) order method with the same coefficients. 
* `Ts45`: relatively new solver (2011) by Tsitouras, optimized with fewer simplifying assumptions [3].


Three higher order explicit Runge Kutta methods by Prince [4] are implemented:
* `Pri6`: a seventh order discrete method with fifth order error estimate, derived from a sixth order continuous method.
* `Pri7`: an eighth order discrete method with sixth order error estimate, derived from a seventh order continuous method.
* `Pri8`: a ninth order discrete method with seventh order error estimate, derived from an eighth order continuous method.

The numbers in the names refer to the continuous methods. These higher order methods, unlike conventional discrete methods, do not require additional function evaluations for dense output.

## Installation

You can install extensisq from [PyPI](https://pypi.org/project/extensisq/):

    pip install extensisq

Or, if you'd rather use [conda](https://anaconda.org/conda-forge/extensisq):

    conda install -c conda-forge extensisq

## Example
Borrowed from the the scipy documentation:

    from scipy.integrate import solve_ivp
    from extensisq import BS45_i
    
    def exponential_decay(t, y): return -0.5 * y
    sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8], method=BS45_i)
    
    print(sol.t)
    print(sol.y)

Note that the class `BS45_i` is passed to `solve_ivp`, not the string `"BS45_i"`. The other methods (`BS45`, `CK45`, `CK45_o`, `Ts45`, `Pri6`, `Pri7` and `Pri8`) can be used in a similar way.

More examples are available as notebooks:
1. [Duffing's equation, Bogacki Shampine method](https://github.com/WRKampi/extensisq/blob/main/docs/Bogacki_Shampine.ipynb)
2. [Non-smooth problem, Cash Karp method](https://github.com/WRKampi/extensisq/blob/main/docs/Cash_Karp.ipynb)
3. [Lotka Volterra equation, all fifth order methods](https://github.com/WRKampi/extensisq/blob/main/docs/all_methods.ipynb)
4. [Riccati equation, higher order Prince methods](https://github.com/WRKampi/extensisq/blob/main/docs/Prince.ipynb)


## References
[1] P. Bogacki, L.F. Shampine, "An efficient Runge-Kutta (4,5) pair", Computers & Mathematics with Applications, Vol. 32, No. 6, 1996, pp. 15-28, ISSN 0898-1221. https://doi.org/10.1016/0898-1221(96)00141-1

[2] J. R. Cash, A. H. Karp, "A Variable Order Runge-Kutta Method for Initial Value Problems with Rapidly Varying Right-Hand Sides", ACM Trans. Math. Softw., Vol. 16, No. 3, 1990, pp. 201-222, ISSN 0098-3500. https://doi.org/10.1145/79505.79507

[3] Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the first column simplifying assumption", Computers & Mathematics with Applications, Vol. 62, No. 2, pp. 770 - 775, 2011. https://doi.org/10.1016/j.camwa.2011.06.002

[4] P.J. Prince, "Parallel Derivation of Efficient Continuous/Discrete Explicit Runge-Kutta Methods", Guisborough TS14 6NP U.K., September 6 2018. http://www.peteprince.co.uk/parallel.pdf
