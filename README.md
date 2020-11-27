# extensisq
This package extends scipy.integrate with OdeSolver objects for the solve_ivp function.

![platform](https://img.shields.io/conda/pn/conda-forge/extensisq)
![python](https://img.shields.io/pypi/pyversions/extensisq)
[![license](https://img.shields.io/github/license/WRKampi/extensisq)](https://github.com/WRKampi/extensisq/blob/main/LICENSE)
[![downloads](https://img.shields.io/pypi/dm/extensisq?label=PyPI%20downloads)](https://pypistats.org/packages/extensisq)
[![downloads](https://img.shields.io/conda/dn/conda-forge/extensisq?label=conda%20downloads)](https://anaconda.org/conda-forge/extensisq)



Currently, three explicit Runge Kutta methods of order 5 are implemented:
* Bogacki Shampine: efficient solver with an accurate high order interpolant (and a free low order one) [1]
* Cash Karp: variable order solver tailored to solve non-smooth problems efficiently [2]
* Tsitouras: relatively new solver (2011), optimized with fewer simplifying assumptions [3]

The first two methods have two variants each.

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

Note that the object `BS45_i` is passed, not the string `"BS45_i"`. The other methods (`BS45`, `CK45`, `CK45_o` and `Ts45`) can be used in a similar way.

More examples are available as notebooks:
1. [Lotka Volterra equation, all methods](https://github.com/WRKampi/extensisq/blob/main/docs/all_methods.ipynb)
2. [Duffing's equation, Bogacki Shampine method](https://github.com/WRKampi/extensisq/blob/main/docs/Bogacki_Shampine.ipynb)
3. [Non-smooth problem, Cash Karp method](https://github.com/WRKampi/extensisq/blob/main/docs/Cash_Karp.ipynb)

## References
[1] P. Bogacki, L.F. Shampine, "An efficient Runge-Kutta (4,5) pair", Computers & Mathematics with Applications, Vol. 32, No. 6, 1996, pp. 15-28, ISSN 0898-1221. https://doi.org/10.1016/0898-1221(96)00141-1

[2] J. R. Cash, A. H. Karp, "A Variable Order Runge-Kutta Method for Initial Value Problems with Rapidly Varying Right-Hand Sides", ACM Trans. Math. Softw., Vol. 16, No. 3, 1990, pp. 201-222, ISSN 0098-3500. https://doi.org/10.1145/79505.79507

[3] Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the first column simplifying assumption", Computers & Mathematics with Applications, Vol. 62, No. 2, pp. 770 - 775, 2011. https://doi.org/10.1016/j.camwa.2011.06.002
