# extensisq
Extend scipy.integrate with extra ode solvers for solve_ivp.

Currently, three explicit Runge Kutta methods of order 5 are implemented:
* Bogacki Shampine: efficient solver with an accurate high order interpolant (and a free low order one)
* Cash Karp: variable order solver tailored to solve non-smooth problems efficiently
* Tsitouras: relatively new solver (2011), optimized with fewer simplifying assumptions

