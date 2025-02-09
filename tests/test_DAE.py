"""Use Kaps problem to test DAE solver.

# vary formats of J and M
# include numerical J
# also check with inconsistent y0
also use a scrambled system with full M
# test yp0
# test if constraint is satisfied
# test if solution is accurate
# test if dense solution is accurate
# test with eps=1e-3
"""
import pytest
import numpy as np
from numpy.testing import assert_, assert_allclose
from itertools import product
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from extensisq import TRBDF2, TRX2, KC3I, KC4I, KC4Ia, Kv3I


methods = [TRBDF2, TRX2, KC3I, KC4I, KC4Ia, Kv3I]

# Kaps DAE


def fun(t, y, eps=0.):
    return np.array(
        [-(1 + 2*eps)*y[0] + y[1]**2,
         y[0] - y[1] - y[1]**2])


def jac(t, y, eps=0.):
    return np.array([
        [-(1 + 2*eps), 2*y[1]],
        [1, -1 - 2*y[1]]])


def jac_sparse(t, y, eps=0.):
    return csr_matrix(jac(t, y, eps))


def ref(t):
    return np.stack([np.exp(-t)**2, np.exp(-t)])


M_dense = np.array([[0, 0], [0, 1]])
M_sparse = csr_matrix(M_dense)
M_diag = np.array([0, 1.])

y0_consistent = [1., 1.]
y0_inconsistent = [2., 1.]
yp0 = [-2., -1.]
t_span = (0, 1.)


@pytest.mark.parametrize("method", methods)
def test_DAE(method):
    interpolant = {}
    if method in [KC3I, KC4I, KC4Ia, Kv3I]:
        interpolant = {'interpolant': 'C1'}
    for M, J, y0 in product([M_diag, M_sparse, M_dense],
                            [jac, None, jac_sparse],
                            [y0_consistent, y0_inconsistent]):
        sol = solve_ivp(fun, t_span, y0, method=method, jac=J, M=M,
                        dense_output=True, **interpolant)

        # initial values
        assert_allclose(sol.y[:, 0], y0)
        assert_allclose(sol.sol(sol.t[0]), y0_consistent)
        h = (sol.t[1] - sol.t[0])/10
        yp_numerical = (sol.sol(sol.t[0]+h) - sol.sol(sol.t[0]))/h
        assert_allclose(yp_numerical, yp0, atol=1e-5, rtol=1e-2)
        # final values
        t_final = t_span[1]
        y_final = ref(t_final)
        assert_allclose(sol.y[:, -1], y_final, atol=1e-5, rtol=1e-2)
        assert_allclose(sol.sol(t_final), y_final, atol=1e-5, rtol=1e-2)
        # dense output
        assert_allclose(sol.sol(sol.t)[:, 1:], sol.y[:, 1:])
        # solution
        assert_allclose(sol.y[:, 1:], ref(sol.t[1:]), atol=1e-5, rtol=1e-2)
        # constraint
        y_0, y_1 = sol.y[:, 1:]
        assert_allclose(y_1**2, y_0, atol=1e-6, rtol=1e-3)


eps = 1e-3
args = (eps, )
Mp_dense = np.array([[eps, 0], [0, 1]])
Mp_sparse = csr_matrix(M_dense)
Mp_diag = np.array([eps, 1.])


@pytest.mark.parametrize("method", methods)
def test_SPP(method):
    interpolant = {}
    if method in [KC3I, KC4I, KC4Ia, Kv3I]:
        interpolant = {'interpolant': 'C1'}
    for M, J, y0 in product([Mp_diag, Mp_sparse, Mp_dense],
                            [jac, None, jac_sparse],
                            [y0_consistent, y0_inconsistent]):
        sol = solve_ivp(fun, t_span, y0, method=method, jac=J, M=M,
                        dense_output=True, args=args, **interpolant)

        # initial values
        assert_allclose(sol.y[:, 0], y0)
        if y0 == y0_consistent:
            assert_allclose(sol.sol(sol.t[0]), y0, atol=1e-5, rtol=1e-2)
            h = (sol.t[1] - sol.t[0])/10
            yp_numerical = (sol.sol(sol.t[0]+h) - sol.sol(sol.t[0]))/h
            assert_allclose(yp_numerical, yp0, atol=1e-5, rtol=1e-2)
        # final values
        t_final = t_span[1]
        y_final = ref(t_final)
        assert_allclose(sol.y[:, -1], y_final, atol=1e-5, rtol=1e-2)
        assert_allclose(sol.sol(t_final), y_final, atol=1e-5, rtol=1e-2)
        # dense output
        assert_allclose(sol.sol(sol.t)[:, 1:], sol.y[:, 1:])


np.random.seed(1)
A = np.random.rand(2, 2)   # transform equations
B = np.random.rand(2, 2)   # transform y and yp
Binv = np.linalg.inv(B)
M_hidden = A @ M_dense @ Binv   # full matrix (not full rank)


def fun_hidden(t, y, eps=0, A=A):
    return A @ fun(t, Binv @ y, eps)


def jac_hidden(t, y, eps=0, A=A):
    return A @ jac(t, Binv @ y, eps) @ Binv


@pytest.mark.parametrize("method", methods)
def test_DAE_hidden(method):
    """Test if the methods can untangle the constraint from a mass matrix"""
    for y0 in [y0_consistent, y0_inconsistent]:
        sol = solve_ivp(fun, t_span, y0, method=method, jac=jac, M=M_dense,
                        dense_output=True, args=args)
        sol_hidden = solve_ivp(fun_hidden, t_span, B @ y0,
                               method=method, jac=jac_hidden, M=M_hidden)
        assert_(sol_hidden.success)
        # nr of steps, fun calls, jac calls
        # print(sol_hidden.t.size, sol.t.size)
        assert_(abs(sol_hidden.t.size - sol.t.size) < 3)
        print(sol_hidden.nfev, sol.nfev)
        assert_(abs(sol_hidden.nfev - sol.nfev) < 25)  # still quite dissimilar
        print(sol_hidden.njev, sol.njev)
        assert_(abs(sol_hidden.njev - sol.njev) < 2)

        # solutions similar
        assert_allclose(Binv @ sol_hidden.y[:, 0], sol.y[:, 0])
        if y0 == y0_consistent:
            assert_allclose(Binv @ sol_hidden.y, sol.sol(sol_hidden.t),
                            atol=1e-5, rtol=1e-2)
        else:
            assert_allclose(Binv @ sol_hidden.y[:, 1:],
                            sol.sol(sol_hidden.t[1:]),
                            atol=1e-5, rtol=1e-2)


def fun_e(t, y, eps, M=Mp_dense):
    return np.linalg.solve(M, fun(t, y, eps))


def jac_e(t, y, eps, M=Mp_dense):
    return np.linalg.solve(M, jac(t, y, eps))


def jac_e_sparse(t, y, eps=0.):
    return csr_matrix(jac_e(t, y, eps))


@pytest.mark.parametrize("method", methods)
def test_Mass(method):
    interpolant = {}
    if method in [KC3I, KC4I, KC4Ia, Kv3I]:
        interpolant = {'interpolant': 'C1'}
    for y0 in [y0_consistent, y0_inconsistent]:
        sol_m = solve_ivp(fun, t_span, y0, method=method, jac=jac, M=Mp_diag,
                          args=args)
        for J in [jac_e, jac_e_sparse]:
            sol = solve_ivp(fun_e, t_span, y0, method=method, jac=J,
                            dense_output=True, args=args, **interpolant)

            # nr of steps, fun calls, jac calls
            # print(sol_m.t.size, sol.t.size)
            assert_(abs(sol_m.t.size - sol.t.size) < 3)
            # print(sol_m.nfev, sol.nfev)
            assert_(abs(sol_m.nfev - sol.nfev) < 20)
            # print(sol_m.njev, sol.njev)
            assert_(abs(sol_m.njev - sol.njev) < 2)
            # solutions similar
            assert_allclose(sol_m.y, sol.sol(sol_m.t), atol=1e-5, rtol=1e-2)
