import pytest
from numpy.testing import assert_, assert_allclose, assert_equal
from scipy.integrate import solve_ivp
import numpy as np
from extensisq import Fi4N, Fi5N, Mu5Nmb
from extensisq.common import norm
from itertools import product


METHODS = [Fi4N, Fi5N, Mu5Nmb]


def fun_linear(t, y):
    return np.array([y[1], -y[0]])


def fun_linear_vectorized(t, y):
    return np.vstack((y[1] * np.ones_like(t),
                      -y[0] * np.ones_like(t)))


def sol_linear(t):
    return np.vstack((np.sin(t),
                      np.cos(t)))


def compute_error(y, y_true, rtol, atol):
    e = (y - y_true) / (atol + rtol * np.abs(y_true))
    return np.linalg.norm(e, axis=0) / np.sqrt(e.shape[0])


y0 = [0, 1]


@pytest.mark.parametrize("solver", METHODS)
def test_coefficient_properties(solver):
    assert_allclose(np.sum(solver.B), 0.5, rtol=1e-13)
    assert_allclose(np.sum(solver.Bp), 1, rtol=1e-13)
    assert_allclose(np.sum(solver.E), 0, atol=1e-13)
    assert_allclose(np.sum(solver.Ep), 0, atol=1e-13)
    assert_allclose(np.sum(solver.Ap, axis=1), solver.C, rtol=1e-13)
    assert_allclose(np.sum(solver.A, axis=1), 0.5*solver.C**2, rtol=1e-13)


@pytest.mark.parametrize("solver_class", METHODS)
def test_error_estimation(solver_class):
    step = 0.2
    solver = solver_class(lambda t, y: [y[1], -y[0]], 0, [1, 0], 1, first_step=step)
    solver.step()
    error_estimate = solver._estimate_error(solver.K, step)
    error = solver.y - np.array([np.cos(step), -np.sin(step)])
    # print(np.abs(error), np.abs(error_estimate))
    assert_(norm(error) < norm(error_estimate))


@pytest.mark.parametrize("solver_class", METHODS)
def test_error_estimation_complex(solver_class):
    h = 0.2
    solver = solver_class(lambda t, y: [y[1], -1j*y[0]], 0, [1j, 1], 1, first_step=h)
    solver.step()
    err_norm = solver._estimate_error_norm(solver.K, h, scale=[1])
    assert np.isrealobj(err_norm)


@pytest.mark.parametrize('method', METHODS)
def test_integration(method):
    rtol = 1e-3
    atol = 1e-6

    for vectorized, t_span in product(
            [False, True],
            [[0, 2*np.pi], [2*np.pi, 0]]
            ):

        if vectorized:
            fun = fun_linear_vectorized
        else:
            fun = fun_linear

        res = solve_ivp(fun, t_span, y0, rtol=rtol, atol=atol, method=method,
                        dense_output=True, vectorized=vectorized)
        assert_equal(res.t[0], t_span[0])
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)

        if method == Mu5Nmb:
            # Mu5Nmb has a relatively low error, with relatively many evals
            assert_(res.nfev < 130)
        else:
            assert_(res.nfev < 60)

        assert_equal(res.njev, 0)
        assert_equal(res.nlu, 0)

        y_true = sol_linear(res.t)
        e = compute_error(res.y, y_true, rtol, atol)
        assert_(np.median(e) < 1)

        tc = np.linspace(*t_span)
        yc_true = sol_linear(tc)
        yc = res.sol(tc)

        e = compute_error(yc, yc_true, rtol, atol)
        assert_(np.median(e) < 1)

        tc = (5*t_span[0] + 3*t_span[-1])/8
        yc_true = sol_linear(tc).T
        yc = res.sol(tc)

        e = compute_error(yc, yc_true, rtol, atol)
        assert_(np.all(e < 5))

        assert_allclose(res.sol(res.t), res.y,
                        rtol=1e-11, atol=1e-12)         # relaxed tol


@pytest.mark.parametrize('cls', METHODS)
def test_classes(cls):
    y0 = [0, 1]
    solver = cls(fun_linear, 0, y0, np.inf)
    #                  fun, t0, y0, t_bound
    assert_equal(solver.n, 1)
    assert_equal(solver.status, 'running')
    assert_equal(solver.t_bound, np.inf)
    assert_equal(solver.direction, 1)
    assert_equal(solver.t, 0)
    assert_equal(solver.y, y0)
    assert_(solver.step_size is None)
    assert_(solver.nfev > 0)
    assert_(solver.njev >= 0)
    assert_equal(solver.nlu, 0)
    with pytest.raises(RuntimeError):
        solver.dense_output()
    message = solver.step()
    assert_equal(solver.status, 'running')
    assert_equal(message, None)
    assert_equal(solver.n, 1)
    assert_equal(solver.t_bound, np.inf)
    assert_equal(solver.direction, 1)
    assert_(solver.t > 0)
    assert_(not np.all(np.equal(solver.y, y0)))
    assert_(solver.step_size > 0)
    assert_(solver.nfev > 0)
    assert_(solver.njev >= 0)
    assert_(solver.nlu >= 0)
    sol = solver.dense_output()
    assert_allclose(sol(0), y0, rtol=1e-14, atol=0)


@pytest.mark.parametrize('method', METHODS)
def test_wrong_problem(method):
    # odd nr of components
    fun = lambda t, y: -y
    with pytest.raises(AssertionError):
        method(fun, 0, [1], 1)
    # dx != v
    fun = lambda t, y: [-y[1], y[0]]
    with pytest.raises(AssertionError):
        method(fun, 0, [0, 1], 1)
    with pytest.raises(AssertionError):
        method(fun, 0, [1, 1], 1)
    with pytest.raises(AssertionError):
        method(fun, 0, [0, 0], 1)
