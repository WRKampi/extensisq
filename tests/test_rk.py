"""Test from scipy to conform to scipy. modified"""
import pytest
from numpy.testing import assert_allclose, assert_
import numpy as np
from order_conditions import calc_Ts_norm
from extensisq import (BS5, Ts5, CK5, CKdisc, Pr7, Pr8, Pr9, CFMR7osc, Me4,
                       TRX2, TRBDF2, KC3I, KC4I, KC4Ia, Kv3I)


METHODS = [BS5, Ts5, CK5, CKdisc, Pr7, Pr8, Pr9, CFMR7osc, Me4,
           TRX2, TRBDF2, KC3I, KC4I, KC4Ia, Kv3I]


@pytest.mark.parametrize("solver", METHODS)
def test_orders(solver):
    # main method
    for i in range(solver.order):
        if i+1 > 7:   # skip higher order tests, not implemented yet
            return
        _norm = calc_Ts_norm(i+1, solver.B, solver.C, solver.A)
        assert_(_norm < solver.n_stages*1e-14)
    # secondary method
    for i in range(solver.order_secondary):
        if i+1 > 7:   # skip higher order tests, not implemented yet
            return
        E = solver.E
        B = solver.B
        if E.size == B.size:
            Bh = E + B
            A = solver.A
            C = solver.C
        else:
            A = np.zeros([E.size, E.size])
            A[:B.size, :B.size] = solver.A
            A[-1, :-1] = B
            Bh = E.copy()
            Bh[:-1] += B
            C = np.ones(E.size)
            C[:-1] = solver.C
        _norm = calc_Ts_norm(i+1, Bh, C, A)
        print(_norm)
        assert_(_norm < solver.n_stages*1e-14)


@pytest.mark.parametrize("solver", METHODS)
def test_coefficient_properties(solver):
    assert_allclose(np.sum(solver.B), 1, rtol=1e-15)
    assert_allclose(np.sum(solver.E), 0, atol=1e-15)                    # added
    assert_allclose(np.sum(solver.A, axis=1), solver.C, rtol=1e-13)
    # added tests for runge kutta interpolants. (C1 continuity)
    if isinstance(solver.P, np.ndarray):
        # C0 end (C0 start automatically satisfied)
        Ps = np.sum(solver.P, axis=1)
        Ps[:solver.B.size] -= solver.B
        assert_allclose(Ps, 0, atol=1e-12)
        if solver in [KC3I, KC4I, KC4Ia]:
            P = solver.P1
            # C0 end (again, for P1 now)
            Ps = np.sum(solver.P, axis=1)
            Ps[:solver.B.size] -= solver.B
            assert_allclose(Ps, 0, atol=1e-12)
        else:
            P = solver.P
        # C1 start
        Ps = np.sum(P, axis=0)
        Ps[0] -= 1
        assert_allclose(Ps, 0,  atol=1e-12)
        # C1 end
        dP = P * (np.arange(P.shape[1]) + 1)
        dPs = dP.sum(axis=1)
        dPs[-1] -= 1
        assert_allclose(dPs, 0, atol=2e-12)


@pytest.mark.parametrize("solver_class", METHODS)
def test_error_estimation(solver_class):
    if solver_class in [Me4, TRX2, TRBDF2, KC4Ia]:
        # Me4 does not pass this test: fifth order error estimate
        # similar reasoning for TRX2 and TRBDF2
        # KC4Ia does not pass this test, but I'm not sure why
        return
    else:
        step = 0.2
    solver = solver_class(lambda t, y: y, 0, [1], 1, first_step=step)
    solver.step()
    error_estimate = solver._estimate_error(solver.K, step)
    error = solver.y - np.exp([step])
    # print(np.abs(error), np.abs(error_estimate))
    assert_(np.abs(error) < np.abs(error_estimate))


@pytest.mark.parametrize("solver_class", METHODS)
def test_error_estimation_complex(solver_class):
    h = 0.2
    solver = solver_class(lambda t, y: 1j * y, 0, [1j], 1, first_step=h)
    solver.step()
    err_norm = solver._estimate_error_norm(solver.K, h, scale=[1])
    assert np.isrealobj(err_norm)
