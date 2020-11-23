"""Test from scipy to conform to scipy. modified"""
import pytest
from numpy.testing import assert_allclose, assert_
import numpy as np
from extensisq import *


METHODS = [BS45, BS45_i, Ts45, CK45, CK45_o]


@pytest.mark.parametrize("solver", METHODS)
def test_coefficient_properties(solver):
    assert_allclose(np.sum(solver.B), 1, rtol=1e-15)
    assert_allclose(np.sum(solver.E), 0, atol=1e-15)                    # added
    assert_allclose(np.sum(solver.A, axis=1), solver.C, rtol=1e-14)
    # added tests for runge kutta interpolants. (C1 continuity)
    if solver is BS45:
        # BS45 uses extra stages I don't know how to test for C1 continuity
        # in that case. 
        return
    if solver is BS45_i:
        P = solver.Pfree
    else:
        P = solver.P
    Ps = np.sum(P, axis=0)
    Ps[0] -= 1
    assert_allclose(Ps, 0,  atol=1e-13)         # C1 start
    Ps = np.sum(P, axis=1)
    Ps[:solver.B.size] -= solver.B
    assert_allclose(Ps, 0, atol=1e-13)          # C0 end
    dP = P  * (np.arange(P.shape[1]) + 1)
    dPs = dP.sum(axis=1)
    dPs[-1] -= 1
    assert_allclose(dPs, 0, atol=1e-13)         # C1 end
    # C0 start is always satisfied


@pytest.mark.parametrize("solver_class", METHODS)
def test_error_estimation(solver_class):
    step = 0.2
    solver = solver_class(lambda t, y: y, 0, [1], 1, first_step=step)
    solver.step()
    error_estimate = solver._estimate_error(solver.K, step)
    error = solver.y - np.exp([step])
    assert_(np.abs(error) < np.abs(error_estimate))


@pytest.mark.parametrize("solver_class", METHODS)
def test_error_estimation_complex(solver_class):
    h = 0.2
    solver = solver_class(lambda t, y: 1j * y, 0, [1j], 1, first_step=h)
    solver.step()
    err_norm = solver._estimate_error_norm(solver.K, h, scale=[1])
    assert np.isrealobj(err_norm)
