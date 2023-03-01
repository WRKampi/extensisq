"""Test sensitivity methods"""
import pytest
from numpy.testing import assert_allclose
import numpy as np
from extensisq import sens_forward, sens_adjoint_int, sens_adjoint_end


METHODS = ["LSODA", "BDF", "Radau"]


def fun(t, y, *p):
    y1, y2, y3 = y
    p1, p2, p3 = p
    return np.array([-p1*y1 + p2*y2*y3,
                     p1*y1 - p2*y2*y3 - p3*y2**2,
                     p3*y2**2])


def jac(t, y, *p):
    y1, y2, y3 = y
    p1, p2, p3 = p
    return np.array([[-p1, p2*y3, p2*y2],
                     [p1, -p2*y3 - 2*p3*y2, -p2*y2],
                     [0., 2*p3*y2, 0.]])


def dfdp(t, y, *p):
    y1, y2, y3 = y
    p1, p2, p3 = p
    return np.array([[-y1, y2*y3, 0.],
                     [y1, -y2*y3, -y2**2],
                     [0., 0., y2**2]])


def g(t, y, *p):
    y1, y2, y3 = y
    p1, p2, p3 = p
    return [y1 + p2*y2*y3]


def dgdy(t, y, *p):
    y1, y2, y3 = y
    p1, p2, p3 = p
    return np.array([1., p2*y3, p2*y2])


def dgdp(t, y, *p):
    y1, y2, y3 = y
    p1, p2, p3 = p
    return np.array([0., y2*y3, 0.])


y0 = np.array([1., 0., 0.])
p = (0.04, 1e4, 3e7)
dy0dp = np.zeros([3, 3])
rtol = 1e-4
atol = np.array([1e-8, 1e-14, 1e-6])
atol_adj = 1e-5
atol_quad = 1e-6

result_forward = {
    'yf': [9.8517e-01, 3.3864e-05, 1.4794e-02],
    'sens': [[-3.5595e-01,  9.5428e-08, -1.5832e-11],
             [3.9026e-04, -2.1310e-10, -5.2900e-13],
             [3.5556e-01, -9.5215e-08, 1.6361e-11]]}
result_adjoint_int = {
    'yf': [5.2016e-05, 2.0808e-10, 9.9995e-01],
    'sens': [-7.8383e+05, 3.1991, -5.3301e-04],
    'G': 1.8219e+04,
    'lambda0': [3.4249e+04, 3.4206e+04, 3.4139e+04]}


@pytest.mark.parametrize('method', METHODS)
def test_sens_forward(method):
    t_span = (0., 0.4)
    use_approx_jac = method == "LSODA"

    sens, yf, _ = sens_forward(
        fun, t_span, y0, jac, dfdp, dy0dp, p=p, method=method,
        rtol=rtol, atol=atol, use_approx_jac=use_approx_jac)

    assert_allclose(yf, result_forward['yf'], rtol=1e-3)
    assert_allclose(sens, result_forward['sens'], rtol=1e-3)


@pytest.mark.parametrize('method', METHODS)
def test_sens_adjoint_int(method):
    t_span = (0., 4e7)

    sens, G, sol_y, sol_bw = sens_adjoint_int(
        fun, t_span, y0, jac, dfdp, dy0dp, p, g, dgdp, dgdy, method=method,
        atol=atol, rtol=rtol, atol_quad=atol_quad, atol_adj=atol_adj)
    yf = sol_y.y[:, -1]
    lambda0 = sol_bw.y[:3, -1]

    assert_allclose(yf, result_adjoint_int['yf'], rtol=1e-2)
    assert_allclose(sens, result_adjoint_int['sens'], rtol=1e-2)
    assert_allclose([G], [result_adjoint_int['G']], rtol=1e-2)
    assert_allclose(lambda0, result_adjoint_int['lambda0'], rtol=1e-2)


@pytest.mark.parametrize('method', METHODS)
def test_sens_adjoint_end(method):
    t_span = (0., 0.4)

    for i in range(3):
        def g(t, y, *p, i=i):
            return [y[i]]

        def dgdy(t, y, *p, i=i):
            a = np.zeros(3)
            a[i] = 1.
            return a

        def dgdp(t, y, *p):
            return np.zeros(3)

        sens, gf, _, _ = sens_adjoint_end(
            fun, t_span, y0, jac, dfdp, dy0dp, p, g, dgdp, dgdy, method=method,
            atol=atol, rtol=rtol, atol_quad=atol_quad/10, atol_adj=atol_adj/10)

        assert_allclose(gf, [result_forward['yf'][i]], rtol=1e-3)
        assert_allclose(sens, result_forward['sens'][i], rtol=1e-2)
