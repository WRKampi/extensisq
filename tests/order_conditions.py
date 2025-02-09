# import sympy
from math import factorial
import numpy as np


def calc_egps(order, c, A, Ap):
    """return w, w1, eta, gamma, phik
    Ap = A
    A = alpha
    """
    # c = sympy.Matrix(c)
    # A = sympy.Matrix(A)
    # Ap = sympy.Matrix(Ap)
    c = np.atleast_2d(c).T
    C = np.diag(c[:, 0])
    e = np.ones([len(c), 1])
    dat = {
        0 : None,
        1 : ((1, 2, 1, 1, e),),
        2 : ((2, 3, 1, 2, c),),
        3 : ((7, 6, 1, 3, C@c),
             (2, 5, 1, 6, Ap@c)),
        4 : ((38, 20, 1, 4, C@C@c),
             (4, 6, 3, 8, C@Ap@c),
             (2, 0, 1, 24, A@c),
             (7, 12, 1, 12, Ap@C@c),
             (2, 8, 1, 24, Ap@Ap@c)),
        5 : ((295, 70, 1, 5, C@C@C@c),
             (14, 18, 6, 10, C@C@Ap@c),
             (4, 0, 4, 30, C@A@c),
             (14, 12, 4, 15, C@Ap@C@c),
             (4, 10, 4, 30, C@Ap@Ap@c),
             (7, 0, 1, 60, A@C@c),
             (2, 0, 1, 120, A@Ap@c),
             (6, 15, 3, 20, np.diag((Ap@c)[:, 0])@(Ap@c)),
             (38, 40, 1, 20, Ap@C@C@c),
             (4, 12, 3, 40, Ap@C@Ap@c),
             (2, 0, 1, 120, Ap@A@c),
             (7, 18, 1, 60, Ap@Ap@C@c),
             (2, 13, 1, 120, Ap@Ap@Ap@c)),
        6 : ((2702, 251, 1, 6, C@C@C@C@c),
             (76, 60, 10, 12, C@C@C@Ap@c),
             (14, 0, 10, 36, C@C@A@c),
             (49, 36, 10, 18, C@C@Ap@C@c),
             (14, 30, 10, 36, C@C@Ap@Ap@c),
             (14, 0, 5, 72, C@A@C@c),
             (4, 0, 5, 144, C@A@Ap@c),
             (12, 30, 15, 24, C@np.diag((Ap@c)[:, 0])@(Ap@c)),
             (4, 0, 10, 72, np.diag((A@c)[:, 0])@(Ap@c)),
             (76, 40, 5, 24, C@Ap@C@C@c),
             (8, 12, 15, 48, C@Ap@C@Ap@c),
             (4, 0, 5, 144, C@Ap@A@c),
             (14, 24, 5, 72, C@Ap@Ap@C@c),
             (4, 16, 5, 144, C@Ap@Ap@Ap@c),
             (38, 0, 1, 120, A@C@C@c),
             (4, 0, 3, 240, A@C@Ap@c),
             (2, 0, 1, 720, A@A@c),
             (7, 0, 1, 360, A@Ap@C@c),
             (2, 0, 1, 720, A@Ap@Ap@c),
             (14, 18, 10, 36, np.diag((Ap@C@c)[:, 0])@(Ap@c)),
             (4, 15, 10, 72, np.diag((Ap@c)[:, 0])@(Ap@Ap@c)),
             (295, 140, 1, 30, Ap@C@C@C@c),    # **4 missing in paper Fine
             (14, 36, 6, 60, Ap@C@C@Ap@c),
             (4, 0, 4, 180, Ap@C@A@c),
             (14, 24, 8, 90, Ap@C@Ap@C@c),
             (4, 20, 4, 180, Ap@C@Ap@Ap@c),
             (7, 0, 1, 360, Ap@A@C@c),
             (2, 0, 1, 720, Ap@A@Ap@c),
             (6, 30, 3, 120, Ap@np.diag((Ap@c)[:, 0])@(Ap@c)),
             (38, 60, 1, 120, Ap@Ap@C@C@c),
             (4, 18, 3, 240, Ap@Ap@C@Ap@c),
             (2, 0, 1, 720, Ap@Ap@A@c),
             (7, 30, 1, 360, Ap@Ap@Ap@C@c),
             (2, 31, 1, 720, Ap@Ap@Ap@Ap@c)),
        7 : (# From Fehlberg
             #1:
             (None, None, 1, 7, C@C@C@C@C@c),
             (None, None, 15, 14, C@C@C@C@Ap@c),
             (None, 0, 20, 42, C@C@C@A@c),
             (None, None, 20, 21, C@C@C@Ap@C@c),
             (None, 0, 15, 84, C@C@A@C@c),
             (None, None, 15, 28, C@C@Ap@C@C@c),
             (None, None, 45, 28, C@C@np.diag((Ap@c)[:, 0])@(Ap@c)),
             #8:
             (None, 0, 6, 140, C@A@C@C@c),
             (None, 0, 60, 84, C@np.diag((A@c)[:, 0])@(Ap@c)),
             (None, None, 6, 35, C@Ap@C@C@C@c),
             (None, None, 60, 42, C@np.diag((Ap@C@c)[:, 0])@(Ap@c)),
             (None, 0, 1, 210, A@C@C@C@c),
             (None, 0, 10, 252, np.diag((A@c)[:, 0])@(A@c)),
             (None, 0, 15, 168, np.diag((A@C@c)[:, 0])@(Ap@c)),
             (None, 0, 20, 126, np.diag((A@c)[:, 0])@(Ap@C@c)),
             (None, None, 1, 42, Ap@C@C@C@C@c),
             (None, None, 15, 56, np.diag((Ap@C@C@c)[:, 0])@(Ap@c)),
             (None, None, 10, 63, np.diag((Ap@C@c)[:, 0])@(Ap@C@c)),
             (None, None, 15, 56, np.diag((Ap@c)[:, 0])@np.diag((Ap@c)[:, 0])@(Ap@c)),
             (None, None, 20, 42, C@C@C@Ap@Ap@c),
             (None, 0, 15, 168, C@C@A@Ap@c),
             #22:
             (None, None, 45, 56, C@C@Ap@C@Ap@c),
             (None, 0, 15, 168, C@C@Ap@A@c),
             (None, None, 15, 84, C@C@Ap@Ap@C@c),
             (None, 0, 18, 280, C@A@C@Ap@c),
             (None, 0, 6, 840, C@A@A@c),
             (None, 0, 6, 420, C@A@Ap@C@c),
             (None, None, 36, 70, C@Ap@C@C@Ap@c),
             (None, 0, 24, 210, C@Ap@C@A@c),
             (None, None, 24, 105, C@Ap@C@Ap@C@c),
             (None, 0, 6, 420, C@Ap@A@C@c),
             (None, None, 6, 140, C@Ap@Ap@C@C@c),
             (None, None, 18, 140, C@Ap@np.diag((Ap@c)[:, 0])@(Ap@c)),
             (None, None, 60, 84, C@np.diag((Ap@c)[:, 0])@(Ap@Ap@c)),
             (None, 0, 6, 420, A@C@C@Ap@c),
             #36:
             (None, 0, 4, 1260, A@C@A@c),
             (None, 0, 4, 630, A@C@Ap@C@c),
             (None, 0, 1, 2520, A@A@C@c),
             (None, 0, 1, 840, A@Ap@C@C@c),
             (None, 0, 3, 840, A@np.diag((Ap@c)[:, 0])@(Ap@c)),
             (None, None, 10, 84, Ap@C@C@C@Ap@c),
             (None, 0, 10, 252, Ap@C@C@A@c),
             (None, None, 10, 126, Ap@C@C@Ap@C@c),
             (None, 0, 5, 504, Ap@C@A@C@c),
             (None, None, 5, 168, Ap@C@Ap@C@C@c),
             (None, None, 15, 168, Ap@C@np.diag((Ap@c)[:, 0])@(Ap@c)),
             (None, 0, 1, 840, Ap@A@C@C@c),
             (None, 0, 10, 504, Ap@np.diag((A@c)[:, 0])@(Ap@c)),
             (None, None, 1, 210, Ap@Ap@C@C@C@c),
             (None, None, 10, 252, Ap@np.diag((Ap@C@c)[:, 0])@(Ap@c)),
             #51:
             (None, None, 10, 252, np.diag((Ap@Ap@c)[:, 0])@(Ap@Ap@c)),
             (None, 0, 20, 252, np.diag((A@c)[:, 0])@(Ap@Ap@c)),
             (None, None, 20, 126, np.diag((Ap@C@c)[:, 0])@(Ap@Ap@c)),
             (None, 0, 15, 336, np.diag((Ap@c)[:, 0])@(A@Ap@c)),
             (None, None, 45, 112, np.diag((Ap@c)[:, 0])@(Ap@C@Ap@c)),
             (None, 0, 15, 336, np.diag((Ap@c)[:, 0])@(Ap@A@c)),
             (None, None, 15, 168, np.diag((Ap@c)[:, 0])@(Ap@Ap@C@c)),
             (None, None, 15, 168, C@C@Ap@Ap@Ap@c),
             (None, 0, 6, 840, C@A@Ap@Ap@c),
             (None, None, 24, 210, C@Ap@C@Ap@Ap@c),
             (None, 0, 6, 840, C@Ap@A@Ap@c),
             (None, None, 18, 280, C@Ap@Ap@C@Ap@c),
             (None, 0, 6, 840, C@Ap@Ap@A@c),
             (None, None, 6, 420, C@Ap@Ap@Ap@C@c),
             (None, 0, 4, 1260, A@C@Ap@Ap@c),
             #66:
             (None, 0, 1, 5040, A@A@Ap@c),
             (None, 0, 3, 1680, A@Ap@C@Ap@c),
             (None, 0, 1, 5040, A@Ap@A@c),
             (None, 0, 1, 2520, A@Ap@Ap@C@c),
             (None, None, 10, 252, Ap@C@C@Ap@Ap@c),
             (None, 0, 5, 1008, Ap@C@A@Ap@c),
             (None, None, 15, 336, Ap@C@Ap@C@Ap@c),
             (None, 0, 5, 1008, Ap@C@Ap@A@c),
             (None, None, 5, 504, Ap@C@Ap@Ap@C@c),
             (None, 0, 3, 1680, Ap@A@C@Ap@c),
             (None, 0, 1, 5040, Ap@A@A@c),
             (None, 0, 1, 2520, Ap@A@Ap@C@c),
             (None, None, 6, 420, Ap@Ap@C@C@Ap@c),
             (None, 0, 4, 1260, Ap@Ap@C@A@c),
             (None, None, 4, 630, Ap@Ap@C@Ap@C@c),
             #81:
             (None, 0, 1, 2520, Ap@Ap@A@C@c),
             (None, None, 1, 840, Ap@Ap@Ap@C@C@c),
             (None, None, 3, 840, Ap@Ap@np.diag((Ap@c)[:, 0])@(Ap@c)),
             (None, None, 10, 504, Ap@np.diag((Ap@c)[:, 0])@(Ap@Ap@c)),
             (None, None, 15, 336, np.diag((Ap@c)[:, 0])@(Ap@Ap@Ap@c)),
             (None, None, 6, 840, C@Ap@Ap@Ap@Ap@c),
             (None, 0, 1, 5040, A@Ap@Ap@Ap@c),
             (None, None, 5, 1008, Ap@C@Ap@Ap@Ap@c),
             (None, 0, 1, 5040, Ap@A@Ap@Ap@c),
             (None, None, 4, 1260, Ap@Ap@C@Ap@Ap@c),
             (None, 0, 1, 5040, Ap@Ap@A@Ap@c),
             (None, None, 3, 1680, Ap@Ap@Ap@C@Ap@c),
             (None, 0, 1, 5040, Ap@Ap@Ap@A@c),
             (None, None, 1, 2520, Ap@Ap@Ap@Ap@C@c),
             (None, None, 1, 5040, Ap@Ap@Ap@Ap@Ap@c))
        }
    return dat[order]
    # return [sympy.simplify(T) for T in dat[order]]


def calc_Ts_norm(order, b, c, A, beta=None, alpha=None, t=1):
    """alpha and beta are the extra matrices for RKN methods"""
    if beta is None:
        assert alpha is None, "need neither or both beta and alpha"
    if alpha is None:
        assert beta is None, "need neither or both beta and alpha"
        # RK method, not RKN
        Tp = calc_Ts(order, b, c, A, beta, alpha, t)
        Tp_norm = np.sqrt(np.sum(np.asarray(Tp)**2))
        return Tp_norm

    # General RKN method, need to add strict RKN method later..................
    T, Tp = calc_Ts(order, b, c, A, beta=beta, alpha=alpha, t=t)
    T_norm = np.sqrt(np.sum(np.asarray(T)**2))
    Tp_norm = np.sqrt(np.sum(np.asarray(Tp)**2))
    return T_norm, Tp_norm


def calc_Ts(order, b, c, A, beta=None, alpha=None, t=1):
    b = np.atleast_2d(b).T
    if beta is None:
        assert alpha is None, "need neither or both beta and alpha"
    if alpha is None:
        assert beta is None, "need neither or both beta and alpha"

        # RK method, not RKN
        egps = calc_egps(order, c, A, A)
        Tp = []
        for w, w1, eta, gamma, phik in egps:
            if w1 == 0:
                continue
            phip = (b.T@phik)[0, 0]
            Tp.append(eta/factorial(order) * (gamma*phip - t**order))
        return Tp

    # General RKN method, need to add strict RKN method later..................
    beta = np.atleast_2d(beta).T
    egps = calc_egps(order, c, alpha, A)
    T = []
    Tp = []
    for w, w1, eta, gamma, phik in egps:
        phi = (beta.T@phik)[0, 0]
        phip = (b.T@phik)[0, 0]
        T.append(eta/factorial(order + 1) * ((order + 1)*gamma*phi - t**(order + 1)))
        Tp.append(eta/factorial(order) * (gamma*phip - t**order))
    return T, Tp



if __name__ == "__main__":
    from extensisq import Ts5, Fi5N
    from math import isclose
    
    # print(calc_Ts_norm(5, Ts5.B, Ts5.C, Ts5.A))
    # print(calc_Ts(5, Ts5.B, Ts5.C, Ts5.A))
    
    # print(calc_Ts_norm(4, Fi5N.Bp, Fi5N.C, Fi5N.Ap, alpha=Fi5N.A, beta=Fi5N.B))
    # print(calc_Ts(1, Fi5N.Bp, Fi5N.C, Fi5N.Ap, alpha=Fi5N.A, beta=Fi5N.B))
    
    # A = np.random.randn(7, 7)
    # alpha = np.random.rand(7, 7)
    # B = np.random.randn(7)
    # beta = np.random.rand(7)
    # C = np.random.randn(7)

    # T = calc_Ts(7, B, C, A)
    # print(len(T))
    # T, Tp = calc_Ts(7, B, C, A, alpha=alpha, beta=beta)
    # print(len(T))

    # for i in range(len(Tp)):
    #     for j in range(i):
    #         if isclose(Tp[i], Tp[j]):
    #             print(i, j, Tp[i], Tp[j])