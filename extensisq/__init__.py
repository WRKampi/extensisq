"""This package extends scipy.integrate with various methods
(OdeSolver classes) for the solve_ivp function.
"""
from extensisq.common import NFS, NFI, NLS
from extensisq.tsitouras import Ts5
from extensisq.bogacki import BS5
from extensisq.cash import CK5, CKdisc
from extensisq.prince import Pr7, Pr8, Pr9
from extensisq.shampine import SWAG
from extensisq.calvo import CFMR7osc
from extensisq.sommeijer import SSV2stab
from extensisq.merson import Me4
from extensisq.fine import Fi4N, Fi5N
from extensisq.murua import Mu5Nmb
from extensisq.mikkawy import MR6NN
from extensisq.hosea import HS2I, HS2Ia, TRBDF2, TRX2
from extensisq.kennedy import KC3I, KC4I, KC4Ia
from extensisq.kvaerno import Kv3I
from extensisq.sensitivity import (sens_forward, sens_adjoint_end,
                                   sens_adjoint_int)

__version__ = '0.6.0'
__author__ = 'W.R. Kampinga'
__copyright__ = 'Copyright 2024, W.R. Kampinga'
__license__ = 'MIT'
__credits__ = (
    'scipy', 'L.F Shampine', 'P. Bogacki', 'R.W. Brankin', 'I. Gladwell',
    'J.R. Cash', 'A.H. Karp', 'Ch. Tsitouras', 'P.J. Prince', 'H.A. Watts',
    'M.K. Gordon', 'G. Soederlind', 'K. Gustafsson', 'M. Calvo', 'J.M. Franco',
    'J.I. Montijano', 'L. Randez', 'B.P. Sommeijer', 'J.G. Verwer',
    'E. Hairer', 'A.C. Hindmarsh', 'R. Serban', 'R.H. Merson', 'J.M. Fine',
    'A. Murua', 'M. El-Mikkawy', 'E.D. Rahmo', 'M.E. Hosea', 'C.A. Kennedy',
    'M.H. Carpenter', 'A. Kvaerno')
