"""This package extends scipy.integrate with various methods
(OdeSolver classes) for the solve_ivp function.
"""
from extensisq.common import NFS
from extensisq.tsitouras import Ts5, Ts45
from extensisq.bogacki import BS5, BS45, BS45_i
from extensisq.cash import CK5, CKdisc, CK45, CK45_o
from extensisq.prince import Pr7, Pr8, Pr9, Pri6, Pri7, Pri8
from extensisq.shampine import SWAG
from extensisq.calvo import CFMR7osc
from extensisq.sommeijer import SSV2stab

__version__ = '0.3.0'
__author__ = 'W.R. Kampinga'
__copyright__ = 'Copyright 2020, W.R. Kampinga'
__license__ = 'MIT'
__credits__ = (
    'scipy', 'L.F Shampine', 'P. Bogacki', 'R.W. Brankin', 'I. Gladwell',
    'J.R. Cash', 'A.H. Karp', 'Ch. Tsitouras', 'P.J. Prince', 'H.A. Watts',
    'M.K. Gordon', 'G. Soederlind', 'K. Gustafsson', 'M. Calvo', 'J.M. Franco',
    'J.I. Montijano', 'L. Randez', 'B.P. Sommeijer', 'J.G. Verwer')
