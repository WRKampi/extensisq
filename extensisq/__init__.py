"""This package extends scipy.integrate with various methods
(OdeSolver classes) for the solve_ivp function.
"""
from extensisq.tsitouras import Ts5, Ts45
from extensisq.bogacki import BS5, BS45, BS45_i
from extensisq.cash import CK5, CKdisc, CK45, CK45_o
from extensisq.prince import Pr7, Pr8, Pr9, Pri6, Pri7, Pri8
from extensisq.shampine import SWAG
from extensisq.common import NFS

__version__ = '0.2.1'
__author__ = 'W.R. Kampinga'
__copyright__ = 'Copyright 2020, W.R. Kampinga'
__license__ = 'MIT'
__credits__ = ('scipy', 'L.F Shampine', 'P. Bogacki', 'J.R. Cash', 'A.H. Karp',
               'Ch. Tsitouras', 'P.J. Prince', 'H.A. Watts', 'M.K. Gordon')
