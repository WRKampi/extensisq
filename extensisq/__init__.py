"""This package extends scipy.integrate with various methods
(OdeSolver classes) for the solve_ivp function.
"""
from extensisq.tsitouras import Ts45
from extensisq.bogacki import BS45, BS45_i
from extensisq.cash import CK45, CK45_o
from extensisq.prince import Pri6, Pri7, Pri8
from extensisq.shampine import SWAG

__version__ = '0.2.0'
__author__ = 'W.R. Kampinga'
__copyright__ = 'Copyright 2020, W.R. Kampinga'
__license__ = 'MIT'
__credits__ = ('scipy', 'P. Bogacki', 'L.F Shampine', 'J.R. Cash', 'A.H. Karp',
               'Ch. Tsitouras', 'P.J. Prince', 'H.A. Watts', 'M.K. Gordon')
