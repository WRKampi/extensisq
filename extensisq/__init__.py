"""This package extends scipy.integrate with OdeSolver objects for 
the solve_ivp function.
"""
from extensisq.tsitouras import Ts45
from extensisq.bogacki import BS45, BS45_i
from extensisq.cash import CK45, CK45_o

__version__ = '0.0.3'
__author__ = 'W.R. Kampinga'
__copyright__ = 'Copyright 2020, W.R. Kampinga'
__license__ = 'MIT'
__credits__ = ('scipy', 'P. Bogacki', 'L.F Shampine', 'J.R. Cash', 'A.H. Karp', 
                'Ch. Tsitouras')
