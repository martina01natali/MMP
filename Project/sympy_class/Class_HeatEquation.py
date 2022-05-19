# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:43:25 2022

@author: MARTINA
"""

class HeatEquation:
    
    # Default values
    PARAMS = {
        'l' : .0,
        'rho' : .0,
        'c' : .0,
        'k' : .0,
        }
    BC = {
        '0' : 0,
        'l' : 0,
        }
    RHS = '0'

    # Init
    def __init__(self,
                 parameters:dict=PARAMS,
                 type_bc:str='dir',
                 boundaries:dict = BC,
                 rhs:str = RHS,
                 ):
        
        self.__parameters = parameters
        self.type_bc = type_bc
        self.boundaries = {
            'type_bc' : type_bc,
            'values' : boundaries,
            }
        # initial should be passed as a function? as a sympy object?
        