# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:49:46 2020

@author: MARTINA
"""

# ivp_one.py
# Author: Alex Gezerlis
# Numerical Methods in Physics with Python (CUP, 2020)

#-------------DESCRIPTION-------------#
#Euler's and Runge-Kutta methods are used to calculate the
#value of an integral with a numerical approximation. 
#-------------------------------------#

from RungeKuttaMethods_class import *

def f(x,y): #define a custom function that is the integrand
#    return - (30/(1-x**2)) + ((2*x)/(1-x**2))*y - y**2
    return y/(x**2+1)
            
if __name__ == '__main__':  #execute (print) only if this is a main file and
                            #not a subroutine of something else

    a, b, n, yinit = 0.05, 0.49, 12, 19.53
    
    xs, ys = RungeKuttaMethods.euler(f,a,b,n,yinit)
    print(ys)
    
    xs, ys = RungeKuttaMethods.rk4(f,a,b,n,yinit)
    print(ys)