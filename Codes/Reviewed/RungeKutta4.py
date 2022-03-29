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

import numpy as np

def f(x,y): #define a custom function that is the integrand
#    return - (30/(1-x**2)) + ((2*x)/(1-x**2))*y - y**2
    return y/(x**2+1)
#--------------------------------------

def euler(f,a,b,n,yinit): #define euler method
    h = (b-a)/(n-1)
    xs = a + np.arange(n)*h
  
    #arange is a np function that creates a row array 
    #with n values from 0 to n-1 (dimension is nx1)
    #Mind that it takes floats but often return errors of
    #evaluation (including, non including endpoint) due to finite precision
    
    #np.arange(n) gives a row with n entries:
    #[0 1 2 3 4 5 ... n-1]
    #then it is multiplied by h, that is, for a fixed n,
    #the lenght of every n-1 interval that divides b and a.
    #xs is a vector whose minimum value is a (a+0) and every
    #entry represents one marker for the integration (h, 2h, 3h,...)
 
    ys = np.zeros(n)    #array of same dimension of xs (nx1) filled with zeros
                
    y = yinit
    for j,x in enumerate(xs): #for each marker set by xs
        ys[j] = y   #setting initial value [j=0] of ys to yinit, then
                    #as y grows ys[j] moves with j
        y += h*f(x, y)  
        #y is equal to y plus the step, that is the value of
        #the function at the marker to which y and x correspond
        #(the minimum value in the interval of lenght h on the x axis)
    return xs, ys

#--------------------------------------
        
def rk4(f,a,b,n,yinit): #Runge-Kutta order 4
    h = (b-a)/(n-1)
    xs = a + np.arange(n)*h
    ys = np.zeros(n)

    y = yinit
    for j,x in enumerate(xs):
        ys[j] = y
        k0 = h*f(x, y)
        k1 = h*f(x+h/2, y+k0*h/2)
        k2 = h*f(x+h/2, y+k1*h/2)
        k3 = h*f(x+h, y+k2*h)
        y += (k0 + 2*k1 + 2*k2 + k3)/6
    return xs, ys
        
if __name__ == '__main__':  #execute (print) only if this is a main file and
                            #not a subroutine of something else

#In the end, and I don't need to do it before, I must
#nevertheless define the values of all those variables
#I almost forgot about.
    a, b, n, yinit = 0.05, 0.49, 12, 19.53
    xs, ys = euler(f,a,b,n,yinit); print(ys)
    xs, ys = rk4(f,a,b,n,yinit); print(ys)
    