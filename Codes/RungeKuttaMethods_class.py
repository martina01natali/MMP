# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:07:32 2022

@author: MARTINA
"""

import numpy as np

class RungeKuttaMethods():
    """Class that groups methods to perform numerical integration"""
    
    @staticmethod
    def euler(f,a,b,n,yinit):
        """Euler method"""        
        h = (b-a)/(n-1)
        xs = a + np.arange(n)*h
      
        #arange is a np function that creates a row array 
        #with n values from 0 to n-1 (dimension is nx1)
        #Mind that it takes floats but often return errors of
        #evaluation (including, non including endpoint) due to finite precision
        
        #np.arange(n) gives a row with n entries:
        #[0 1 2 3 4 5 ... n-1]
        #then it is multiplied by h, that is, for a fixed n,
        #the lenght of every n-1 interval that divides band a.
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
    
    # @staticmethod
    # def rk2(f,a,b,n,yinit):
    #    h = (b-a)/(n-1)
    #    xs = a + np.arange(n)*h  
    #    ys = np.zeros(n)
    #    y = yinit
    #    for j,x in enumerate(xs):
    #        ys[j] = y
    #        y += h*f(x, y)  
    #    return xs, ys

    @staticmethod
    def rk4(f,a,b,n,yinit):
        """Runge-Kutta method, order 4"""        
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