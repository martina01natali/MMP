# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:08:07 2020

@author: MARTINA
"""

#Basics of plotting

from numpy import linspace #generate a finite number of equally distributed
                            #numbers
import matplotlib.pyplot as plt

v0 = 5
g = 9.81
t = linspace(0, 1, num=1001) #array with 1001 numbers

y = v0*t - 0.5*g*t**2

plt.plot(t, y)
plt.xlabel('t (s)')
plt.ylabel('y (m)')
plt.show()