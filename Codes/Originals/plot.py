# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:40:58 2021

@author: MARTINA
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 01:02:07 2020

@author: Drago
"""


from numpy import linspace  
import matplotlib.pyplot as plt  

v0 = 5  
g = 9.81  
t = linspace(0, 1, 1001)  

y = v0*t - 0.5*g*t**2  

plt.plot(t, y)  
plt.xlabel('t (s)')  
plt.ylabel('y (m)')  
plt.show()