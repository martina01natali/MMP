# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:00:04 2020

@author: matias
"""

'''
To see the animation, go to Tools -> Preferences -> IPython Console -> Graphics
-> Graphics backend -> Automatic

I STILL DON'T SEE ANYTHING
'''

import numpy as np
import matplotlib.pyplot as plt  

# ---Modifica Federico Matias Melendi--

from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

# ---Fine modifica Federico Matias Melendi--

#definition of linear pieceweise function and its vectorization

def fun(x):
    return np.piecewise(x, [x < 0.4, 
                            (0.4<=x) & (x<= 0.5),
                            (0.5<x) & (x<= 0.6),
                            x> 0.6], 
                        [0, x-0.4,-(x -0.6),0])

vfun=np.vectorize(fun)
grid=np.linspace(0,1,101)
fgrid=vfun(grid)

plt.plot(grid,fgrid)
plt.show()

#recall that in Python arrays and matrixes contain the element 0

#definition of the integrand for the projection of f(x) on the elements of the basis
from scipy.integrate import quad

def fun1(x,n):
    return 2*fun(x)*np.sin(n*np.pi*x)

def b(n):
    return quad(fun1,0,1,args=(n))[0]

tb=np.empty(201)
for i in range(201):
    tb[i]=b(i)

c=522
tf=1/261.

def sol(x,t,nmax):
    sol=0
    for n in range(nmax+1):
        sol=sol+tb[n]*np.cos(c*n*np.pi*t)*np.sin(n*np.pi*x)
    return sol


# ---Modifica Federico Matias Melendi--

fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(-0.1, 0.1))
line, = ax.plot([], [], lw=3)

# ---Fine modifica Federico Matias Melendi--

vsol=np.vectorize(sol)

#psol8=vsol(grid,tf/28,101)
#psol38=vsol(grid,tf*3./8,101)
#psol2=vsol(grid,tf/2,101)
#psol1=vsol(grid,tf,101)

#plt.plot(grid,psol8)
#plt.plot(grid,psol38)
#plt.plot(grid,psol2)
#plt.plot(grid,psol1)
#plt.show()

# ---Modifica Federico Matias Melendi--

def init():
    line.set_data([], [])
    return line,

def animate(t):
    x=grid
    y=vsol(x,(tf/200*t),101)
    line.set_data(x,y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                             frames=2000, interval=20, blit=True)




