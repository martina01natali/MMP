# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:10:27 2020

@author: Drago
"""
import numpy as np
import matplotlib.pyplot as plt  
from scipy.linalg import solve_banded 

from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

# parameters ###########################################


c=522
t0=0 
tf=1/(c/2) 
ntimestep=100
dt=tf/ntimestep
n=20
h=1/n 

# piecewise function ###################################

def fun(x,i):
    return np.piecewise(x, [x < (i-1)*h, 
                            ((i-1)*h<=x) & (x<= i*h),
                            (i*h<x) & (x<= (i+1)*h),
                            x>(i+1)*h], 
                        [0, (x - (i - 1)* h)/h,-(x - (i + 1)* h)/h,0])

# matrices M and K both in normal form and in band form ################

maindiagm=np.empty(n-1)
offdiagm=np.empty(n-2)
for i in range(n-1):
    maindiagm[i]=2*h/3
for i in range(n-2):
    offdiagm[i]=h/6
    
#m=np.diag(offdiagm, -1) + np.diag(maindiagm) + np.diag(offdiagm, 1)
    
mb=np.zeros((3,n-1))
for i in range(n-2):
    mb[0,i+1]=offdiagm[i]
    mb[2,i]=offdiagm[i]
for i in range(n-1):
    mb[1,i]=maindiagm[i]
    
    
maindiagk=np.empty(n-1)
offdiagk=np.empty(n-2)
for i in range(n-1):
    maindiagk[i]=2/h
for i in range(n-2):
    offdiagk[i]=-1/h
        
k=np.diag(offdiagk, -1) + np.diag(maindiagk) + np.diag(offdiagk, 1) 
k=k*c**2

kb=np.zeros((3,n-1))
for i in range(n-2):
    kb[0,i+1]=offdiagk[i]
    kb[2,i]=offdiagk[i]
for i in range(n-1):
    kb[1,i]=maindiagk[i]
    
kb=kb*c**2

#   initial condition ######################

def psi(x):
    return 0.01*x*(1-x)

#   initial valus on the grid ##########################

y0=np.empty(n-1)
for i in range(n-1):
    y0[i]=psi((i+1)*h)
    
#print("y0=",y0)

def vib(d):
    vib=0
    for i in range(n-1):
        vib=vib+y0[i]*fun(d,i+1)
    return vib

vvib=np.vectorize(vib)


# check that the initial value "init" and the approximation of it obtained
# by using vib with y0 "sol0" are very similar

grid=np.linspace(0,1,101)

vpsi=np.vectorize(psi)
init=vpsi(grid)
sol0=vvib(grid)
plt.plot(grid,init)
plt.plot(grid,sol0) 
plt.show() 


def soluzione(nfinale):
    y0=np.empty(n-1)
    for i in range(n-1):
        y0[i]=psi((i+1)*h)
    z0=np.zeros(n-1)

    y=np.copy(y0)
    z=np.copy(z0)

    for i in range(nfinale):
        s1=solve_banded((1,1),mb, -np.dot(k,y))
        k1y=z
        k1z=s1
        s2=solve_banded((1,1),mb, -np.dot(k,y+(1/2)*dt*k1y))
        k2y=z+(1/2)*dt*k1z
        k2z=s2
        s3=solve_banded((1,1),mb, -np.dot(k,y+(1/2)*dt*k2y))
        k3y=z+(1/2)*dt*k2z
        k3z=s3
        s4=solve_banded((1,1),mb, -np.dot(k,y+dt*k3y))
        k4y=z+dt*k3z
        k4z=s4
        y0=np.copy(y)
        z0=np.copy(z)
        y=y0+(k1y+2*k2y+2*k3y+k4y)*dt/6
        z=z0+(k1z+2*k2z+2*k3z+k4z)*dt/6
    
    def vib(d):
        vib=0
        for i in range(n-1):
            vib=vib+y[i]*fun(d,i+1)
        return vib

    vvib=np.vectorize(vib)

    soluzione=vvib(grid)
    return soluzione

fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(-0.005, 0.005))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(nf):
    x=grid
    y=soluzione(nf)
    line.set_data(x,y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                             frames=2000, interval=20, blit=True)

