# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:47:23 2021

@author: MARTINA
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:56:00 2020

@author: Drago
"""


import numpy as np
import matplotlib.pyplot as plt  

# initial condition

def psi(x):
    return 5-np.abs(x-25)/5

vpsi=np.vectorize(psi)
t = np.linspace(0, 50, 101)
ppsi=vpsi(t)
plt.plot(t,ppsi)

#recall that in Python arrays and matrixes contain the element 0

#elements of the basis
def phi(x,n):
    return np.sin(n*np.pi*x/50)

def psi1(x,n):
    return psi(x)*phi(x,n)

# initial condition projecting psi(x) on the elements of the basis
from scipy.integrate import quad
nmax=100

b=np.empty(nmax)
for n in range(nmax):   
    b[n]=(2/50)*quad(psi1,0,50,args=(n))[0]
print(b)


# evaluation of the solution of the equation

c=0.437
r=7.88
k=0.836

def coeff(t,n):
    return np.exp(-k*n**2*np.pi**2*t/(50**2*r*c))

def sol(x,t,ntop):
    sol=0
    for i in range(ntop):
        sol = sol + b[i]*coeff(t,i)*phi(x,i)
    return sol


vsol=np.vectorize(sol)
s20=vsol(t,20,10)
s40=vsol(t,40,10)
s60=vsol(t,60,10)
plt.plot(t,s20)
plt.plot(t,s40)
plt.plot(t,s60) 
plt.show() 


s202=vsol(t,20,2)
s204=vsol(t,20,4)
s206=vsol(t,20,6)
s208=vsol(t,20,8)
plt.plot(t,s202)
plt.plot(t,s204)
plt.plot(t,s206)
plt.plot(t,s208)
plt.plot(t,s20)
plt.show()