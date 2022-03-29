# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt  

#Drago says that numpy routines are partially compiled (space alocated for
#each variable, array is already defined and the cpu doesn't have to compute it
#every single time)

# initial condition at time = 0; blue curve (angular function)

def psi(x):
    return 5-np.abs(x-25)/5

vpsi=np.vectorize(psi) #vectorization allows to not use loops and it is faster
t = np.linspace(0, 50, 101)
ppsi=vpsi(t)
plt.plot(t,ppsi)

#recall that in Python arrays and matrixes contain the element 0

#elements of the basis
def phi(x,n):
    return np.sin(n*np.pi*x/50)

#recall that the pi constant is already in numpy and is called as np.pi

def psi1(x,n):
    return psi(x)*phi(x,n)

# initial condition projecting psi(x) on the elements of the basis
from scipy.integrate import quad
nmax=100

#quad is the simple routine to compute an integral
#we are integrating numerically instead of analytically, that we don't know
#how to do in python yet

b=np.empty(nmax)
for n in range(nmax):   
    b[n]=(2/50)*quad(psi1,0,50,args=(n))[0]
print(b)

#the entries of the matrix b are the projections of the function on the elements
#of the basis
#The columns 0 and 2 are almost 0 because the function is symmetrical with
#respect to 25 and is integrated bet 0 and 50 


#----------Iron bar, heat diffusion----------#
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

#the solutions are plotted at different times
#almost immediatly after t=0, the irregularity disappears
#the reason is that the coefficients with very large n are immediatly suppressed
#by the exponential suppression that goes as -n^2

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