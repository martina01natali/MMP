# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:43:48 2021

@author: Drago
"""
# Consider an elastic string that, when stretched by a tension of 10 N, has a
# length of 50 cm. Suppose that the density of the stretched string is of
# 0.2 g/cm. If the string is fixed horizontally and sags under gravity, what
# shape does it assume?

        #********************************************************#

# We have a string attached to two extremes, so the boundary conditions
# are Dirichlet.
# First we solve the ordinary diff. eq. analytically by using sympy.
# You have to import from symbolic python the names of the variables that
# you want to use.

# The equation is the time-independent wave equation applied to a string
# whose allowed dispacement is on the transversal direction only:
# -T d^2(u(x))/dx^2 = -pg
# where T is the constant tension on the longitudinal direction,
# u(x) is the solution of the ODE,
# p is the linear mass density of the string,
# g is the gravitational acceleration

import numpy as np
import matplotlib.pyplot as plt
from sympy import Function, Derivative, Eq, dsolve #Symbols

# x, t, r, g, l = Symbols('x t r g l')
from sympy.abc import x, t, r , g, l # alternative way

# The equations in sympy must be defined by Eq(lhs:rhs).
# We must define the nature of all the letters we use.
# Initial conditions are passed to the equation by ics.
# Derivative(f(x),x,x) = d^2/dx^2(f(x))

f = Function('f')
sol = dsolve(Eq(t*Derivative(f(x),x,x),r*g),f(x), ics={f(0):0, f(l):0})
print(sol)

#%%
# Define the actual solution of the ODE (is known analytically)
def fun(x):
    fun=-g*l*r*x/(2*t) + g*r*x**2/(2*t)
    return fun

g=980. # cm/s^2
r=0.2 # g/cm
l=50. # cm
t=10**6 # g cm/s^2 (u.m. dynes)

# Allow fun(x) to take a vector of values as argument by vectorizing it
vfun = np.vectorize(fun)

# Create x and evaluate y values = values of the function
x = np.linspace(0,l,200)
y = vfun(x)

#Plot
plt.axis([0,l,-0.1,0])
plt.plot(x,y,label='exact solution')
plt.legend()
plt.show()

#%%
from scipy.integrate import quad

# Define eigenfunctions of the kinetic differential operator
# That'll be the basis for our Fourier decomposition

def eigenf(x,n):
    return np.sin(n*np.pi*x/l)

def rhs(r):
    return -r*g

def func1(x,n):
    return 2/l*rhs(r)*eigenf(x,n)

# Compute coefficients of the decomposition of the function at the rhs
def fouriercn(n):
    return quad(func1,0,l,args=(n))[0]

# Compute coefficients of the decomposition of the solution
def fourieran(n):
    return fouriercn(n)*l**2/(t*n**2*np.pi**2)
# these coefficients scale as 1/n**3 (very fast mean convergence)

# If  needed one can compute those coeffs as elements of an array
vec_fouriercn = np.vectorize(fouriercn)
res = vec_fouriercn(np.arange(1,100,1))
print(res)

vec_fourieran = np.vectorize(fourieran)
res = vec_fourieran(np.arange(1,100,1))

#%%
# Here we approximate the rhs with a decomposition with 10 and 50 terms
def approxrhs10(x):
    approxrhs10=0
    for j in range(1,10):
        approxrhs10 += np.sin(j*np.pi*x/l)*fouriercn(j)
    return approxrhs10

def approxrhs50(x):
    approxrhs50=0
    for j in range(1,50):
        approxrhs50 += np.sin(j*np.pi*x/l)*fouriercn(j)
    return approxrhs50

# Here we prepare the arrays to plot the rhs and its approximations
x = np.linspace(0,l,200)
vec_approxrhs10=np.vectorize(approxrhs10)
vec_approxrhs50=np.vectorize(approxrhs50)
vec_rhs=np.vectorize(rhs)
y10=vec_approxrhs10(x)
y50=vec_approxrhs50(x)
z=vec_rhs(x)
    
plt.axis([0,l,-250,0])
plt.title("Rhs Fourier decomposition")
plt.plot(x,y10,'-', label='approx10')
plt.plot(x,y50,'-', label='approx50')
plt.plot(x,z, label='exact')
plt.legend()
plt.show()

#%%
# Here we approximate the solution with its Fourier decomposition finite series
# with 2 and 4 terms: it is enough, since the coefficients (from what we saw
# by analytical solution) scale very rapidly (1/n**3) and thus it converges
# quickly, and requires a few terms.
def sol2(x):
    sol2=0
    for j in range(1,2):
        sol2 += np.sin(j*np.pi*x/l)*fourieran(j)
    return sol2

def sol4(x):
    sol4=0
    for j in range(1,4):
        sol4 += np.sin(j*np.pi*x/l)*fourieran(j)
    return sol4

x = np.linspace(0,l,200)
vec_sol2=np.vectorize(sol2)
vec_sol4=np.vectorize(sol4)
y2=vec_sol2(x)
y4=vec_sol4(x)
z=vec_rhs(x)
    
plt.axis([0,l,-0.1,0])
plt.title("Solution's Fourier decomposition")
plt.plot(x,y2,'-', label='sol2')
plt.plot(x,y4,'-', label='sol4')
plt.plot(x,y, label='exact')
plt.legend()
plt.show()

diff2=y-y2
diff4=y-y4
plt.axis([0,l,-0.01,0.01])
plt.title("Plot of differences betw approx and exact analytical sol")
plt.plot(x,diff2,'-',label='diff2')
plt.plot(x,diff4,'-',label='diff4')
plt.legend()
plt.show()