# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:37:41 2021

@author: Martina
"""

# Goal is to expand functions on some provided basis.
# We are asking ourselves if we can expand an arbitrary function that is
# not subject to the boundary conditions and doesn't fit them to the same
# basis that we have found by imposing thoe boundary conditions.
# First we solve the equation analytically by using sympy (symbolic python).

import numpy as np
import matplotlib.pyplot as plt

# These are the two functions, defined in the domain [0;1]. The function f1
# vanishes at the limits of the interval.

def f1(x):
    return x*(1-x)

def f2(x):
    return 1-x

def base(x,n):
    return np.sin(n*np.pi*x)

# We want to compute the Fourier components of f1 and f2 wrt the basis base.
# So we have to integrate f1 and f2 with each of the elements of the basis.
# How to do it:
# - analytic way: use sympy
# - numerical way: Simpson's rule, Euler, Runge-Kutta...
# - gaussian quadrature: exploits a method that optimizes the points chosen
# where you evaluate the function that you're integrating
# - choose a technique already made in python

# I import a simple numerical integration routine, quad, from the scipy 
# library, scipy.integrate. The function quad computes DEFINITE integrals.
# It takes as arguments the function to integrate, the limits of integration,
# and the variables to consider as parameters. For example, if the function
# that you're integrating depends on x and n, and you want to integrate
# in x, then you should pass n as an additional argument, args = (n).
# The function quad returns the result of the integration and the associated
# error as a list of two floats. The routine quad exploits the Gauss-Legendre
# method (Gaussian quadrature).

from scipy.integrate import quad

# Projecting the function on the basis means to integrate the product of the
# function and the elements of the basis.

# Integrand functions
def fun1(x,n):
    return f1(x)*base(x,n)

def fun2(x,n):
    return f2(x)*base(x,n)

# Integrals with quad
def fourierf1(n):
    return 2*quad(fun1,0,1,args=(n))[0]

def fourierf2(n):
    return 2*quad(fun2,0,1,args=(n))[0]

# These are the coefficient of the expansion of the functions:
# those for f1 scale as 1/n**3, those of f2 scale as 1/n.
# This is very general: you have a basis, and the basis corresponds to some
# function (sin in this case, that is vanishing at 0): the way in which
# the coefficients of the expansion are behaving depends on the comparison
# between the function you are expanding with the elements of the base.
# In the case of f1, both the function and the basis have the first
# derivative that is smooth.The coefficients go to zero very rapidly: this
# means that the convergence of the series is very fast.
# In the case of f2, there's a discontinuity between the elements of the
# basis and the expanded function: in x=0, the basis vanishes but the
# function goes to one. Then the coefficients of the expansion go to zero
# as 1/n. If the discontinuity in the behavior of the two was at the level
# of the first derivative, the coefficient would have gone as 1/n^2.

# The more the function behaves as the basis and as the boundary conditions,
# the more the series converges rapidly.

# Then we build an approximation of f1 and f2 using a finite number of
# elements of the basis for writing the Fourier expansion of the functs.

def approxf1(x,n):
    approxf1=0
    for j in range(1,n+1): #remember that the n+1 index is not included
        approxf1 += base(x,j)*fourierf1(j)
    return approxf1

def approxf2(x,n):
    approxf2=0
    for j in range(1,n+1):
        approxf2 += base(x,j)*fourierf2(j)
    return approxf2

# We want to plot my function and the approximation to see how well they fit.
# The fact is that even with the first 10 terms, taking f1 for ex., they
# are impossible to distinguish. So it is smarter to plot the difference.

# Here we prepare the arrays to plot the right hand side (the function)
# and its approximations.
# Vectorization through the function np.vectorize means that the function
# that is vectorized accepts as arguments even vectors (numpy arrays)
# instead of simple single numbers. This is very convenient because it
# allows you to skip the creation of heavy loops inside your customary
# functions.

x = np.linspace(0,1,200)

vec_approxf1=np.vectorize(approxf1)
vec_approxf2=np.vectorize(approxf2)
vec_f1=np.vectorize(f1)
vec_f2=np.vectorize(f2)

f1_10=vec_approxf1(x,10)
f1_20=vec_approxf1(x,20)
f1_40=vec_approxf1(x,40)
f1_80=vec_approxf1(x,80)
f1_exact=vec_f1(x)

f2_10=vec_approxf2(x,10)
f2_20=vec_approxf2(x,20)
f2_40=vec_approxf2(x,40)
f2_80=vec_approxf2(x,80)
f2_exact=vec_f2(x)

# I have created the approximations for f1 and f2 and their exact value.
# Now I want to plot the difference bet the functions and the approximations.
# Since the order of magnitudes may differ greatly, we evaluate the log10 of
# the differences directly and then we plot it

difff1_10=np.log10(np.abs(f1_exact-f1_10))
difff1_20=np.log10(np.abs(f1_exact-f1_20))
difff1_40=np.log10(np.abs(f1_exact-f1_40))
difff1_80=np.log10(np.abs(f1_exact-f1_80))

plt.figure()
plt.ylim(-10, 0)
plt.plot(x,difff1_10,'-', label='approx10')
plt.plot(x,difff1_20,'-', label='approx20')
plt.plot(x,difff1_40,'-', label='approx40')
plt.plot(x,difff1_80,'-', label='approx80')
plt.title("Plot of differences")
plt.ylabel(r"$\Delta$ [log]")
plt.xlabel(r"x")
plt.legend()
plt.show()

# The different plots with the approximations improving the description of
# the function are made for f2.
# There is not a point-like convergence, but there's an average (sort of)
# convergence, meaning that the integral of the difference vanishes.

plt.axis([0,1,0,1.5])
plt.plot(x,f2_10,label='approx10')
plt.plot(x,f2_exact)
plt.legend()
plt.show()

plt.axis([0,1,0,1.5])
plt.plot(x,f2_20,label='approx20')
plt.plot(x,f2_exact)
plt.legend()
plt.show()

plt.axis([0,1,0,1.5])
plt.plot(x,f2_40,label='approx40')
plt.plot(x,f2_exact)
plt.legend()
plt.show()

plt.axis([0,1,0,1.5])
plt.plot(x,f2_80,label='approx80')
plt.plot(x,f2_exact)
plt.legend()
plt.show()
