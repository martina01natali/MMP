# -*- coding: utf-8 -*-

        #***************************************************#

# Example 6.1 - Homogeneous Dirichlet boundary conditions

# We consider a 50 cm iron bar, with specific heat c = 0.437 J/(gK),
# density \rho = 7.88 g/cm^3, and thermal conductivity k = 0.836 W/(cmK).
# We assume that the bar is insulated except at the ends and that the
# initial temperature distribution is
# \psi(x) = 5 - 1/5*|x-25|
# in degrees Celsius. Finally, we assume that at time t=0 the ends of the
# bar are placed in an ice bath (0 degree Celsius). Compute the temperature
# distribution after 20, 60 and 300 seconds.

        #***************************************************#

import numpy as np
import matplotlib.pyplot as plt  
from scipy.integrate import quad

# Define the data that are given by the problem
c = 0.437
r = 7.88
k = 0.836

# Define the initial condition function (blue curve (angular function))
def psi(x):
    return 5-np.abs(x-25)/5

vpsi=np.vectorize(psi) # vectorization allows to pass an array as argument

# Plot the initial condition
x = np.linspace(0, 50, 101)
ppsi=vpsi(x)
plt.plot(x,ppsi)

#%%
# Define the elements of the Fourier sine basis
def phi(x,n):
    return np.sin(n*np.pi*x/50)

# Define the integrand function for computing the *b_n* coefficients of the
# initial condition function, and integrate with quad
def psi1(x,n):
    return psi(x)*phi(x,n)

# Set the maximum number of coefficients to compute
nmax=100

# Create an array that contains the coefficients
# Notice that we are first exploiting a list comprehension to produce a list,
# passed to np.array() as an argument
b = np.array([(2/50)*quad(psi1,0,50,args=(n))[0] for n in range(nmax)])
print(b)

#%%

# Define the coefficients *a_n* of the projection of the solution u(x,t)
# Here we are exploiting the analytical solution that is known and can
# be found on the book (Gockenbach, chap. 6.1.1, pag. 198)
def acoeff(t,n):
    return np.exp(-k*n**2*np.pi**2*t/(50**2*r*c))

tgrid = np.linspace(0, 100)
acoeff_v = np.vectorize(acoeff)

# Plot the coefficients a, that only depend on time, for increasing n,
# to see how they vanish faster as n grows (as exp(-n**2))
plt.figure(3)
for i in range(9):
    plt.plot(tgrid, acoeff_v(tgrid, i+1))

#%%
# Build an approximation of the solution as the summation of a finite
# number of elements of the Fourier series
def sol(x,t,ntop):
    """Approximated solution as a finite sine series with ntop terms.

    Parameters
    ----------
    x : float
        Position along the bar or node of a mesh.
    t : float
        Time instant.
    ntop : int
        Number of terms to sum.

    Returns
    -------
    sol : float
        Approximated solution at ntop terms.

    """
    sol=0
    for i in range(ntop):
        sol += b[i]*acoeff(t,i)*phi(x,i)
    return sol

vsol=np.vectorize(sol)


# Plot the solution at different times, with the approximated solution
# truncated at 10 terms
plt.figure(1)
s20=vsol(x,20,10)
s40=vsol(x,40,10)
s60=vsol(x,60,10)
plt.plot(x,s20)
plt.plot(x,s40)
plt.plot(x,s60) 
plt.show()

#%%
# Plot the solution at the same time (20), with an increasing number of terms
# for the solution 
plt.figure(2)
s202=vsol(x,20,2)
s204=vsol(x,20,4)
s206=vsol(x,20,6)
s208=vsol(x,20,8)
s20100=vsol(x,20,100)
plt.plot(x,s202)
plt.plot(x,s204)
plt.plot(x,s206)
plt.plot(x,s208)
plt.plot(x,s20100)
plt.show()

        #***************************************************#

# Comment

# Almost immediatly after t=0, the irregularity disappears:
# the reason is that the coefficients with very large n are immediately
# suppressed by the exponential that vanishes with -n^2.
# In fact, since our function is decomposed on a sine basis, the
# components that contribute to the cusp (cuspide) are the ones with
# a very high frequency (they oscillate rapidly since their argument, that
# goes as n, is very big), since components with low frequency are, obviously,
# much smoother. The latter can be easily seen in Figure 2, where the solution
# is plotted at a fixed time, with an increasing number of terms in the
# finite series that approximates it.
#
# More observations can be found at chap. 6.1.1 of Gockenbach, Example 6.1,
# pag 198-199.