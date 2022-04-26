# -*- coding: utf-8 -*-

        #***************************************************#

# Example 7.3 - Fourier series method for the homogeneous wave equation with
# homogeneous Dirichlet boundary conditions and initial conditions

# Reference: Gockenbach, chap. 7.2.1, Example 7.3, pag. 269

# Consider the homogeneous (time-dep) wave equation with initial conditions
# and homogeneous Dirichlet boundary conditions, with l = 1 m, c = 522 m/s,
# f(x,t) = 0 (thus homogeneity), \gamma(x) = du/dt(x,t0) = 0 (initial
# condition for time-derivative of solution) and a piecewise initial condition
# for the solution \psi(x) (defined below). Compute the solution at different,
# arbitrary instants of time.

# The fundamental frequency of the string is given by c/2l = 522/2 = 261 Hz.
# Furthermore, from the initial conditions we can see that the string is fixed
# from 0 to 0.4 and from 0.6 to 1 m, and that no initial condition is defined
# for the center x = 0.5 m, from which we can deduce that the string is
# actually plucked there to start its vibration.

        #***************************************************#


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Definition of initial conditions to the solution.
# Recall that the syntax for np.piecewise() is
# np.piecewise(variable, [intervals], [values])
def psi(x):
    return np.piecewise(x,
                        [x < 0.4,
                         (0.4 <= x) & (x <= 0.5),
                         (0.5 < x) & (x <= 0.6),
                         x > 0.6],
                        [0, x-0.4, -(x-0.6), 0])


vpsi = np.vectorize(psi)
grid = np.linspace(0, 1, 101)
psigrid = vpsi(grid)

# Plot the initial condition
plt.plot(grid, psigrid)
plt.show()

# %%

# Since the external force function is null, and the initial conditions on the
# time derivative of u is null as well, the only function that we have to
# expand in series on the Fourier basis is the initial condition on u.
# To do this we prepare the integrand function ps1.
def psi1(x, n):
    return 2*psi(x)*np.sin(n*np.pi*x)


def b(n):
    return quad(psi1, 0, 1, args=(n))[0]


tb = np.array([b(i) for i in range(201)])

l = 1
c = 522
nat_freq = c/2*l # 261 Hz
tf = 1/nat_freq


def sol(x, t, nmax):
    sol = 0
    for n in range(nmax+1):
        sol += tb[n]*np.cos(c*n*np.pi*t)*np.sin(n*np.pi*x)
    return sol


vsol = np.vectorize(sol)
psol8 = vsol(grid, tf/8, 101) # nmax = 101 = number of Fourier series terms

# when the solution arrives at the boundaries, it flips
psol38 = vsol(grid, tf*3./8, 101)
psol2 = vsol(grid, tf/2, 101)
psol1 = vsol(grid, tf, 101)
plt.plot(grid, psol8, 'r')
plt.plot(grid, psol38, 'g')
plt.plot(grid, psol2, 'b')
plt.plot(grid, psol1, 'y')
plt.show()