# -*- coding: utf-8 -*-

        #***************************************************#

# Example 7.7 - Finite elements method for the homogeneous wave equation with
# homogeneous Dirichlet boundary conditions and initial conditions

# Reference: Gockenbach, chap. 7.3.1, Example 7.7, pag. 283

# Consider the homogeneous wave equation (f(x,t) = 0) with homogeneous
# Dirichlet boundary conditions, null initial velocity (du/dt(x,t0) = 0) and
# initial condition on u, u(x,t0) = \psi(x). The lenght of the string is 1 m,
# the speed of the waves is 522 m/s, thus the natural frequency being 261 Hz.
# The initial shape of the string is u(x,t0) = \psi(x) = 0.01x*(1-x).
# We assume (and we get in the end) that the solution of the IBVP is smooth,
# so we can apply the finite elements method, knowing that it fails (or at
# least the standard procedure with Galerkin that we know fails) for
# solutions with singularities such as jumps.

        #***************************************************#

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

c = 522
t0 = 0
tf = 1/(c/2)
ntimestep = 50
dt = tf/ntimestep
n = 20  # Intervals for the linear polynomials
h = 1/n  # Step

# Standard triangular basis for piecewise linear polynomials


def fun(x, i):
    return np.piecewise(x, [x < (i-1)*h,
                            ((i-1)*h <= x) & (x <= i*h),
                            (i*h < x) & (x <= (i+1)*h),
                            x > (i+1)*h],
                        [0, (x - (i - 1) * h)/h, -(x - (i + 1) * h)/h, 0])


# Build the mass M and stiffness K matrixes in banded form to pass to
# function solve_banded
# Recall that
# M_ij = integral_0^l(fun(x,j)*fun(x,i))
# K_ij = integral_0^l(c**2*dfun(x,j)/dx*dfun(x,i)/dx)
# and piecewise derivatives are better computed analytically (carta e penna)
maindiagm = np.ones(n-1)*(2*h/3)
offdiagm = np.ones(n-2)*(h/6)

maindiagk = np.ones(n-1)*(2/h)
offdiagk = np.ones(n-2)*(-1/h)

m = np.diag(offdiagm, -1) + np.diag(maindiagm) + np.diag(offdiagm, 1)
k = np.diag(offdiagk, -1) + np.diag(maindiagk) + np.diag(offdiagk, 1)
k = k*c**2

mb = np.zeros((3, n-1))
for i in range(n-2):
    mb[0, i+1] = offdiagm[i]
    mb[2, i] = offdiagm[i]
for i in range(n-1):
    mb[1, i] = maindiagm[i]

kb = np.zeros((3, n-1))
for i in range(n-2):
    kb[0, i+1] = offdiagk[i]
    kb[2, i] = offdiagk[i]
for i in range(n-1):
    kb[1, i] = maindiagk[i]

kb = kb*c**2

# Initial condition/shape of u(x,t)


def psi(x):
    return 0.01*x*(1-x)


# Initial values of vector-valued function u(t0)=y0
y0 = np.array([psi((i+1)*h) for i in range(n-1)])


def vib(d):
    vib = 0
    for i in range(n-1):
        vib = vib+y0[i]*fun(d, i+1)
    return vib


vvib = np.vectorize(vib)

# check that the initial value "init" and the approximation of it obtained
# by using vib with y0 "sol0" are very similar
grid = np.linspace(0, 1, 101)
vpsi = np.vectorize(psi)
init = vpsi(grid)
sol0 = vvib(grid)

plt.figure(1, dpi=300)
plt.plot(grid, init, label="init", linestyle="-", lw=3, alpha=1.)
plt.plot(grid, sol0, label="sol0", linestyle="--", lw=5, alpha=0.7)
plt.legend()
plt.show()

plt.figure(2, dpi=300)
plt.plot(grid, abs(init-sol0))
plt.ylim(-1e-05, 1e-05)
plt.title("Plot of difference bet init and sol0")
plt.show()


z0 = np.zeros(n-1)
y = np.copy(y0)
z = np.copy(z0)

# Implementation of Runge-Kutta 4
# I have to solve the equation for s1 for every correction, after the
# implementation of the result found on the previous step

for i in range(13):
    s1 = solve_banded((1, 1), mb, -np.dot(k, y))
    k1y = z
    k1z = s1
    s2 = solve_banded((1, 1), mb, -np.dot(k, y+(1/2)*dt*k1y))
    k2y = z+(1/2)*dt*k1z
    k2z = s2
    s3 = solve_banded((1, 1), mb, -np.dot(k, y+(1/2)*dt*k2y))
    k3y = z+(1/2)*dt*k2z
    k3z = s3
    s4 = solve_banded((1, 1), mb, -np.dot(k, y+dt*k3y))
    k4y = z+dt*k3z
    k4z = s4
    y0 = np.copy(y)
    z0 = np.copy(z)
    y = y0+(k1y+2*k2y+2*k3y+k4y)*dt/6
    z = z0+(k1z+2*k2z+2*k3z+k4z)*dt/6


def vib(d):
    vib = 0
    for i in range(n-1):
        vib = vib+y[i]*fun(d, i+1)
    return vib


vvib = np.vectorize(vib)

sol = vvib(grid)

plt.figure(3, dpi=300)
plt.plot(grid, init, label="initial condition")
plt.plot(grid, sol, label="solution")
plt.legend()
plt.title("Plot of the initial condition and the solution computed with RK4")
plt.show()
