# -*- coding: utf-8 -*-

        #***************************************************#

# Example 6.8 - Method of lines for the heat equation with homogeneous
# Dirichlet boundary conditions and using the forward Euler method.

# Suppose an iron bar (\rho = 7.88, c = 0.437, k = 0.836) is chilled to a
# constant temperature of 0 °C and then heated internally with both ends
# mantained at 0 °C. Suppose further that the bar is 100 cm long and heat
# energy is added at the rate of f(x,t) = 1e-08*tx*(100-x)**2 W/cm**3.
# What is the temperature distribution after 3 minutes (180 s)?

        #***************************************************#

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import solve_banded

# Definition of number of intervals for the piecewise linear basis
# and related step in x, given by lenght of bar / number of intervals
n = 10
h = 100/n

# Definition of parameters that enter in the IBVP
r = 7.88
c = 0.437
lam = 0.836

# Build the mass and stiffness matrixes.
# The coefficients of the matrixes are computed analytically, but in
# principle can be computed with any integration routine, provided that
# we first define the piecewise linear basis elements and their
# derivatives wrt space, that enter in the stiffness matrix.
# (check Gockenbach, chap. 6.4.1, Example 6.8, pag. 240)
# Note: matrixes m and k have indexes ranging from 0 to n-2,
# spanning n-1 values
maindiagm = np.empty(n-1)
offdiagm = np.empty(n-2)
for i in range(n-1):
    maindiagm[i] = 2*h*r*c/3
for i in range(n-2):
    offdiagm[i] = h*r*c/6

# As usual, the matrixes whose entries are given by inner product of the
# elements of the standard piecewise linear basis are sparse (as the Galerkin
# method requires) and more precisely are triadiagonal and symmetric
m = np.diag(offdiagm, -1) + np.diag(maindiagm) + np.diag(offdiagm, 1)
print("m=", m)

# %%
maindiagk = np.empty(n-1)
offdiagk = np.empty(n-2)
for i in range(n-1):
    maindiagk[i] = 2*lam/h
for i in range(n-2):
    offdiagk[i] = -lam/h

k = np.diag(offdiagk, -1) + np.diag(maindiagk) + np.diag(offdiagk, 1)

print("k=", k)

# %%
# We want to diagonalize M and K by using the solve_banded routine of
# scipy.linalg, so I have to build them in a way that is compatible with
# the syntax of the function. solve_banded requires a 3xn matrix in
# which each row is a diagonal (starting from the lowest to the highest)
# of the original banded matrix.
mb = np.zeros((3, n-1))
for i in range(n-2):
    mb[0, i+1] = offdiagm[i]
    mb[2, i] = offdiagm[i]
for i in range(n-1):
    mb[1, i] = maindiagm[i]

print("mb=", mb)


# Looking for a way to print this matrixes nicely but haven't found it yet

# for element in mb:
#     for entry in element:
#         print("{:.2f}".format(entry))

# for line in mb:
# print(*line)

# %%
kb = np.zeros((3, n-1))
for i in range(n-2):
    kb[0, i+1] = offdiagk[i]
    kb[2, i] = offdiagk[i]
for i in range(n-1):
    kb[1, i] = maindiagk[i]

print("kb=", kb)

# Good ol' piecewise linear basis of triangular functions between 0 and l
# The indexes of the elements of the basis go from 1 to n-1, where n is the
# number of intervals chosen


def fun(x, i):
    return np.piecewise(x, [x < (i-1)*h,
                            ((i-1)*h <= x) & (x <= i*h),
                            (i*h < x) & (x <= (i+1)*h),
                            x > (i+1)*h],
                        [0, (x - (i - 1) * h)/h, -(x - (i + 1) * h)/h, 0])


vfun = np.vectorize(fun)

# Build the projection of the rhs f(x,t) of the IBVP on the basis
# First define the rate of heat function and plot it a fixed time
def rhs(x, t):
    return 10**(-8)*t*x*(100-x)**2


vrhs = np.vectorize(rhs)

grid = np.linspace(0, 100, 101)
gridrhs = vrhs(grid, 1)
plt.figure(1)
plt.plot(grid, gridrhs)
plt.title("Plot of f(x,t) VS x at time t = 1 s")
plt.show()

# pfun=vfun(grid,2)
# plt.plot(grid,pfun)
# plt.show()

# Build the integrand function for computing the vector of coefficients
def fun1(x, t, i):
    return fun(x, i)*rhs(x, t)


def f(t, i):
    return quad(fun1, 0, 100, args=(t, i+1))[0]

# To speed-up calculations f(t,i) can be vectorized so to evaluate
# all its values corresponding to i in range 0 to n-2 and all t ranging
# from t0 to tmax in steps dt in one command
vf = np.vectorize(f)

# I make a column of values that depend on t only, and are the integration of
# f(x,t) multiplied by the basis functions, that means
# the projections of f(x,t) on each element of the basis
vali = np.linspace(0, n-2, n-1)


def fvalues(t):
    """Returns the values of f for all the x_i points at a given time t"""

    return vf(t, vali)


print(fvalues(2))
# print(vf(1,vali))

# %%

# def approx(x):
#     approxim=0
#     for i in range(n-1):
#         approxim += fun(x,i+1)*fvalues[1,i]
#     return approx/5

# vapprox=np.vectorize(approx)
# plt.plot(grid,vapprox(grid))
# plt.show()

         #***************************************************#
         
# Build the solution of the linear system of ODEs such that
# da/dt = M**-1(-Ka+f)
# We implement the Euler method to compute the integral that is needed to
# solve the above system of ODEs.
# The (forward) Euler method takes the form
# a(i+1) = a(i)+dt*M**-1(-Ka+f)
# but computing M**-1 is not efficient at all, since M is tridiagonal and
# thus M**-1 is completely dense. So instead we solve an ausiliary linear
# algebra problem, M*s(i) = -K*a(i)+f in s(i), thus getting
# a(i+1) = a(i) + s(i)*dt
# In this way, we need to compute s(i) for each iteration, but we save a lot
# of computational time. Each iteration happens on a different instant in
# time, so the range that we take for the loop will be related to the
# number of steps in time.

         #***************************************************#
t0 = 0
tfin = 180
ntstep = 260
dt = (tfin-t0)/ntstep

al = np.zeros(n-1)

for t in range(ntstep+1):
    # if t0 != 0 you should put (t0+t)*dt
    coeff = solve_banded((1, 1), mb, -np.dot(k, al)+fvalues(t*dt))
    al += dt*coeff

# --------------------------------------------
# alternative version using the normal linear algebra solver, that solves a
# problem of type ax = b finding x

# coeff=np.linalg.solve(m,-np.dot(k,al)+fvalues(t*dt))
# --------------------------------------------

#    print(t/2,"  coeff=",coeff)
#    print(t/2,"  al=",al)


# Evaluation of the approximated solution u_n of the equation
def sol(x):
    sol = 0
    for i in range(n-1):
        sol += fun(x, i+1)*al[i]
    return sol


vsol = np.vectorize(sol)
s = vsol(grid)

plt.figure(2)
plt.plot(grid, s)
plt.show()
