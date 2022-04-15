# -*- coding: utf-8 -*-
 
# Ex. 5.18
# FINITE ELEMENTS TECHNIQUE - HOMOGENEOUS DIRICHLET BOUNDARY CONDITIONS
# We have -d^2u/dx^2 = -exp(x)
#     - k(x) = 1
#     - f(x) = exp(x)
#     - 0 <= 0 <= 1
#     - u(0) = 0 = u(1)
 
import numpy as np
import matplotlib.pyplot as plt  
 
# 1. Define the basis of piecewise linear polynomials.
# piecewise is defined with a list of conditions, that have to be
# mutually exclusive and correspond to an equal number of values for the
# function. Note that & works between booleans but the expression is
# evaluated only if you put your condition between brackets.
# FIRST DEFINE THE X-GRID, THE STEP, THE DIMENSION OF THE BASIS,
 
n=10
h=1/n

def fun(i,x,h):
    """Element of the basis in finite element method with piecewise polynomials
    
    Parameters
    ----------
    i : int
        Index (not-null integer) of the piecewise polynomials' basis, must
        be generated with some iterable and run inside a loop. Maximum value
        is the number of intervals-1.
    x : float
    h : step
        Must correspond to width of interval/number of intervals.

    Returns
    -------
    ndarray
        Returns a piecewise function that is the element of the S_n basis of
        piecewise polynomials, with provided index, x and step.

    """
    return np.piecewise(x,
                        [x < (i-1)*h,
                         ((i-1)*h<=x)&(x<= i*h),
                         (i*h<x)&(x<= (i+1)*h),
                         x>(i+1)*h], 
                        [0,
                         (x - (i - 1)* h)/h,
                         -(x - (i + 1)* h)/h,
                         0]
                        )
vfun=np.vectorize(fun)
 
# I build the x-grid with 10 intervals (9 points) between 0 and 1, so with
# a step h = 0.1.
 
t = np.linspace(0, 1, 101) # values for x, must be more than intervals chosen
y=[]
for i in range(1,n-1) :
    y=vfun(i,t,0.1)
    plt.plot(t,y)
plt.xlabel('x')  
plt.ylabel('y')  
plt.show()

#%%
# 2. I define a customary routine for building a tridiagonal matrix.
# The matrix is the matrix k on the base defined by fun.
 
def tridiagmod(a, b, c, n):
    aa=a*np.ones(n-1)
    bb=b*np.ones(n)
    cc=c*np.ones(n-1)
    return np.diag(aa, -1) + np.diag(bb) + np.diag(cc, 1)
 
kmatrix=tridiagmod(-1/h,2/h,-1/h,n-1)
print(kmatrix)

#%%
# Now we have to expand the function f=exp(x) on the base given by fun.
# I define fun1 as the product between the elements of the basis and the
# function on the right hand side (so it is the integrand)

# Now I can build the column containing the values of the expansion of f.
# I employ the pre-build routine quad to integrate even if the solution is
# analytical.
# The index i of the loop goes to 0 to n-2 (range(n-1) = 0,...,(n-1)-1) but I
# want f to be expanded on the elements of the basis from index 1 to n-1.
# I can do that by passing i+1 to the arguments of the integration. 
 
from scipy.integrate import quad
def fun1(x,i,h):
    return fun(i,x,h)*np.exp(x)

f = np.ones(n-1)
for i in range(n-1):
    f[i]=quad(fun1,0,1,args=(i+1,h))[0]
print(f)

#%%
# Now I have to solve the linear algebra problem k*coeff=f and I use a
# pre-made routine for doing that (linalg.solve())

coeff = np.linalg.solve(kmatrix, f)
print(coeff)

#%%
# Then I have to build the solution as a finite series of n-1 terms given by
# the product of the coefficients coeff given by the linear algebra problem
# and the elements of the piecewise polynomials' basis.
# Remember that we only need n-1 elements of the basis and of the solution
# because it is vanishing at 0 and 1. The little trick on the range and on the
# arguments of the function is the same as done in the loop above.
# At the end, plot the exact solution for comparison.
 
def sol(x):
    sol=0
    for i in range(n-1):
        sol += fun(i+1,x,h)*coeff[i]
    return sol

def exact(x):
    return -np.exp(x)+(np.e-1)*x +1

vexact=np.vectorize(exact)
vsol=np.vectorize(sol)
s=vsol(t)
ex=vexact(t)
plt.plot(t,s) 
plt.plot(t,ex)
plt.show()