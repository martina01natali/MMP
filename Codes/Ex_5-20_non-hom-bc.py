# -*- coding: utf-8 -*-

# Ex. 5.20
# FINITE ELEMENTS TECHNIQUE - NON-HOM. DIRICHLET BOUNDARY CONDITIONS
# Differences from 5.18,5.19:
#     - k(x) = 1+x
#     - f(x) = 1
#     - u(0) = a, u(1) = b
#     - solution of the hom. problem is w
#     - solution of the non-hom problem is u = g+w
# The tridiag matrix must be written differently because the integral is no more
# simply the product between the derivatives of the elements of the basis with
# respect to x, but also multiplied by (1+x)
# The tridiagonal matrix is not affected by the non-hom BC, but the solution
# and the right hand side of the equation are.

import numpy as np
import matplotlib.pyplot as plt  

#definition of linear pieceweise function and its vectorization

def fun(x,i,h):
    return np.piecewise(x, [x < (i-1)*h, 
                            ((i-1)*h<=x) & (x<= i*h),
                            (i*h<x) & (x<= (i+1)*h),
                            x>(i+1)*h], 
                        [0, (x - (i - 1)* h)/h,-(x - (i + 1)* h)/h,0])

vfun=np.vectorize(fun)

#definition of the integrand for the projection of f(x) on the elements of the basis
from scipy.integrate import quad

def fun1(x,i,h):
    return fun(x,i,h)*1

# from here forth we chose the dimension of the base by fixing n
n=5
h=1/n
maindiag=np.empty(n-1)
offdiag=np.empty(n-2)
for i in range(n-1):
    maindiag[i]=2*((i+1)*h+1)/h
for i in range(n-2):
    offdiag[i]=-(h+2*(i+1)*h+2)/(2*h)


kmatrix=np.diag(offdiag, -1) + np.diag(maindiag) + np.diag(offdiag, 1)

print(kmatrix)

# definition of the vector f and its evaluation

f = np.empty(n-1)
for i in range(n-1):
    f[i]=quad(fun1,0,1,args=(i+1,h))[0]
    
f[0]=h+2*(h+2)/(2*h)
f[n-2]=h-(h-4)/(2*h)
print(f)

# solution of the linear system
coeff = np.linalg.solve(kmatrix, f)
print(coeff)

# The solution of the equation is obtained as the sum w+g, where w is the
# solution of the hom problem and is given by the summation of the product bet
# the functions of the basis and the coefficients of the lin. algebra problem,
# and g is the linear function made by g = a*fun[0] + b*fun[n]

def sol(x):
    sol=0
    for i in range(n-1):
        sol += fun(x,i+1,h)*coeff[i]
    sol += 2*fun(x,0,h) + 1*fun(x,n,h)
    return sol

# exact solution
def exact(x):
    return np.log(1+x)/np.log(2)-x

vexact=np.vectorize(exact)
vsol=np.vectorize(sol)
t = np.linspace(0, 1, 101) 
s=vsol(t)
#ex=vexact(t)
plt.plot(t,s) 
#plt.plot(t,ex)
plt.show() 



