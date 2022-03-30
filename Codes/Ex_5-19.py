# -*- coding: utf-8 -*-
"""
Ex. 5.19
FINITE ELEMENTS TECHNIQUE - HOM. DIRICHLET BOUNDARY CONDITIONS
Differences from 5.18:
    - k(x) = 1+x
    - f(x) = 1
The tridiag matrix must be written differently because the integral is no more
simply the product between the derivatives of the elements of the basis with
respect to x, but also multiplied by (1+x)
"""
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


#recall that in Python arrays and matrixes contain the element 0

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
#here I wrote the analytical solution of the integral that I am calculating
#in the case I have no analytical solution fror the integral I should use
#a numerical technique and insert the routine here  
    
for i in range(n-2):
    offdiag[i]=-(h+2*(i+1)*h+2)/(2*h)


kmatrix = np.diag(offdiag, -1) + np.diag(maindiag) + np.diag(offdiag, 1)
print(kmatrix)

# definition of the vector f and its evaluation
# note that in fun(x,i,h), fun1(x,i,h) i must run from 1 to n-1
# and not from 0 to n-2. Therefore we need to shift the argument 
f = np.empty(n-1)
for i in range(n-1):
    f[i]=quad(fun1,0,1,args=(i+1,h))[0]
    
print(f)

# solution of the linear system
# calling a generic routine to solve a linear algebra problem
coeff = np.linalg.solve(kmatrix, f)
print(coeff)

# evaluation of the solution of the equation
def sol(x):
    sol=0
    for i in range(n-1):
        sol = sol + fun(x,i+1,h)*coeff[i]
    return sol

# exact solution
def exact(x):
    return np.log(1+x)/np.log(2)-x
vexact=np.vectorize(exact)
vsol=np.vectorize(sol)
t = np.linspace(0, 1, 101) 
s=vsol(t)
ex=vexact(t)
plt.plot(t,s) 
plt.plot(t,ex)
plt.show()