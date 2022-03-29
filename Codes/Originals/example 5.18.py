# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:56:00 2020

@author: Drago
"""


import numpy as np
import matplotlib.pyplot as plt  

#definition of linear pieceweise function and its vectorization

def fun(i,x,h):
    return np.piecewise(x, [x < (i-1)*h, 
                            ((i-1)*h<=x) & (x<= i*h),
                            (i*h<x) & (x<= (i+1)*h),
                            x>(i+1)*h], 
                        [0, (x - (i - 1)* h)/h,-(x - (i + 1)* h)/h,0])

vfun=np.vectorize(fun)

###############################################################

#plot of a few examples
t = np.linspace(0, 1, 101) 
y1 = vfun(1,t,0.1)
y2 = vfun(2,t,0.1)
y3 = vfun(3,t,0.1)
y9 = vfun(9,t,0.1)

plt.plot(t, y1) 
plt.plot(t, y2) 
plt.plot(t, y3)  
plt.plot(t,y9)
plt.xlabel('x')  
plt.ylabel('y')  
plt.show() 

#################################################################

# defition of tridiagonal matrixes
def tridiagmod(a, b, c, n):
    ym1=np.ones(n-1)
    aa=a*ym1
    y=np.ones(n)
    bb=b*y
    cc=c*ym1
    return np.diag(aa, -1) + np.diag(bb) + np.diag(cc, 1)

#recall that in Python arrays and matrixes contain the element 0

#definition of the integrand for the projection of f(x) on the elements of the basis
from scipy.integrate import quad

def fun1(x,i,h):
    return fun(i,x,h)*np.exp(x)

# from here forth we chose the dimension of the base by fixing n
n=10
h=1/n
kmatrix=tridiagmod(-1/h,2/h,-1/h,n-1)
print(kmatrix)

# definition of the vector f and its evaluation
# note that fun, fun1
f = np.ones(n-1)
for i in range(n-1):
    f[i]=quad(fun1,0,1,args=(i+1,h))[0]
    
print(f)

# solution of the linear system
coeff = np.linalg.solve(kmatrix, f)
print(coeff)

# evaluation of the solution of the equation
def sol(x):
    sol=0
    for i in range(n-1):
        sol = sol + fun(i+1,x,h)*coeff[i]
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


