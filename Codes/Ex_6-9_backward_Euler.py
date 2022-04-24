# -*- coding: utf-8 -*-

         #***************************************************#

# Example 6.9 - Method of lines for the heat equation with homogeneous
# Dirichlet boundary conditions and using the backward Euler method.

# The direct Euler technique fails with STEEP DIFFERENTIAL EQUATIONS, because
# the solution explodes. The problem is to be found in the rapidly changing
# components. The trick is to use very small time steps (this makes the
# computational time longer). I should try to find a step not too small but
# not to large either.
# N.B. I have a condition on choosing h (step on x), that is \Delta(t) goes as h^2
# The HEAT EQUATION is STEEP indeed. You can see it by looking at the solution
# of the equation by the Fourier components: the coefficients are exponentials
# that go as n^2 (n=number of elements that we use for the base). E.g. if the
# initial situation is of a finite temperature != 0, I need a lot of components
# to describe the first phases, but the components then vanish very rapidly.
# Here, I should use inverse Euler.

         #***************************************************#

import numpy as np
import matplotlib.pyplot as plt  

n=20
h=100/n

r= 7.88
c= 0.437 
lam= 0.836 
t0= 0
tfin=180 
ntstep=90 
dt=(tfin-t0)/ntstep


maindiagm=np.empty(n-1)
offdiagm=np.empty(n-2)
for i in range(n-1):
    maindiagm[i]=2*h*r*c/3
for i in range(n-2):
    offdiagm[i]=h*r*c/6
    
m=np.diag(offdiagm, -1) + np.diag(maindiagm) + np.diag(offdiagm, 1)

#print("m=",m)
    
maindiagk=np.empty(n-1)
offdiagk=np.empty(n-2)
for i in range(n-1):
    maindiagk[i]=2*lam/h
for i in range(n-2):
    offdiagk[i]=-lam/h

k=np.diag(offdiagk, -1) + np.diag(maindiagk) + np.diag(offdiagk, 1)
    
#print("k=",k)

mb=np.zeros((3,n-1))
for i in range(n-2):
    mb[0,i+1]=offdiagm[i]
    mb[2,i]=offdiagm[i]
for i in range(n-1):
    mb[1,i]=maindiagm[i]
    
#print("mb=",mb) 
    
kb=np.zeros((3,n-1))
for i in range(n-2):
    kb[0,i+1]=offdiagk[i]
    kb[2,i]=offdiagk[i]
for i in range(n-1):
    kb[1,i]=maindiagk[i]

#print("kb=",kb)

# in fun(x,i) and similarly in fun1(x,t,i) the index i runs between 1 and n-1

def fun(x,i):
    return np.piecewise(x, [x < (i-1)*h, 
                            ((i-1)*h<=x) & (x<= i*h),
                            (i*h<x) & (x<= (i+1)*h),
                            x>(i+1)*h], 
                        [0, (x - (i - 1)* h)/h,-(x - (i + 1)* h)/h,0])

vfun=np.vectorize(fun)


def rhs(x,t):
    return 10**(-8)*t*x*(100-x)**2

#vrhs=np.vectorize(rhs)

grid=np.linspace(0,100,101)
#gridrhs=vrhs(grid,1)
#plt.plot(grid,gridrhs)
#plt.show()

#pfun=vfun(grid,2)
#plt.plot(grid,pfun)
#plt.show()

#print(fun)

from scipy.integrate import quad

def fun1(x,t,i):
    return fun(x,i)*rhs(x,t)

# in f(t,i) the index i runs from 0 to n-2

def f(t,i):
    return quad(fun1,0,100,args=(t,i+1))[0]

# to speed-up calculations f(t,i) can be vectorized so to evaluate
# all its values corresponding to i in range 0 to n-2 and all t ranging
# from t0 to tmax in steps dt in one command

vf=np.vectorize(f)  

vali=np.linspace(0,n-2,n-1)  


def fvalues(t):
    return vf(t,vali)

#print(fvalues(2))
    
#print(vf(1,vali)) 

#def approx(x):
#    approx=0
#    for i in range(n-1):
#        approx = approx + fun(x,i+1)*fvalues[1,i]
#    return approx/5

#vapprox=np.vectorize(approx)
#plt.plot(grid,vapprox(grid))
#plt.show()

"""
I want to use solve_banded here as well, and I have to create a banded matrix
"""
mback=mb+dt*kb
    
al=np.zeros(n-1)

from scipy.linalg import solve_banded 

#tvalues=np.linspace(0,10,10)

for t in range(ntstep+1):
    alp = solve_banded((1,1),mback, np.dot(m,al)+dt*fvalues((t+1)*dt))
        # f is evaluated at t+1 by the analytical construction of the problem
        # (see notes for more)

#--------------------------------------------
# alternative version using the normal linear algebra solver, that solves a
# problem of type ax = b finding x

# coeff=np.linalg.solve(m,-np.dot(k,al)+fvalues(t*dt))
#--------------------------------------------

    al=alp
    
#    print(t/2,"  coeff=",coeff)    
#    print(t/2,"  al=",al)


# evaluation of the solution of the equation
def sol(x):
    sol=0
    for i in range(n-1):
        sol = sol + fun(x,i+1)*al[i]
    return sol


vsol=np.vectorize(sol)
s=vsol(grid)

plt.plot(grid,s) 
plt.show() 


