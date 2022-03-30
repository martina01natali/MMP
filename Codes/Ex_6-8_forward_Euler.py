# -*- coding: utf-8 -*-
"""
26/11/2020
Heat equation solved by direct Euler and using routine solve_banded
As usual, this example is taken from the Gochenback (?check)
"""
import numpy as np
import matplotlib.pyplot as plt  

n=10
h=100/n

r= 7.88
c= 0.437 
lam= 0.836 
t0= 0
tfin=180 
ntstep=260 
dt=(tfin-t0)/ntstep


# matrixes m and k have indexes ranging from 0 to n-2, spanning n-1 values

maindiagm=np.empty(n-1)
offdiagm=np.empty(n-2)
for i in range(n-1):
    maindiagm[i]=2*h*r*c/3
for i in range(n-2):
    offdiagm[i]=h*r*c/6

# build a tri-diagonal matrix
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

# I have to build m and k in a way that is compatible with solve_banded
mb=np.zeros((3,n-1))
for i in range(n-2):
    mb[0,i+1]=offdiagm[i]
    mb[2,i]=offdiagm[i]
for i in range(n-1):
    mb[1,i]=maindiagm[i]
    
print("mb=",mb) 
    
kb=np.zeros((3,n-1))
for i in range(n-2):
    kb[0,i+1]=offdiagk[i]
    kb[2,i]=offdiagk[i]
for i in range(n-1):
    kb[1,i]=maindiagk[i]

#print("kb=",kb)

# in fun(x,i) and similarly in fun1(x,t,i) the index i runs between 1 and n-1

def fun(x,i):
    return np.piecewise(x, [x < (i-1)*h, #first conditions gives first result
                            ((i-1)*h<=x) & (x<= i*h), #2nd condition
                            (i*h<x) & (x<= (i+1)*h), #3rd condition
                            x>(i+1)*h], #4th condition
                        [0, (x - (i - 1)* h)/h,-(x - (i + 1)* h)/h,0])

vfun=np.vectorize(fun)


def rhs(x,t): #right hand side (f(x,t))
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

# we perform the integration of function f(x,t) by a numerical method,
# instead of using the analytical form 
from scipy.integrate import quad

def fun1(x,t,i):
    return fun(x,i)*rhs(x,t)

# in f(t,i) the index i runs from 0 to n-2

def f(t,i):
    return quad(fun1,0,100,args=(t,i+1))[0]
# I put i+1 because the piecewise function fun is defined from 0 to n, but
# I need to integrate between 1 and n-1 because the 0 and n functions are 0

# to speed-up calculations f(t,i) can be vectorized so to evaluate
# all its values corresponding to i in range 0 to n-2 and all t ranging
# from t0 to tmax in steps dt in one command

vf=np.vectorize(f)  

vali=np.linspace(0,n-2,n-1)  

# I make a column of values that depend on t only, and are the integration of
# f(x,t) multiplied by the basis functions (see also notes 26/11), that means
# the projections of f(x,t) on each element of the basis
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
    
al=np.zeros(n-1)

from scipy.linalg import solve_banded 

#tvalues=np.linspace(0,10,10)

# I compute the function u that is the "velocity" of the coefficients
# (take a look at the notes, it's necessary)
# u depends on the inverse of matrix M and I can evaluate M^-1 by solving
# explicitly the linear algebra problem or exploiting solve_banded

for t in range(ntstep+1): #how many step you have to make in time
    coeff = solve_banded((1,1),mb, -np.dot(k,al)+fvalues(t*dt))
        # if t0 != 0 you should put (t0+t)*dt

#--------------------------------------------
# alternative version using the normal linear algebra solver, that solves a
# problem of type ax = b finding x

# coeff=np.linalg.solve(m,-np.dot(k,al)+fvalues(t*dt))
#--------------------------------------------
    
    al=al+dt*coeff
    
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


