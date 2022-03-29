# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:43:48 2021

@author: Drago
"""
# first we solve the equation analytically by using sympy

from sympy import *

from sympy.abc import x, t, r , g, l

import numpy as np
import matplotlib.pyplot as plt

f=Function('f')
sol=dsolve(Eq(t*Derivative(f(x),x,x),r*g),f(x), ics={f(0):0, f(l):0})
print(sol)

def fun(x):
    fun=-g*l*r*x/(2*t) + g*r*x**2/(2*t)
    return fun

g=980.
r=0.2
l=50.
t=10**6

vfun = np.vectorize(fun)

x = np.linspace(0,l,200)
y = vfun(x)

plt.axis([0,l,-0.1,0])
plt.plot(x,y,label='exact')
plt.legend()
plt.show()

########################################################

#we compute the Fourier transform of the equation

from scipy.integrate import quad

def fun1(x,n):
    return -r*g*np.sin(n*np.pi*x/l)/(l/2)


def fouriercn(n):
    return quad(fun1,0,l,args=(n))[0]

# these are the coefficient of the expansion of the solution, 
# they scale as 1/n**3
    

def fourieran(n):
    return fouriercn(n)*l**2/(t*n**2*np.pi**2)


# if  needed one can compute those coeffs as elements of an array
#vec_fouriercn=np.vectorize(fouriercn)
#res=vec_fouriercn(np.arange(1,100,1))
#print(res)

#vec_fourieran=np.vectorize(fourieran)
#res=vec_fourieran(np.arange(1,100,1))


#this is the rhs, it is just a constant
def rhs(x):
    return -r*g


#here we approximate the rhs, 10 and 50 terms
def approxrhs10(x):
    approxrhs10=0
    for j in range(1,10):
        approxrhs10 += np.sin(j*np.pi*x/l)*fouriercn(j)
    return approxrhs10

def approxrhs50(x):
    approxrhs50=0
    for j in range(1,50):
        approxrhs50 += np.sin(j*np.pi*x/l)*fouriercn(j)
    return approxrhs50

#here we prepare the arrays to plot the rhs and its approximations
x = np.linspace(0,l,200)
vec_approxrhs10=np.vectorize(approxrhs10)
vec_approxrhs50=np.vectorize(approxrhs50)
vec_rhs=np.vectorize(rhs)
y10=vec_approxrhs10(x)
y50=vec_approxrhs50(x)
z=vec_rhs(x)
        
    
plt.axis([0,l,-250,0])
plt.plot(x,y10,'-', label='approx10')
plt.plot(x,y50,'-', label='approx50')
plt.plot(x,z, label='exact')
plt.legend()
plt.show()
   

#############################################

#here we approximate the solution, recall that range(1,5) means (1,2,3,4)
def sol2(x):
    sol2=0
    for j in range(1,2):
        sol2 += np.sin(j*np.pi*x/l)*fourieran(j)
    return sol2
def sol4(x):
    sol4=0
    for j in range(1,4):
        sol4 += np.sin(j*np.pi*x/l)*fourieran(j)
    return sol4

x = np.linspace(0,l,200)
vec_sol2=np.vectorize(sol2)
vec_sol4=np.vectorize(sol4)
y2=vec_sol2(x)
y4=vec_sol4(x)
z=vec_rhs(x)
        
    
plt.axis([0,l,-0.1,0])
plt.plot(x,y2,'-', label='sol2')
plt.plot(x,y4,'-', label='sol4')
plt.plot(x,y, label='exact')
plt.legend()
plt.show()

diff2=y-y2
diff4=y-y4
plt.axis([0,l,-0.01,0.01])
plt.plot(x,diff2,'-',label='diff2')
plt.plot(x,diff4,'-',label='diff4')
plt.legend()
plt.show()