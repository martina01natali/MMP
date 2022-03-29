# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:37:41 2021

@author: DRAGO
"""

#
# first we solve the equation analytically by using sympy
#



import numpy as np
import matplotlib.pyplot as plt



def f1(x):
    return x*(1-x)

def f2(x):
    return 1-x

#we compute the Fourier components of f1 and f2

from scipy.integrate import quad

def base(x,n):
    return np.sin(n*np.pi*x)

def fun1(x,n):
    return f1(x)*base(x,n)

def fun2(x,n):
    return f2(x)*base(x,n)


def fourierf1(n):
    return 2*quad(fun1,0,1,args=(n))[0]

def fourierf2(n):
    return 2*quad(fun2,0,1,args=(n))[0]

# these are the coefficient of the expansion of the functions, 
# those for f1 scale as 1/n**3, those of f2 scale as 1/n
    



#here we approximate f1 and f2
def approxf1(x,n):
    approxf1=0
    for j in range(1,n+1):
        approxf1 += base(x,j)*fourierf1(j)
    return approxf1

def approxf2(x,n):
    approxf2=0
    for j in range(1,n+1):
        approxf2 += base(x,j)*fourierf2(j)
    return approxf2


#here we prepare the arrays to plot the rhs and its approximations
x = np.linspace(0,1,200)
vec_approxf1=np.vectorize(approxf1)
vec_approxf2=np.vectorize(approxf2)
vec_f1=np.vectorize(f1)
vec_f2=np.vectorize(f2)

f1_10=vec_approxf1(x,10)
f1_20=vec_approxf1(x,20)
f1_40=vec_approxf1(x,40)
f1_80=vec_approxf1(x,80)
f1_exact=vec_f1(x)

f2_10=vec_approxf2(x,10)
f2_20=vec_approxf2(x,20)
f2_40=vec_approxf2(x,40)
f2_80=vec_approxf2(x,80)
f2_exact=vec_f2(x)
        
    
difff1_10=np.log10(np.abs(f1_exact-f1_10))
difff1_20=np.log10(np.abs(f1_exact-f1_20))
difff1_40=np.log10(np.abs(f1_exact-f1_40))
difff1_80=np.log10(np.abs(f1_exact-f1_80))

plt.axis([0,1,-8,0])
plt.plot(x,difff1_10,'-', label='approx10')
plt.plot(x,difff1_20,'-', label='approx20')
plt.plot(x,difff1_40,'-', label='approx40')
plt.plot(x,difff1_80,'-', label='approx80')
plt.legend()
plt.show()

plt.axis([0,1,0,1.5])
plt.plot(x,f2_10,label='approx10')
plt.plot(x,f2_exact)
plt.legend()
plt.show()

plt.axis([0,1,0,1.5])
plt.plot(x,f2_20,label='approx20')
plt.plot(x,f2_exact)
plt.legend()
plt.show()

plt.axis([0,1,0,1.5])
plt.plot(x,f2_40,label='approx40')
plt.plot(x,f2_exact)
plt.legend()
plt.show()

plt.axis([0,1,0,1.5])
plt.plot(x,f2_80,label='approx80')
plt.plot(x,f2_exact)
plt.legend()
plt.show()

#plt.plot(x,difff1_20,'-', label='approx20')
#plt.plot(x,difff1_40,'-', label='approx40')
#plt.plot(x,difff1_80,'-', label='approx80')
#plt.legend()
#plt.show()

#plt.plot(x,difff1_20,'-', label='approx20')
#plt.legend()
#plt.show()

#plt.axis([0,1,-6,0])
#plt.plot(x,difff1_40,'-', label='approx40')
#plt.legend()
#plt.show()

#plt.axis([0,1,-6,0])
#plt.plot(x,difff1_80,'-', label='approx80')
#plt.legend()
#plt.show()
   





#diff2=y-y2
#diff4=y-y4
#plt.axis([0,l,-0.01,0.01])
#plt.plot(x,diff2,'-',label='diff2')
#plt.plot(x,diff4,'-',label='diff4')
#plt.legend()
#plt.show()







