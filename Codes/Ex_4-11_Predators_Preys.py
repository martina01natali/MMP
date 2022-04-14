# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:15:16 2020

@author: MARTINA
"""

#Predators and preys

import numpy as np
import matplotlib.pyplot as plt

"""
-------------DESCRIPTION-------------
Predators and preys is a system of 2 non-linear DE with 4 arbitrary parameters:
dx1/dt = e2*e1*x1*x2 - q*x1
dx2/dt = r*x2 - e1*x1*x2
and initial conditions x1(0)=x10, x2(0)=x20.
By considering a vector x of components x1 and x2 we can recast the system as a
I order equation dx/dt=f(t,x) with initial values, so an IVP.
The idea to solve it by arbitrarly fix the parameters, try different initial
conditions and see what happens, by applying RK methods to approximate the
integral.
-------------------------------------
"""

e1 = 0.01   #predators' "eating" coefficient
e2 = 0.2    #predators' "reproduction" coefficient
r = 0.2     #preys' reproduction coefficient
q = 0.4     #predators' dying coefficient
x10 = 20    #initial condition on x1
x20 = 500   #initial condition on x2
dt = 0.05   #time steps
nstep = 750
x0 = np.array([x10,x20],dtype=np.float64)    

"""
creates a Numpy array with two entries, that are the right hand side of the
system. The variation of the type is necessary to produce homogeneous arrays.
"""

def f(x1,x2):   #defining the system of 2 equations as an
                #arbitrary defined (def) function
    f1 = e1*e2*x1*x2-q*x1 #define 1st equation = rate x1
    f2 = r*x2-e1*x1*x2    #define 2nd equation = rate x2
    fa = np.array([f1,f2])    #system is built as array
    return fa   #this is the result of the function f(x1,x2),
                #the array cointaining the values of f1 and f2
                #for given x1, x2
   
x=x0 #so x[0]=x1 and x[1]=x2
sol = []
"""
Set initial value of x = array of 2 initial values using x (size=2) to speed
up the process and we access the initial conditions of x1 and x2 by indexing,
x[0] and x[1]. We also create an empty list to store the solutions, that will
be a list of lists of three elements: one temporal component and the values of
the two solutions.

We use the Runge-Kutta method to find the values of the solution as it evolves
in time. The only difference with RK4 example is that here the variable is a
vector of size 2 and the function is a vector too.
"""

for i in range(nstep):
    k1=f(x[0],x[1])
    k2=f(x[0]+dt*k1[0]/2,x[1]+dt*k1[1]/2)
    k3=f(x[0]+dt*k2[0]/2,x[1]+dt*k2[1]/2)
    k4=f(x[0]+dt*k3[0]/2,x[1]+dt*k3[1]/2)
    x += (k1+2*k2+2*k3+k4)*dt/6
    sol.append([i*dt,x[0],x[1]]) #sol will be an nstep-by-3 matrix

"""    
We have to make the plot with the array of the x values and y values,
and all of this is contained inside sol, but there the values depend
on time. So I plot both solutions as functions of time.
"""

def plotex(sol0,sol1,sol2):
    plt.plot(sol0,sol1,'r-',label='Predators')
    plt.plot(sol0,sol2,'b-',label='Preys')
    plt.legend()
    plt.show()

# It is interesting to see what happens by plotting x1 vs x2, so the preys' and
# predators' populations one as a function of the other. To do this, you can
# simply make a scatterplot, since they both depend on time.

def plotex1(sol1,sol2):
    plt.plot(sol1,sol2, 'r-')
    plt.xlabel('Predators', fontsize=15)
    plt.ylabel('Preys', fontsize=15)
    plt.show()
    
solMatrix = np.array(sol) # it was a list
solMatrix = np.around(solMatrix, decimals=3)

sol0 = solMatrix[:,0] # time column
sol1 = solMatrix[:,1] # x1 column
sol2 = solMatrix[:,2] # x2 column

plt.figure()
plotex(sol0, sol1, sol2)
plotex1(sol1, sol2)
