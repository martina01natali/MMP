# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:31:46 2020

@author: Drago
"""
import numpy as np
import matplotlib.pyplot as plt


e1 = 0.01
e2 = 0.2
r = 0.2 
q = 0.4
x10 = 40
x20 = 500
dt = 0.05
nstep = 1000
x0 = np.array([x10,x20])


sol=[]
for i in range(nstep):
    sol.append([i,0,0])
    

     
def f(x1,x2):
    f1=e1*e2*x1*x2-q*x1
    f2=r*x2-e1*x1*x2
    fa=np.array([f1,f2])
    return fa


x=x0



for i in range(nstep):
    k1=f(x[0],x[1])
    k2=f(x[0] + dt*k1[0]/2,x[1]+dt*k1[1]/2)
    k3=f(x[0] + dt*k2[0]/2,x[1]+dt*k2[1]/2)
    k4=f(x[0] + dt*k3[0],x[1]+dt*k3[1])
    x0 = x
    x = x0 + (k1 + 2*k2 + 2*k3 + k4)*dt/6
    sol[i]=[i*dt,x[0],x[1]]
    

    
import matplotlib.pyplot as plt

def plotex(sol0,sol1,sol2):
#    plt.xlabel('x', fontsize=20)
#    plt.ylabel('f(x)', fontsize=20)
    plt.plot(sol0, sol1, 'r-', label='predators')
    plt.plot(sol0, sol2, 'b-', label='preys')
    plt.legend()
    plt.show()
    
def plotex1(sol1,sol2):
    plt.xlabel('predators', fontsize=20)
    plt.ylabel('preys', fontsize=20)
    plt.plot(sol1, sol2, 'r-')
#    plt.legend()
    plt.show()
    
sol0=[]
for i in range(nstep):
    sol0.append(sol[i][0])
    
sol1=[]
for i in range(nstep):
    sol1.append(sol[i][1])
    
sol2=[]
for i in range(nstep):
    sol2.append(sol[i][2])

plotex(sol0, sol1, sol2)

plt.figure()

plotex1(sol1, sol2)
    
    