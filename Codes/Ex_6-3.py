# -*- coding: utf-8 -*-

        #***************************************************#

# Example 6.3 - Inhomogeneous boundary conditions
# Gockenbach, chap. 6.1.4

# Inhomogeneous boundary conditions can be handled by "shifting" the data as
# we have done in previous examples of steady-state (time-independent)
# problems.

# Suppose the iron bar of Example 6.1 is heated at a constant temperature of
# 4 °C and that at one end (x = 0) is placed in an ice bath (0 °C), while the
# other end is mantained at 4 °C. What is the temperature distribution after
# 5 minutes (300 s)?

# At first we define an ausiliary function p(x) that satisfies the boundary
# conditions: p(x) = 4x/50, that is given by the dimensionless quantity
# x/l multiplied by the temperature that we need to satisfy the further-end
# boundary condition. Having p(x), we rewrite our IBVP with respect to the
# function v(x,t) = u(x,t) - p(x). This function satisfies homogeneous Dir.
# boundary conditions. Having defined v in this way, the function that
# provides the initial condition becomes v(x,0) = 4-4*x/50.

        #***************************************************#

import numpy as np
import matplotlib.pyplot as plt  
from scipy.integrate import quad

# Given values of the parameters (from Example 6.1)
c=0.437
r=7.88
k=0.836

# Initial condition function
def psi(x):
    return 4-4*x/50

# Elements of the Fourier sine basis
def phi(x,n):
    return np.sin(n*np.pi*x/50)

# Integrand function for computing coefficients of the expansion of the
# initial condition function
def psi1(x,n):
    return psi(x)*phi(x,n)

# Initial condition projecting psi(x) on the elements of the basis
nmax=100
b = np.array([(2/50)*quad(psi1,0,50,args=(n))[0] for n in range(nmax)])
print(b)

#%%

# Coefficients a_n of the expansion of the solution v(x,t).
# The fact that these coefficients depend on time is peculiar and solely
# due to the time-dependance of the original IBVP.
def coeff(t,n):
    return np.exp(-k*n**2*np.pi**2*t/(50**2*r*c))

# The solution is v(x,t) that is the solution of the dimensionless version
# of the equation of diffusion. To obtain the solution of the normal eq we have
# to sum together v(x,t) and the dimension-normalization p(x) = 4x/50
def sol(x,t,ntop):
    sol=0
    for i in range(ntop):
        sol += + b[i]*coeff(t,i)*phi(x,i)
    sol += 4*x/50
    return sol

vsol=np.vectorize(sol)
t = np.linspace(0, 50, 101)

#%%
plt.figure(1)
s20=vsol(t, 20, 30)
s40=vsol(t, 40, 30)
s60=vsol(t, 60, 30)
plt.plot(t, s20)
plt.plot(t, s40)
plt.plot(t, s60) 
plt.show() 

#%%
plt.figure(2)
s202=vsol(t, 20, 2)
s206=vsol(t, 20, 6)
s2010=vsol(t, 20, 10)
plt.plot(t, s202, color="pink")
plt.plot(t, s206, color="red")
plt.plot(t, s2010, color="brown")
plt.plot(t, s20, color="black")
plt.show()

# Comment the meaning of every order of approx