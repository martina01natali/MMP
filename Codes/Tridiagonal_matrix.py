# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:35:42 2020

@author: Drago
""" 
import numpy as np
 
# These routines allows you to speed up the process of writing trigiagonal
# matrixes by filling the diagonals only
 
def tridiag(a, b, c, k1=-1, k2=0, k3=1): #k are the offsets of the diagonals
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
 
# np.diag returns a matrix made of diagonal lines of numbers a, b, c and with
# an offset k with respect to the main diagonal; all other entries are filled
# with zeros. The size of the matrix is defined by the size of the lists
# of numbers passed to diag as arguments.
 
a = [1, 1]; b = [2, 2, 2]; c = [3, 3]
A = tridiag(a, b, c) 
print(A)

#%%
# The following routine allows to make the process of creating the lists
# of entries automatic, by creating first an array of ones and then 
# multiplying by the numbers that you want. Remember that numpy arrays and
# matrixes can be trated as geometrical ones, see last operation below:
# result is a matrix which entries are the sum of the entries.
# n is the dimension of the matrix and only the main diagonal has length n
 
def tridiagmod(a, b, c, n):
    aa=a*np.ones(n-1)
    bb=b*np.ones(n)
    cc=c*np.ones(n-1)
    return np.diag(aa, -1) + np.diag(bb) + np.diag(cc, 1)
Anew = tridiagmod(1, 2, 3, 5)
print(Anew)

#%%
from scipy.linalg import eigh_tridiagonal

# We now use scientific python, that contains very specific routines for
# dealing with tridiagonal matrixes specifically and it is much faster.
# tridiagonal deals with symmetric, tridiagonal matrixes

# This pre-made routine computes eigenvalues and eigenfunctions of a 
# tri-diagonal matrix

d = 3*np.ones(4)
e = -1*np.ones(3)
w, v = eigh_tridiagonal(d, e)
A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
print(A)
print(w)
print(v)