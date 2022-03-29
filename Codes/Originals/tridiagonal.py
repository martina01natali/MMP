# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:35:42 2020

@author: Drago
"""
import numpy as np

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

a = [1, 1]; b = [2, 2, 2]; c = [3, 3]
A = tridiag(a, b, c)

print(A)

def tridiagmod(a, b, c, n):
    ym1=np.ones(n-1)
    aa=a*ym1
    y=np.ones(n)
    bb=b*y
    cc=c*ym1
    return np.diag(aa, -1) + np.diag(bb) + np.diag(cc, 1)


Anew = tridiagmod(1, 2, 3, 5)

print(Anew)

from scipy.linalg import eigh_tridiagonal
d = 3*np.ones(4)
e = -1*np.ones(3)
w, v = eigh_tridiagonal(d, e)
A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
print(A)
print(w)
print(v)