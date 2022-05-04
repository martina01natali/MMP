# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:03:34 2022

@author: MARTINA
"""

# Following the official sympy documentation, I will build this playfield
# to be able to perform a large set of numerical and analytical operations
# by exploiting the analytical construction of functions and equations of any
# sort that is provided by the use of sympy, with the goal of implementing
# some of its features in the bigger project of Mathematical Methods.

# Main ref.: https://docs.sympy.org/latest/index.html

        #***************************************************#
# %%
from sympy import *
init_printing(use_unicode=True)

# Basic operations
# Following https://docs.sympy.org/latest/tutorial/basic_operations.html
print("#***************** BASIC OPERATIONS ********************#")

# Building variables
x = symbols('x')
print(x)

# Building expressions
psi = 5-1/5*abs(x-25)
print(psi)

# Sunstituting a value inside an expression
# Notice that this syntax works as well with any variable, so one can subs
# any variable y to x to obtain an expression that depends on y, instead of x,
# or one can also pass a list of tuples to substitute multiple variables
# all at once.
# Also notice that subs does not perform in-place substitution.
print(psi.subs([(x, 2)]))

# sympify converts a string expression to a sympy expression
psi_str = "5-1/5*abs(x-25)"
psi_sym = sympify(psi_str)
# To do the opposite, you can recast an expression as you would normally do
# Notice how the sympy-born expression and this below differ from each other
print(str(psi_sym))

# evalf evaluates a sympy expression:
# in the case in which you want to evaluate an expression with variables
# for given values of those variables, you may use subs inside evalf
# In the case in which your expression is a numerical expression (say sin(1))
# then the first argument of evalf is the number of decimals that you will get
# in the displayed result.
# Also evalf can restitute only 1 significative number by setting
# the chop flag, chop=True
psi_sym.evalf(subs={x: 2})

# lambdify is a lambda function that converts sympy expressions to other
# numerical modules' expressions, such as numpy and scipy
psi_np = lambdify(x, psi_sym, "numpy")
print(psi_np(2))
psi_np = lambdify(x, psi_sym, "scipy")
print(psi_np(2))

        #***************************************************#        
# %%

# Printing
# https://docs.sympy.org/latest/tutorial/printing.html

# The default setup() function in sympy is
# init_session()
# that performs a series of commands, imports everything from the library
# and also gives an idea of how variables and functions are built in sympy.

from sympy import init_session
init_session()
print("#***************** PRINTING ********************#")

# Check if LaTeX is installed and recognized by python by running the
# following line in the interactive line
# Integral(sqrt(1/x), x)

# >>> LaTeX is indeed installed and recognized, and sympy prints pretty

# You can get the LaTeX expression (with LaTeX commands) of an expression
# (I beg your pardon) with the latex() command
print(latex(Integral(sqrt(1/x), x)))

# This is an important note: there is no way, as I see at the moment,
# to make this script print pretty expressions in LaTeX rendering
# directly from the script, it can only be made possible by using
# the interactive console. Nevertheless, you can print out things in
# unicode rendering by using pprint() with option use_unicode
# set to True.
pprint(Integral(exp(x**2),x), use_unicode=True)

        #***************************************************#
# %%

# Simplification
# https://docs.sympy.org/latest/tutorial/simplification.html

from sympy import init_session
init_session()
print("#***************** SIMPLIFICATION ********************#")
print("#***************** WIP ********************#")

        #***************************************************#
# %%

# Calculus
# https://docs.sympy.org/latest/tutorial/calculus.html

from sympy import init_session
init_session()
print("\n#*************", "CALCULUS", "*************#\n")
print("\n#*************", "Derivatives", "*************#\n")

# Derivatives are computed with the diff function, and are restituded as
# analytical expressions.
# Higher order derivatives are computed by passing the variable as many
# times as the order of derivation
print(diff(sin(x)))
print(diff(sin(x), x)) # equivalent to above since single variable
print(diff(sin(x), x, x))

# diff is also supported as a method of expr objects of sympy, with arguments
# the variable and the order of derivation
expr = exp(-x**2)
print(expr.diff(x, 1))

# Objects are also supported via Derivative class
deriv = Derivative(expr, x, 1)
print(deriv)
# To evaluate the derivative object, use the method doit()
print(deriv.doit())

# if you don't see the derivative "pretty printed", initialize printing via
# init_printing(use_unicode=True) doesn't work, and I don't know what else
# could

print("\n#*************", "Integrals", "*************#\n")
# The easiest is integrate(), you may pass the expression as an argument
print(integrate(cos(x), x))
print(integrate(cos(x), (x, 0, 1)))

# Unevaluated integral objects can also be made with the Integral class
integ = Integral(cos(x), x)
print(integ)
print(integ.doit())

print("\n#*************", "Limits[WIP]", "*************#\n")
# Limits can also be computed

print("\n#*************", "Series expansion", "*************#\n")
# Performs standard Taylor - Mc Laurin series expansion
print((sin(x).series(x, 0, 4)))

print("\n#*************", "Finite differences[WIP]", "*************#\n")
# https://docs.sympy.org/latest/tutorial/calculus.html#finite-differences

# %%
from sympy import init_session
init_session()
# init_printing(pretty_print=True, use_unicode=False, use_latex=True)


print("\n#*************", "SOLVERS", "*************#\n")

# Equation objects may be the first thing we want to build and solve
# The arguments passed to the Eq constructor are the lhs and the rhs of the
# equation
print(Eq(x, y))

# solveset() is the first solver that we see, and it interprets any expression
# as an equation in which the expression that is passed is equal to 0,
# so it only requires the variable that we have wrt we have to solve the
# equation
print(solveset(x**2-1, x))

# solveset is a solver for algebraic expressions, and takes as arguments
# the expression of the equation, the variable to find an explicit form for,
# and the domain in which we are working, such as S.Reals, S.Complexes.
# It returns a FiniteSet, Interval, ImageSet, EmptySet for the solutions.
print(solveset(cos(x)-x, x))

pprint(Integral(x, x), use_latex=True)
# Solve linear systems with linsolve
