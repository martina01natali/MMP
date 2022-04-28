# Author: Alex Gezerlis
# Numerical Methods in Physics with Python (CUP, 2020)

# How do you evaluate the derivative of a function numerically?
# One can use the definition of derivative by using the incremental ratio.
# We compute the right derivative: incremental ratio between f and f+h.
# We evaluate the value of the derivative by expanding in Taylor series
# the function: we get that this method introduces an error of the order
# of h/2*f''. Same goes for the left hand derivative (bet f-h and f).

from math import exp, sin, cos, log10


def f(x):
    return exp(sin(2*x))


def fprime(x):
    return 2*exp(sin(2*x))*cos(2*x)


def calc_fd(f, x, h):  # function to compute right derivative
    fd = (f(x+h) - f(x))/h
    return fd

# Then I define a function to compute the central derivative.
# Here, the error scales as h^3, that gives a smaller error than the right or
# left derivatives.


def calc_cd(f, x, h):
    cd = (f(x+h/2) - f(x-h/2))/h
    return cd

# I try to see what happens if I decrease the size of the step and I print the
# absolute value of the difference between the central derivative and the
# exact value.


if __name__ == '__main__':
    x = 0.5
    an = fprime(x)  # exact value

    hs = [10**(-i) for i in range(1, 12)]
    fds = [abs(calc_fd(f, x, h) - an) for h in hs]
    cds = [abs(calc_cd(f, x, h) - an) for h in hs]

    rowf = "{0:1.0e} {1:1.16f} {2:1.16f}"
    # first number is in the exponential notation, with 0 digits after the dot
    # 2nd, 3rd number are written in scientific notation with 16 digits
    print("h     abs. error in fd   abs. error in cd")
    for h, fd, cd in zip(hs, fds, cds):
        print(rowf.format(h, fd, cd))
