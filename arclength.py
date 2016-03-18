"""
File: arclength.py
Copyright (c) 2016 Andrew Malfavon
Excerise A.13
License: MIT
Calculate the arclength of a curve using an integral
"""

import numpy as np
import matplotlib.pyplot as plt

#using a difference equation(as explained in the book) to calculate the integral
def arclength(f, a, b, n):
    index_set = range(n + 1)
    b = np.linspace(a, b, n + 1)
    f_ = np.zeros_like(b)
    s = np.zeros_like(b)
    f_[0] = f(b[0])
    s[0] = 0
    h=1E-5
    for i in index_set[1:]:
        f_[i] = np.sqrt(((f(b[i] + h) - f(b[i] - h)) / (2 * float(h)))**2 + 1)
        s[i] = s[i - 1] + 0.5 * (b[i] - b[i - 1]) * (f_[i - 1] + f_[i])
    return s, f

#s and f will be graphed on the same graph
def graph(s, f, xmin, xmax, ymin, ymax, n=10000):
    sx_values = np.linspace(xmin, xmax, n)
    sy_values = s[1:]
    plt.plot(sx_values, sy_values)
    fx_values = np.linspace(xmin, xmax, n)
    fy_values = f(fx_values)
    plt.plot(fx_values, fy_values)
    plt.title('f(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

#test arclength of a straight line
def func_line(x):
    return (3 * x) / 4

#test arclength of a semi-circle
def func_semicircle(x):
    return np.sqrt(1.0001-x**2)

def test_arclength():
    assert round(arclength(func_line, 0, 4, 10000)[0][-1]) == 5.0
    assert round(arclength(func_semicircle, 0, 1, 100000)[0][-1], 1) == round((np.pi) / 2, 1)