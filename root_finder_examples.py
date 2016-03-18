"""
File: root_finder_examples.py
Copyright (c) 2016 Andrew Malfavon
Excerise A.11
License: MIT
Approximates f(x) = 0 using Newton's Method, the Bisection Method, and the Secant Method.
Demonstrates 7 example for f(x).
The code for each method is based off of the examples in the book with some modifications.
"""

import numpy as np
import matplotlib.pyplot as plt

#newton's method takes in a function, its derivative, and an intial x-value.
def newtons_method(f, fprime, x_0, eps = 1.0E-7, N = 100):
    n = 1
    x = np.zeros(N + 1)
    x[0] = x_0
    while abs(f(x[n - 1])) > eps and n <= N:
        if abs(fprime(x[n - 1])) < 1E-14:
            raise ValueError('divide by zero')
        x[n] = x[n - 1] - (f(x[n - 1]) / fprime(float(x[n - 1])))
        n += 1
    return x[:n]

#bisection method takes in a function and a lower and upper x-bound.
def bisection_method(f, a, b, eps = 1.0E-5):
    if f(a) * f(b) > 0:
        return None
    x = []
    while b - a > eps:
        midpoint = (a + b) / 2.0
        if f(midpoint) * f(a) <= 0:
            b = midpoint
        else:
            a = midpoint
        x.append(midpoint)
    return x

#secant method takes in a function and an intial x-value and a second x-value.
def secant_method(f, x_0, x_1, eps = 1.0E-7, N = 100):
    n = 2.0
    x = np.zeros(N + 1)
    x[0] = x_0
    x[1] = x_1
    while abs(f(x[n - 1]) - f(x[n - 2])) > eps and n <= N:
        x[n] = x[n - 1] - ((f(x[n - 1]) * (x[n-1] - x[n - 2])) / (f(float(x[n - 1])) - f(float(x[n - 2]))))
        n += 1
    return x[:n]

#used to graph the functions in order to determine appropriate initial conditions
def graph(f, xmin, xmax, ymin, ymax, n = 100):
    x_values = np.linspace(xmin, xmax, n)
    y_values = f(x_values)
    plt.plot(x_values, y_values)
    plt.title('f(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

#functions and their derivatives
def func1(x):
    return np.sin(x)
def func2(x):
    return x - np.sin(x)
def func3(x):
    return x**5 - np.sin(x)
def func4(x):
    return (x**4) * np.sin(x)
def func5(x):
    return x**4 - 16
def func6(x):
    return x**10 - 1
def func7(x):
    return np.tanh(x) - x**10
def funcp1(x):
    return np.cos(x)
def funcp2(x):
    return 1 - np.cos(x)
def funcp3(x):
    return (5 * x**4) - np.cos(x)
def funcp4(x):
    return (4 * x**3) * np.sin(x) + (x**4) * np.cos(x)
def funcp5(x):
    return 4 * x**3
def funcp6(x):
    return 10 * x**9
def funcp7(x):
    return ((4 * (np.cosh(x))**2) / ((1 + np.cosh(2 * x))**2)) - (10 * x**9)

#tests using the fifth function.
def test_newtons_method():
    assert round(newtons_method(func5, funcp5, 1.0)[-1]) == 2.0
def test_bisection_method():
    assert round(bisection_method(func5, 1.0, 3.0)[-1]) == 2.0
def test_secant_method():
    assert secant_method(func5, 0.5, 1.5)[-1] == 2.0