"""
File: sin_Taylor_series_diffeq.py
Copyright (c) 2016 Andrew Malfavon
Excerise A.14
License: MIT
Computes a Taylor polynomial approximation for sinx and shows the accuracy in a table
"""
import numpy as np

#computes the approximation based on the exercises in the book
def sin_Taylor(x, n):
    i = 1
    an_prev = x
    sn_prev = 0.0
    while i <= n + 1:
        sn = sn_prev + an_prev
        an = (-x**2 * an_prev) / (((2 * i) + 1) * (2 * i))
        sn_prev = sn
        an_prev = an
        i += 1
    return abs(an), sn

#test for small x=0.0001 and n = 2, sin(x) is zero to the third decimal
def test_sin_Taylor():
    assert round(sin_Taylor(0.0001, 2)[1], 3) == 0.0


#create a table to display the approximations for various n and x
def table():
    print '%15s %15s %20s %8s' % ('x Values:', 'n Values:', 'Aproximations:', 'Exact:')
    x_list = [0.01, np.pi/2, np.pi, 3*np.pi/2]
    n_list = [2, 5, 10, 100]
    for x in x_list:
        for n in n_list:
            table = sin_Taylor(x, n)
            print '%15f %15f %15f %15f' % (x, n, table[1], np.sin(x))