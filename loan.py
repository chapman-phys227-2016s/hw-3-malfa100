"""
File: loan.py
Copyright (c) 2016 Andrew Malfavon
Excerise A.4
License: MIT
Calculates the payback of a loan based on the loan amount, interest rate, and number of months.
"""

import numpy as np

#y represents the payment each month. x represents the value of the loan.
#creates an array for each showing their values at each month.
def loan(L, p, N):
    y = np.zeros(N + 1)
    x = np.zeros(N + 1)
    y[0] = 0
    x[0] = L
    for n in range(1, N + 1):
        y[n] = ((p / (12.0 * 100.0)) * x[n-1]) + (L / float(N))
        x[n] = x[n-1] + ((p / (12.0 * 100.0)) * x[n-1]) -y[n]
    return y, x

def test_loan():
    assert round(loan(100, 5, 12)[0][0]) == 0.0
    assert round(loan(100, 5, 12)[0][12], 2) == 8.37
    assert round(loan(100, 5, 12)[1][0]) == 100.00
    assert round(loan(100, 5, 12)[1][12]) == 0.0