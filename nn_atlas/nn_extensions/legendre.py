import sympy as sp
import itertools
from functools import reduce
import torch.nn as nn
import torch
import operator
import numpy as np


def sympy_to_torch(expr, substitutions):
    '''Mimimal translator that handles operators in Legendre polyn. expressions'''
    # Terminals
    if not expr.args:
        if expr in substitutions:
            return substitutions[expr]

        if expr.is_number:
            return float(expr)

        raise ValueError(str(expr))

    rule = {sp.Add: torch.add,
            sp.Mul: torch.mul,
            sp.Pow: torch.pow} 
    # Compounds. NOTE that sympy operands can have many args wherase
    # the above ufl guys have only two at most
    args = tuple(sympy_to_torch(arg, substitutions) for arg in expr.args)
    if len(args) < 3:
        return rule[type(expr)](*args)
    # So we get the first arg and ask for the other
    # (+ 3 4 5) -> (+ 3 (+ 3 4))
    head, tail = args[0], args[1:]
    return rule[type(expr)](head, sympy_to_torch(type(expr)(*expr.args[1:]), substitutions))


def legval(x, c):
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x


def legendre(degree, var):
    '''UFL Legendre 1d polynomial of given degree'''
    var_ = sp.Symbol('x')
    l = sp.legendre(degree, var_)
    # Translated from one symbolic representation to another
    return sympy_to_torch(l, {var_: var})


class LegendreNetwork1D(nn.Module):
    def __init__(self, degree):
        super().__init__()
 
        self.lin = nn.Linear(degree+1, 1, bias=False)
        self.degree = degree
        self.cache = None
 
    def forward(self, x):

        bsize, npts  = x.shape[:2]
        legendre_values = torch.zeros(bsize, npts, self.degree+1,
                                      dtype=x.dtype)

        coefs = np.zeros(self.degree + 1)
        var = x
        bcs = (1.0 - var**2)
        for col, d in enumerate(range(self.degree+1)):
            # legendre_values[..., col] = legendre(d, var)*bcs
            coefs = np.zeros(d+1)
            coefs[d] = 1.0
            legendre_values[..., col] = legval(x, coefs)*bcs

        self.cache = legendre_values
            
        return self.lin(legendre_values).squeeze(2)

from nn_atlas.nn_extensions.calculus import grad

deg = 20
nn_trial = LegendreNetwork1D(deg)
nn_trial.double()

nn_test = LegendreNetwork1D(deg)
nn_test.double()

u_true = lambda x: torch.sin(2*np.pi*x)
f_true = lambda x: (2*np.pi)**2*torch.sin(2*np.pi*x)

A = np.zeros((deg+1, deg+1))
b = np.zeros(deg+1)

trial_coefs = torch.zeros_like(nn_trial.lin.weight[0])
test_coefs = torch.zeros_like(trial_coefs)

xq, wq = np.polynomial.legendre.leggauss(deg+1)
xq = xq.reshape((-1, ))

xq, wq = torch.tensor([xq]), torch.tensor([wq])

nn_trial(xq)

bsize, npts  = xq.shape[:2]
legendre_values = torch.zeros(bsize, npts, deg+1, dtype=xq.dtype)
        
xq.requires_grad = True

f_ = f_true(xq)

for i in range(deg+1):
    trial_coefs[i] = 1.
    with torch.no_grad():
        nn_trial.lin.weight[0] = trial_coefs

    u_ = nn_trial(xq)
    grad_u = grad(u_, xq)
    A[i, i] = torch.sum(grad_u*grad_u*wq)

    b[i] = torch.sum(wq*u_*f_)
    
    for j in range(i+1, deg+1):
        test_coefs[j] = 1.
        with torch.no_grad():
            nn_test.lin.weight[0] = test_coefs
        
        grad_v = grad(nn_test(xq), xq)
        # Reset
        test_coefs[j] = 0.

        value = torch.sum(grad_u*grad_v*wq)
        A[i, j] = value
        A[j, i] = value
    # Reset
    trial_coefs[i] = 0.


weight = np.linalg.solve(A, b)
with torch.no_grad():
    nn_trial.lin.weight[0] = torch.tensor([weight])


import matplotlib.pyplot as plt

xq = torch.tensor([np.linspace(-1, 1, 1000)])

plt.figure()
plt.plot(xq.detach().numpy().ravel(),
         nn_trial(xq).detach().numpy().ravel(),
         label='NN')

plt.plot(xq.detach().numpy().ravel(),
         u_true(xq).detach().numpy().ravel(),
         marker='x', linestyle='none',
         label='u')

#plt.plot(xq.detach().numpy().ravel(),
#         f_true(xq).detach().numpy().ravel(),
#         label='f')

plt.legend()
plt.show()
