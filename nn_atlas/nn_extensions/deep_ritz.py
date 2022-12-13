from p1net import ScalarP1FunctionSpace
from nn_atlas.nn_extensions.calculus import grad
import dolfin as df
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


mesh = df.UnitSquareMesh(5, 5)

u = ScalarP1FunctionSpace(mesh, invalidate_cache=True)
u.double()

x = torch.rand(1, 200, 2, dtype=torch.float64)
x.requires_grad = True

f = x[..., 0] + x[..., 1]

# -----------------------------------------------------------------
if True:
    parameters = [u.lin.weight]
    
    maxiter = 100
    optimizer = optim.LBFGS(parameters, max_iter=maxiter,
                            history_size=1000, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    epoch_loss = [np.nan]
    def closure(history=epoch_loss):
        optimizer.zero_grad()

        ux = u(x)
        loss = torch.mean(0.5*torch.sum(grad(ux, x)**2, axis=2) + 0.5*ux*ux - f*ux)

        loss.backward(retain_graph=True)

        history.append((float(loss), ))

        return loss

    try:
        epoch_loss.clear()
        epoch_loss.append(np.nan)

        epochs = 100
        for epoch in range(epochs):
            print(f'Epoch = {epoch} {epoch_loss[-1]}')
            optimizer.step(closure)
    except KeyboardInterrupt:
        pass

# FEM based on optimized coefs
V = df.FunctionSpace(mesh, 'CG', 1)
y = torch.tensor(V.tabulate_dof_coordinates().reshape((1, V.dim(), 2)),
                 dtype=torch.float64)

coefs = u(y).detach().numpy().flatten()

uh = df.Function(V)
uh.vector().set_local(coefs)

df.File('foo.pvd') << uh
