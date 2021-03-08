import torch
import torch.nn as nn
import torch.optim as optim

import itertools

from nn_atlas.domains.cusp_square import CuspSquare
from nn_atlas.domains.utils import mesh_interior_points
from nn_atlas.nn_extensions.calculus import grad

import matplotlib.pyplot as plt
import numpy as np


class DeformationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.lin1 = nn.Linear(2, 48)
        self.lin2 = nn.Linear(48, 48)
        self.lin3 = nn.Linear(48, 32)        
        self.lin4 = nn.Linear(32, 2, bias=False)
 
    def forward(self, x):
        # A network for displacement
        y = self.lin1(x)
        y = torch.tanh(y)
        y = self.lin2(y)
        y = torch.tanh(y)
        y = self.lin3(y)
        y = torch.tanh(y)        
        y = self.lin4(y)
        # Final deformation
        return x + y

phi = DeformationNetwork()
phi.double()

# Let's get the training set
d = 0.9
R1 = 1/d
domain = CuspSquare(d=d, R1=R1)

# Use boundary points to fix the deformation
t = np.linspace(0, 1, 100)
bdry_ref, bdry_target = domain.ref_target_pairs(t)

# On the interior we "just" want some regularity from the extension map
# For now we will take points from the mesh
mesh, _ = domain.get_reference_mesh(resolution=0.5)

interior_ref = mesh_interior_points(mesh)

bdry_ref, bdry_target, interior_ref = (torch.tensor([array], dtype=torch.float64)
                                       for array in (bdry_ref, bdry_target, interior_ref))

maxiter = 1000
optimizer = optim.LBFGS(phi.parameters(), max_iter=maxiter,
                        history_size=1000, tolerance_change=1e-12,
                        line_search_fn="strong_wolfe")

interior_ref.requires_grad = True

epoch_loss = []
def closure(history=epoch_loss):
    optimizer.zero_grad()

    # Match on the boundary
    f_x = phi(bdry_ref)
    bdry_loss = ((bdry_target - f_x)**2).mean()

    # Refularize
    J = torch.det(grad(phi(interior_ref), interior_ref))
    penalty_J = torch.tensor(1E-5, dtype=torch.float64)
    
    loss = (bdry_loss
            +penalty_J*torch.mean(1/J**2)
            )
    print(f'Loss = {float(loss)}')
    loss.backward()

    history.append((float(loss), ))
    
    return loss
 
try:
    epoch_loss.clear()
    
    epochs = 1
    for epoch in range(epochs):
        print(f'Epoch = {epoch}')
        optimizer.step(closure)
except KeyboardInterrupt:
    pass

closure()

print('Done')

x_ref = torch.tensor([mesh.coordinates()])
# NOTE: Paraview's warp by vector is based on displacement
displacement_values = phi(x_ref) - x_ref

#xmid_hat = torch.tensor([cell_midpoints(mesh)], requires_grad=True)
#detF_values = torch.det(grad(phi(xmid_hat), xmid_hat))

import meshio

mesh_io = meshio.Mesh(mesh.coordinates(), {'triangle': mesh.cells()})
mesh_io.point_data = {'phi': displacement_values.squeeze(0).detach().numpy()}
#mesh_io.cell_data = {'det': detF_values.squeeze(0).detach().numpy()}

mesh_io.write('test_nn_extension.vtk')

