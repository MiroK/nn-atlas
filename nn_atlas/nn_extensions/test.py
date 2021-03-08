import torch
import torch.nn as nn
import torch.optim as optim

import itertools

from nn_atlas.domains.cusp_square import CuspSquare
from nn_atlas.domains.utils import mesh_interior_points
from nn_atlas.nn_extensions.calculus import grad, diff, cross
from nn_atlas.analysis import topological_edges

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
        y = self.lin4(y)  # Final deformation

        bdry_mask = (x[..., 0]-0)*(x[..., 0]-1)*(x[..., 1]-(-1))
        bdry_mask = bdry_mask.unsqueeze(2)
        # Here we are enforcing dirichlet on the |_| boundaries which are
        # simple
        # FIXME: In case of cusp square we could also use x[..., 0] as
        # a parametrization of the arc and compute the displacement?
        return y*bdry_mask

u = DeformationNetwork()
u.double()

phi = lambda x: x + u(x)

# Let's get the training set
d = 0.916
R1 = 1/d
domain = CuspSquare(d=d, R1=R1)

# Use boundary points to fix the deformation
t = np.linspace(0, 1, 100)
bdry_ref, bdry_target = domain.ref_target_pairs(t)

bdry_ref, bdry_target = np.row_stack(bdry_ref), np.row_stack(bdry_target)
# On the interior we "just" want some regularity from the extension map
# For now we will take points from the mesh but they really could be random.
# However, these points could be random. Then, for the simplex determinant
# we could take combinations of len 3 or Delaunay triangulation
mesh, _ = domain.get_reference_mesh(resolution=0.5)

interior_ref = mesh.coordinates()

edges_idx = topological_edges(mesh.cells())
edges_idx = [np.fromiter(e.flat, dtype='int32').reshape(e.shape) for e in edges_idx]

bdry_ref, bdry_target, interior_ref = (torch.tensor([array], dtype=torch.float64)
                                       for array in (bdry_ref, bdry_target, interior_ref))

# For simplex determinant we can precompute reference element sizes
# (A-B-C) ->  (B-A, C-A)
arrows = [diff(interior_ref[0, edge], axis=1).squeeze(1) for
          edge in edges_idx]
sizes_ref = cross(*arrows)  # There is a constant scaling missing but eh

maxiter = 1000
optimizer = optim.LBFGS(u.parameters(), max_iter=maxiter,
                        history_size=1000, tolerance_change=1e-8,
                        line_search_fn="strong_wolfe")



epoch_loss = []
def closure(history=epoch_loss):
    optimizer.zero_grad()

    # Match on the boundary
    f_x = phi(bdry_ref)
    bdry_loss = ((bdry_target - f_x)**2).mean()

    # Refularize
    penalty_J = torch.tensor(1E-4, dtype=torch.float64)    
    # Local determinant
    # J = torch.det(grad(phi(interior_ref), interior_ref))

    # Mesh based determinant
    y = phi(interior_ref)
 
    arrows = [diff(y[0, edge], axis=1).squeeze(1) for
              edge in edges_idx]
    sizes_target = cross(*arrows)  # There is a constant scaling missing but eh

    J = sizes_target/sizes_ref
    
    loss = (bdry_loss
            + penalty_J*torch.mean(1/J**2)
            )
    print(f'{len(history)} => Loss = {float(loss)} {float(J.min())}')
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

import meshio

mesh_io = meshio.Mesh(mesh.coordinates(), {'triangle': mesh.cells()})
mesh_io.point_data = {'phi': displacement_values.squeeze(0).detach().numpy()}
#mesh_io.cell_data = {'det': detF_values.squeeze(0).detach().numpy()}

mesh_io.write('test_nn_extension.vtk')

