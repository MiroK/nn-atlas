import torch
import torch.nn as nn
import torch.optim as optim

import itertools

from nn_atlas.domains.cusp_square import CuspSquare
from nn_atlas.domains.utils import mesh_interior_points
from nn_atlas.nn_extensions.calculus import grad, diff, cross
from nn_atlas.nn_extensions.p1net import VectorP1FunctionSpace
from nn_atlas.analysis import topological_edges

import matplotlib.pyplot as plt
import dolfin as df
import numpy as np


def HarmonicExtension(domain, *, structured=False, resolution=0.5):
    '''Compute harmonic extension using domain mesh of resolution'''
    rmesh, rentities = domain.get_reference_mesh(structured=structured, resolution=resolution)
    rmesh_boundaries = rentities[1]

    Velm = df.VectorElement('Lagrange', rmesh.ufl_cell(), 1)
    # Compute displacement on the boundary
    bc_displacement = domain.set_mapping_bcs(Velm,
                                             boundaries=rmesh_boundaries,
                                             tags=domain.tags,
                                             mode='displacement')

    # Extension as a function
    V = df.FunctionSpace(rmesh, Velm)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    bcs = df.DirichletBC(V, bc_displacement, 'on_boundary')

    a = df.inner(df.grad(u), df.grad(v))*df.dx
    L = df.inner(df.Constant((0, 0)), v)*df.dx
    
    uh = df.Function(V)
    df.solve(a == L, uh, bcs)

    # Now for pytorch
    he = VectorP1FunctionSpace(rmesh)
    # Set the network to represent uh
    he.set_from_coefficients(uh.vector().get_local())

    return he

    
class DisplacementNetwork(nn.Module):
    '''
    Neural network defined on [0, 1]^2 which extends boundary discplacements.
    The mesh deformation is X -> X + DisplacementNetwork(X). The structure 
    of the net is bdry_mask*NN(X) + HarmonicExtension(X). Harmonic extension 
    is computed on the mesh.
    '''
    def __init__(self, domain, *, structured=False, resolution=0.5):
        super().__init__()
 
        self.lin1 = nn.Linear(2, 48)
        self.lin2 = nn.Linear(48, 48)
        self.lin3 = nn.Linear(48, 32)        
        self.lin4 = nn.Linear(32, 2, bias=False)

        self.harmonic_extension = HarmonicExtension(domain,
                                                    structured=structured,
                                                    resolution=resolution)        
    def forward(self, x):
        # A network for displacement
        y = self.lin1(x)
        y = torch.tanh(y)
        y = self.lin2(y)
        y = torch.tanh(y)
        y = self.lin3(y)
        y = torch.tanh(y)        
        y = self.lin4(y)  

        # [0, 1] x [-1, 0]
        bdry_mask = (x[..., 0]-0.)*(x[..., 0]-1.)*(x[..., 1]-(-1.))*(x[..., 1]-0.)
        bdry_mask = bdry_mask.unsqueeze(2)

        he = self.harmonic_extension(x)
        # y*bdry_mask is zero on boundary, the right bcs are due to he
        return y*bdry_mask + he

# Let's get the training set
d = 0.910
R1 = 1/d
domain = CuspSquare(d=d, R1=R1)

u = DisplacementNetwork(domain)
u.double()

phi = lambda x: x + u(x)

# We only train the NN and leave extension untouched.
# NOTE: u.paremeters() would include harmonic_extension layer
parameters = itertools.chain(*[l.parameters() for l in (u.lin1, u.lin2, u.lin3, u.lin4)])

mesh, _ = domain.get_reference_mesh(resolution=0.5)
# FIXME: random sample/Sobol sequences/midpoints?
interior_ref = mesh.coordinates()
interior_ref = torch.tensor([interior_ref], dtype=torch.float64)

edges_idx = topological_edges(mesh.cells())
edges_idx = [np.fromiter(e.flat, dtype='int32').reshape(e.shape) for e in edges_idx]

# For simplex determinant we can precompute reference element sizes
# (A-B-C) ->  (B-A, C-A)
arrows = [diff(interior_ref[0, edge], axis=1).squeeze(1) for
          edge in edges_idx]
sizes_ref = cross(*arrows)  # There is a constant scaling missing but eh

maxiter = 10000
optimizer = optim.LBFGS(parameters, max_iter=maxiter,
                        history_size=1000, tolerance_change=1e-12,
                        line_search_fn="strong_wolfe")

epoch_loss = []
def closure(history=epoch_loss):
    optimizer.zero_grad()

    # Local determinant
    # J = torch.det(grad(phi(interior_ref), interior_ref))

    # Mesh based determinant
    y = phi(interior_ref)
 
    arrows = [diff(y[0, edge], axis=1).squeeze(1) for
              edge in edges_idx]
    sizes_target = cross(*arrows)  # There is a constant scaling missing but eh

    J = sizes_target/sizes_ref
    
    loss = torch.mean(1/J**2)
    print(f'{len(history)} => Loss = {float(loss)} {float(torch.abs(J).min())}')
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


extensions = {'nn': phi,
              'harmonic': lambda x: x + u.harmonic_extension(x)}

for key in extensions:
    # Grab method
    y = extensions[key](interior_ref)
 
    arrows = [diff(y[0, edge], axis=1).squeeze(1) for
              edge in edges_idx]
    sizes_target = cross(*arrows)  # There is a constant scaling missing but eh

    J = sizes_target/sizes_ref
    # Fill results
    extensions[key] = J.detach().numpy().flatten()
    print(f'{float(torch.abs(J).min())} {float(torch.abs(J).mean())} {float(torch.abs(J).max())}')

print('Done')

x_ref = torch.tensor([mesh.coordinates()])
# NOTE: Paraview's warp by vector is based on displacement
displacement_values = phi(x_ref) - x_ref
harmonic_values = u.harmonic_extension(x_ref)

import meshio

mesh_io = meshio.Mesh(mesh.coordinates(), {'triangle': mesh.cells()})

mesh_io.point_data = {'phi': displacement_values.squeeze(0).detach().numpy(),
                      'phi_harmonic': harmonic_values.squeeze(0).detach().numpy()}

mesh_io.cell_data = {'J': extensions['nn'], 'J_harmonic': extensions['harmonic']}

mesh_io.write('test_nn_extension.vtk')

# FIXME: cleanup
#        what is the final quality - determinant as cell function
#        train on local mapped stiffness matrix?
