from nn_atlas.domains.cusp_square import CuspSquare
from nn_atlas.extensions import harmonic_extension, elastic_extension, biharmonic_extension
from nn_atlas.extensions import interior_dofs_values, boundary_dofs_values
from nn_atlas.domains.utils import sample_interior, sample_boundary
import dolfin as df
import numpy as np


d = 0.916
resolution = 0.5
extension = 'harmonic'

identifier = f'CuspSquare_d{d}_res{resolution}_{extension}'

# --------------

R1 = 1/d
domain = CuspSquare(d=d, R1=R1)

rmesh, rentities = domain.get_reference_mesh(structured=False, resolution=resolution)
rmesh_boundaries = rentities[1]

Velm = df.VectorElement('Lagrange', rmesh.ufl_cell(), 2)
# Compute displacement on the boundary
bc_displacement = domain.set_mapping_bcs(Velm,
                                         boundaries=rmesh_boundaries,
                                         tags=domain.tags,
                                         mode='displacement')

dirichlet_bcs = {tag: bc_displacement for tag in domain.tags}
V = df.FunctionSpace(rmesh, Velm)

extend = {'harmonic': harmonic_extension,
          'elastic': elastic_extension,
          'biharmonic': biharmonic_extension}[extension]

uh = extend(V, rmesh_boundaries, dirichlet_bcs=dirichlet_bcs)

uh.rename('uh', '0')
df.File(f'uh_{identifier}.pvd') << uh

# Consider quality of the mesh
x = df.SpatialCoordinate(rmesh)
Q = df.FunctionSpace(rmesh, 'DG', 0)
q = df.TestFunction(Q)

phi = x + uh
qh = df.Function(Q)
df.assemble((1/df.CellVolume(rmesh))*df.inner(q, abs(df.det(df.grad(phi))))*df.dx,
            tensor=qh.vector())
print(qh.vector().min(), qh.vector().max())


ix, ixvals = interior_dofs_values(uh)
bx, bxvals = boundary_dofs_values(uh)

iy = sample_interior(rmesh, npts=1_000)
by = sample_boundary(rmesh, npts=1_000)

iyvals = np.array([uh(y) for y in iy])
byvals = np.array([uh(y) for y in by])

# Compine
interior_x = np.r_[ix, iy]
interior_vals = np.r_[ixvals, iyvals]

boundary_x = np.r_[bx, by]
boundary_vals = np.r_[bxvals, byvals]

path = f'data_{identifier}.npz'
np.savez(path,
         interior_x=interior_x, interior_vals=interior_vals,
         boundary_x=boundary_x, boudnary_vals=boundary_vals)
