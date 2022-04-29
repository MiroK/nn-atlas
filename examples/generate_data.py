# This example shows how to generate data for learning the neural network
# that extends the boundary data similar to some PDE operator. The data
# will be stored in npz files and in general are formed by pairs of D-dim
# vector (inputs; spatial points) and D-dim vectors (outpurs, the displacements)
from nn_atlas.domains.cusp_square import CuspSquare
from nn_atlas.extensions import harmonic_extension, elastic_extension, biharmonic_extension
from nn_atlas.extensions import interior_dofs_values, boundary_dofs_values
from nn_atlas.domains.utils import sample_interior, sample_boundary
import dolfin as df
import numpy as np

# Our target domain is the square with a cusp on the top edge. The cusp
# forms sort of a tent and the higher the tent/pitch the more difficult
# is to have a smooth deformation
d = 0.916                                    # ish vent height
resolution = 0.5                             # mesh resolution: accuracy and amount of data
extension = 'harmonic'                       # which PDE to use for generating data

identifier = f'CuspSquare_d{d}_res{resolution}_{extension}'

# -------------- DONE WITH CONFIGURATION ------------------

domain = CuspSquare(d=d, R1=1/d)
# The network will always work on a reference mesh which is square/rectangle
# of some resolution. The structured == True parameter means that the mesh will
# be formed by identically sized triangles
rmesh, rentities = domain.get_reference_mesh(structured=False, resolution=resolution)
rmesh_boundaries = rentities[1]
# To generate the data we will compute displacement in a some finite element
# space ...
Velm = df.VectorElement('Lagrange', rmesh.ufl_cell(), 2)
# Compute displacement on the boundary for setting up boundary conditions
bc_displacement = domain.set_mapping_bcs(Velm,
                                         boundaries=rmesh_boundaries,
                                         tags=domain.tags,
                                         mode='displacement')
# Here we will prescribed the displacement everywhere
dirichlet_bcs = {tag: bc_displacement for tag in domain.tags}
V = df.FunctionSpace(rmesh, Velm)

extend = {'harmonic': harmonic_extension,
          'elastic': elastic_extension,
          # FIXME: biharnomic is not tested
          'biharmonic': biharmonic_extension}[extension]

uh = extend(V, rmesh_boundaries, dirichlet_bcs=dirichlet_bcs)
# Store the displacement field for later
uh.rename('uh', '0')
df.File(f'uh_{identifier}.pvd') << uh

# For information we compute the mesh quality: re wonder about squishing of
# the cells by the computed deformation field (relative volume change)
x = df.SpatialCoordinate(rmesh)
Q = df.FunctionSpace(rmesh, 'DG', 0)
q = df.TestFunction(Q)

phi = x + uh
qh = df.Function(Q)
df.assemble((1/df.CellVolume(rmesh))*df.inner(q, abs(df.det(df.grad(phi))))*df.dx,
            tensor=qh.vector())
print(qh.vector().min(), qh.vector().max())

# Now we generate the data from the displacement
# First we grab all the dof values; i.e. point values of the displacement
# field in the locations of the degrees of freedom.
ix, ixvals = interior_dofs_values(uh)
# We keep separate the interior dofs (where PDE holds) and boundary dofs
# which might go into their seperate loss function
bx, bxvals = boundary_dofs_values(uh)

# If more data is desired we can also sample the field in other points.
iy = sample_interior(rmesh, npts=1_000)
by = sample_boundary(rmesh, npts=1_000)
# and we again separate the points on the boudary from the interior
iyvals = np.array([uh(y) for y in iy])
byvals = np.array([uh(y) for y in by])

# Combine into interior/boundary
interior_x = np.r_[ix, iy]
interior_vals = np.r_[ixvals, iyvals]

boundary_x = np.r_[bx, by]
boundary_vals = np.r_[bxvals, byvals]

# And finally save
path = f'data_{identifier}.npz'
np.savez(path,
         interior_x=interior_x, interior_vals=interior_vals,
         boundary_x=boundary_x, boudnary_vals=boundary_vals)

# This should be what is needed to learn a mappign NN(x) --> displacement@x
