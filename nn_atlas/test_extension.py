from nn_atlas.domains import CuspSquare
from nn_atlas.extensions import (harmonic_extension, elastic_extension,
                                 mixed_elastic_extension)
import nn_atlas.analysis as analysis
import dolfin as df
import numpy as np

# Domain parametrization
d = 0.916
# NOTE: below we are going to pick radius as 1/d. A consequence of this
# choise is that we are limited to (0, root). Considering line (0.5, y)
# there will always be 2 intersects of is and the cirlce; one is our point
# (0.5, d). If d < root than the other point has y > d and is not interesting
# However, if d > root, then the circle arches overlap inside the domain -
# game over. We find the root by realizing that the turning point is when
# there is just one intersect, i.e. (0.5, d) - center is aligned with x-axis
# This leeds to 
# In [41]: a = (-d**2*(d**2 + 1/4)/4 + 1) - (d**2/4)*(4*d**2*(d**2 + 1/4))

# In [42]: sp.roots(a, d)
#  0.916875478860700
R1 = 1./d
R2 = 1./d

domain = CuspSquare(d=d, R1=R1, R2=R2)
# Where we will compute the deformation    
mesh, subdomains = domain.get_reference_mesh(resolution=0.5)
boundaries = subdomains[1]

tags = (1, 2, 3, 4, 5)

Velm = df.VectorElement('Lagrange', df.triangle, 1)
bc_map = domain.set_mapping_bcs(Velm,
                                boundaries=boundaries,
                                tags=tags,
                                mode='displacement')

d_bcs = {tag: bc_map for tag in tags}

cell = mesh.ufl_cell()
Welm = df.MixedElement([df.VectorElement('Lagrange', cell, 2),
                        df.FiniteElement('Lagrange', cell, 1)])

W = df.FunctionSpace(mesh, Welm)

displacement = mixed_elastic_extension(W,
                                       boundaries=boundaries,
                                       dirichlet_bcs=d_bcs,
                                       lmbda=df.Constant(1E-2))

grad_d = analysis.deformation_gradient(displacement,
                                       is_displacement=True)

# Determinant as field
J, = analysis.get_J(grad_d)
J_values = J.vector().get_local()

stats = (np.abs(J_values).min(),
         np.abs(J_values).max(),
         J_values.min(),
         J_values.max(),
         df.sqrt(df.assemble(df.inner(J, J)*df.dx)))

print(stats)

df.File('foo.pvd') << displacement
