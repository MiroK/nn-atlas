from nn_atlas.domains import CuspSquare
from nn_atlas.extensions import harmonic_extension
import dolfin as df

# Domain parametrization
d = 0.8
R1 = 1./d
R2 = 1./d

domain = CuspSquare(d=d, R1=R1, R2=R2)
# Where we will compute the deformation    
mesh, subdomains = domain.get_reference_mesh(resolution=1.0)
boundaries = subdomains[1]

tags = (1, 2, 3, 4, 5)

Velm = df.VectorElement('Lagrange', df.triangle, 1)
bc_map = domain.set_mapping_bcs(Velm,
                                boundaries=boundaries,
                                tags=tags,
                                mode='displacement')

d_bcs = {tag: bc_map for tag in tags}

displacement = harmonic_extension(bc_map.function_space(),
                                  boundaries=boundaries,
                                  dirichlet_bcs=d_bcs)

df.File('foo.pvd') << displacement
