from functools import reduce
from operator import or_
from dolfin import *


def check_bcs(bdry, dirichlet, neumann):
    '''No overlaps in bcs and bcs make sense given data'''
    mesh = bdry.mesh()
    assert mesh.topology().dim() - 1 == bdry.dim()

    found_tags = set(bdry.array())  # Local
    found_tags = reduce(or_, mesh.mpi_comm().allgather(found_tags))

    assert set(dirichlet.keys()) <= found_tags
    assert set(neumann.keys()) <= found_tags
    
    assert not set(dirichlet.keys()) & set(neumann.keys())

    return True


def harmonic_extension(V, boundaries, dirichlet_bcs, neumann_bcs={}, diffusion=Constant(1)):
    '''
    Given boundary displacement we extend to the entire domain by solving

      -div(diffusion*grad(u)) = 0.

    Neuamnn_bcs is data on diffusion*grad(u).n
    '''
    assert check_bcs(boundaries, dirichlet_bcs, neumann_bcs)
    assert dirichlet_bcs

    u, v = TrialFunction(V), TestFunction(V)
    
    a = inner(diffusion*grad(u), grad(v))*dx
    L = inner(Constant((0, )*len(u)), v)*dx
    # Neumann bcs
    ds = Measure('ds', domain=boundaries.mesh(), subdomain_data=boundaries.mesh())
    for tag, value in neumann_bcs.items():
        L += inner(value, v)*ds(tag)
    # Dirichlet ones
    bcs = [DirichletBC(V, value, boundaries, tag) for tag, value in dirichlet_bcs.items()]

    A, b = PETScMatrix(), PETScVector()
    assemble_system(a, L, bcs, A_tensor=A, b_tensor=b)

    solver = PETScKrylovSolver('cg', 'hypre_amg')
    solver.parameters['monitor_convergence'] = True
    solver.parameters['relative_tolerance'] = 1E-13

    solver.set_operator(A)

    uh = Function(V)
    niters = solver.solve(uh.vector(), b)

    return uh
