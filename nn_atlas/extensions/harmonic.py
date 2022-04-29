from nn_atlas.extensions.utils import check_bcs
from dolfin import *


def harmonic_extension(V, boundaries, dirichlet_bcs, neumann_bcs={}, diffusion=Constant(1), f=None):
    '''
    Given boundary displacement we extend to the entire domain by solving

      -div(diffusion*grad(u)) = 0.

    Neuamnn_bcs is data on diffusion*grad(u).n
    '''
    assert check_bcs(boundaries, dirichlet_bcs, neumann_bcs)
    assert dirichlet_bcs

    u, v = TrialFunction(V), TestFunction(V)
    
    a = inner(diffusion*grad(u), grad(v))*dx
    if f is not None:
        L = inner(f, v)*dx
    else:
        L = inner(Constant((0, )*len(u)), v)*dx
    # Neumann bcs
    ds = Measure('ds', domain=boundaries.mesh(), subdomain_data=boundaries)
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
