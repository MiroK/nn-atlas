from nn_atlas.extensions.utils import check_bcs
from dolfin import *


def elastic_extension(V, boundaries, dirichlet_bcs, neumann_bcs={},
                      mu=Constant(1), lmbda=Constant(1), sym_grad=False):
    '''
    Given boundary displacement we extend to the entire domain by solving

      -div(sigma(u)) = 0 with 

    sigma(u) = mu*grad(u) + lambda*div(u)*I (sym_grad is False)
    sigma(u) = 2*mu*sym(grad(u)) + lambda*div(u)*I

    Neuman_bcs is data on sigma(u).n

    NOTE: if lmbda >> mu this formulation might lock and then mixed formulation
    should be used
    '''
    assert check_bcs(boundaries, dirichlet_bcs, neumann_bcs)
    assert dirichlet_bcs

    u, v = TrialFunction(V), TestFunction(V)

    if not sym_grad:
        a = inner(mu*grad(u), grad(v))*dx + lmbda*inner(div(u), div(v))*dx
    else:
        a = inner(2*mu*sym(grad(u)), sym(grad(v)))*dx + lmbda*inner(div(u), div(v))*dx
        
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


def mixed_elastic_extension(V, boundaries, dirichlet_bcs, neumann_bcs={},
                            mu=Constant(1), lmbda=Constant(1), sym_grad=False):
    '''Mixed formulation. Assuming here V is a stable space'''
    assert check_bcs(boundaries, dirichlet_bcs, neumann_bcs)
    assert dirichlet_bcs

    u, p = TrialFunctions(V)
    v, q = TestFunctions(V)

    if not sym_grad:
        a = (inner(mu*grad(u), grad(v))*dx + inner(p, div(v))*dx +
             inner(q, div(u))*dx - (1/lmbda)*inner(p, q)*dx)
    else:
        a = (inner(mu*sym(grad(u)), sym(grad(v)))*dx + inner(p, div(v))*dx +
             inner(q, div(u))*dx - (1/lmbda)*inner(p, q)*dx)
        
    L = inner(Constant((0, )*len(u)), v)*dx
    # Neumann bcs
    ds = Measure('ds', domain=boundaries.mesh(), subdomain_data=boundaries.mesh())
    for tag, value in neumann_bcs.items():
        L += inner(value, v)*ds(tag)
    # Dirichlet ones
    bcs = [DirichletBC(V.sub(0), value, boundaries, tag) for tag, value in dirichlet_bcs.items()]

    A, b = PETScMatrix(), PETScVector()
    assemble_system(a, L, bcs, A_tensor=A, b_tensor=b)

    solver = LUSolver('mumps')
    solver.set_operator(A)

    uh_ph = Function(V)
    niters = solver.solve(uh_ph.vector(), b)

    uh, ph = uh_ph.split(deepcopy=True)

    return uh
