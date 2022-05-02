from nn_atlas.extensions.utils import check_bcs
from dolfin import *


def biharmonic_extension(V, boundaries, dirichlet_bcs, diffusion=Constant(1), f=None):
    '''
    Given boundary displacement we extend to the entire domain by solving

      -Delta(diffusion*Delta(u)) = 0.
    '''
    assert check_bcs(boundaries, dirichlet_bcs, {})
    assert dirichlet_bcs
    assert V.ufl_element().degree() > 1
    
    u, v = TrialFunction(V), TestFunction(V)
    
    mesh = V.mesh()
    h = FacetArea(mesh)
    h_avg = (h('+') + h('-'))/2.0
    n = FacetNormal(mesh)

    alpha = Constant(8.0)
    
    a = (inner(div(grad(u)), div(grad(v)))*dx 
         - inner(avg(div(grad(u))), jump(grad(v), n))*dS 
         - inner(jump(grad(u), n), avg(div(grad(v))))*dS 
         + alpha('+')/h_avg*inner(jump(grad(u),n), jump(grad(v),n))*dS)

    if f is not None:
        L = inner(f, v)*dx
    else:
        L = inner(Constant((0, )*len(u)), v)*dx
    # Dirichlet ones
    bcs = [DirichletBC(V, value, boundaries, tag) for tag, value in dirichlet_bcs.items()]

    A, b = PETScMatrix(), PETScVector()
    assemble_system(a, L, bcs, A_tensor=A, b_tensor=b)

    solver = LUSolver('mumps')
    solver.set_operator(A)

    uh = Function(V)
    niters = solver.solve(uh.vector(), b)

    return uh
