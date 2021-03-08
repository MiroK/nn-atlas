from dolfin import *
import numpy as np


def deformation_gradient(phi, mesh=None, is_displacement=True, deg_decrement=0):
    '''Deformation gradient on mesh'''
    assert deg_decrement >= 0

    if mesh is None:
        mesh = phi.function_space().mesh()
    
    T = TensorFunctionSpace(mesh,
                            'DG',
                            phi.function_space().ufl_element().degree()-deg_decrement)
    
    if is_displacement:
        form = Identity(len(phi)) + grad(phi)
    else:
        form = grad(phi)
        
    return project(form, T)


def build_dg_function(v, transform, nout):
    '''For v in DG0 space use the defs to compute transport'''
    # The idea here is to avod projection and compute on each cell
    V = v.function_space()
    elm = V.ufl_element()
    assert elm.family() == 'Discontinuous Lagrange', elm.family()

    nsubs = V.num_sub_spaces()
    Vdms = [V.sub(i).dofmap() for i in range(nsubs)]
    v_values = v.vector().get_local()
    
    # Each of the outputs will be a function in scalar space
    S = V.sub(0).collapse()
    Sdm = S.dofmap()

    outputs = tuple(Function(S) for _ in range(nout))
    outputs_values = [o.vector().get_local() for o in outputs]
    
    for cell in range(V.mesh().num_cells()):
        # In row we have values for each scalar dof
        cell_dofs = np.column_stack([dm.cell_dofs(cell) for dm in Vdms])
        dof_values = v_values[cell_dofs]
        
        for Sdof, values in zip(Sdm.cell_dofs(cell), dof_values):
            for o, x in zip(outputs_values, transform(values)):
                o[Sdof] = x

    for f, vals in zip(outputs, outputs_values):
        f.vector().set_local(vals)

    return outputs


def get_J(grad_phi):
    '''deteminant(grad_phi) field'''
    n, m = grad_phi.value_shape()
    assert n == m
    
    value_from_dofs = lambda x, n=n: (np.linalg.det(x.reshape((n, n))), )
    
    return build_dg_function(grad_phi,
                             transform=value_from_dofs,
                             nout=1)


def get_J_simplex(phi, x_points, cells):
    '''Compute the determinant as the ratio of dv/dV'''
    # Typical use case is x_points = mesh.coordinates() and cells = mesh.cells()
    
    # FIXME: I only need it in 2d now
    y = np.array([phi(x) for x in x_points])
    # Here each for in cells defines a simplex
    cells = cells.T
    # Just as indices
    edges = [np.c_[cells[i], cells[0]] for i in range(1, len(cells))]
    # Displacement vectors before ...
    x_disp = [np.diff(x[e], axis=1).squeeze(1) for e in edges]
    # ... after
    y_disp = [np.diff(y[e], axis=1).squeeze(1) for e in edges]

    dVs = np.cross(*x_disp)
    dvs = np.cross(*y_disp)

    return dvs/dVs  # Can still be negative


def get_lambdas(grad_phi):
    '''Fields of eigenvalues'''
    n, m = grad_phi.value_shape()
    assert n == m
    
    def value_from_dofs(x, n=n, tol=1E-10):
        eigvals = np.linalg.eigvals(x.reshape((n, n)))
        assert np.linalg.norm(eigvals.imag) < tol

        return np.sort(eigvals.real)

    return build_dg_function(grad_phi,
                             transform=value_from_dofs,
                             nout=n)

# --------------------------------------------------------------------

if __name__ == '__main__':
    import sympy as sp
    import ufl
    
    x, y = sp.symbols('x[0] x[1]')
    # Deformation is Identity + displacement
    f0 = sp.Matrix([x, y]) + sp.Matrix([x**2 + x, 2*y**2 - y])
    # f0 = sp.Matrix([x+y, 2*y])    
    grad_f0 = sp.Matrix([[f0[0].diff(x, 1), f0[0].diff(y, 1)],
                         [f0[1].diff(x, 1), f0[1].diff(y, 1)]])

    J0 = Expression(sp.printing.ccode(sp.det(grad_f0)), degree=4)
    
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, 'CG', 2)

    f = interpolate(Expression((sp.printing.ccode(f0[0]),
                                sp.printing.ccode(f0[1])), degree=2), V)

    grad_f = deformation_gradient(f, mesh=mesh, is_displacement=False)

    xx = Expression(((sp.printing.ccode(grad_f0[0, 0]), sp.printing.ccode(grad_f0[0, 1])),
                     (sp.printing.ccode(grad_f0[1, 0]), sp.printing.ccode(grad_f0[1, 1]))),
                    degree=2)
    eJ = assemble(inner(grad_f - xx, grad_f - xx)*dx)
    print(eJ)

    
    J, = get_J(grad_f)

    eJ = assemble(inner(J - J0, J - J0)*dx)
    print(eJ)  # NOTE: determinant is essentially a polynomial so we want
               # exactness ...

    # ----------

    eig0, eig1 = (Expression(sp.printing.ccode(expr), degree=4)
                  for expr in grad_f0.eigenvals().keys())

    eig_min0 = ufl.Min(abs(eig0), abs(eig1))
    eig_max0 = ufl.Max(abs(eig0), abs(eig1))    
    
    eig_min, eig_max = get_lambdas(grad_f)

    # ... on the other hand eigenvalues are roots of a polynomial so
    # we settle for mosh convergence of the error
    print(assemble(inner(eig_min - eig_min0, eig_min - eig_min0)*dx))
    print(assemble(inner(eig_max - eig_max0, eig_max - eig_max0)*dx))    


# TODO: 
#  - test analysis functionality
#  - wire up
#  - elasticity (migh need mixed because of incompresibility)
#
#  - training data - seed reference
#  - detF training
#  - cross product training <-- first chec that we get the formula right
