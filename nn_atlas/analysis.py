from dolfin import *
import numpy as np


def deformation_gradient(phi, mesh, is_displacement=True):
    '''Deformation gradient (in its natural space) on mesh'''
    T = TensorFunctionSpace(mesh, 'DG', phi.function_space().ufl_element().degree()-1)
    
    if is_displacement:
        return project(Identity(len(phi)) + grad(phi), T)

    return project(grad(phi), T)


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
    
    value_from_dofs = lambda x, n=n: (np.linalg.det(x.reshape((n, n))), )
    return build_dg_function(grad_phi,
                             transform=value_from_dofs,
                             nout=1)


# def get_lambdas(grad_phi):
#     '''Fields of eigenvalues'''
#     n = len(grad_phi)
#     value_from_dofs = lambda x, n=n: np.linalg.det(x.reshape((n, n)))
#     return build_dg_function(grad_phi,
#                              transform=value_from_dofs,
#                              nout=n)

# --------------------------------------------------------------------

if __name__ == '__main__':

    mesh = UnitSquareMesh(5, 5)
    V = VectorFunctionSpace(mesh, 'CG', 2)

    f = interpolate(Expression(('x[0]', '2*x[1]'), degree=1), V)

    grad_f = deformation_gradient(f, mesh=mesh, is_displacement=False)
    J, = get_J(grad_f)

    print(J.vector().get_local())

# TODO: 
#  - test analysis functionality
#  - wire up
#  - elasticity (migh need mixed because of incompresibility)
#
#  - training data - seed reference
#  - detF training
#  - cross product training <-- first chec that we get the formula right
