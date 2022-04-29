# PDE extension operators
from nn_atlas.extensions.harmonic import harmonic_extension
from nn_atlas.extensions.elastic import elastic_extension, mixed_elastic_extension
from nn_atlas.extensions.biharmonic import biharmonic_extension
import dolfin as df
import numpy as np


def null(u):
    '''Zero like u'''
    return df.Constant(np.zeros(u.ufl_shape))


def all_dofs_values(u):
    '''Coordinates of dofs of u the values of u there.'''
    V = u.function_space()
    # Since we don't want to interpolate we are restricted to spaces with
    # dofs being nodal evaluations so ... there are more but
    assert V.ufl_element().family() in ('Lagrange', 'Discontinuous Lagrange', 'Crouzeix-Raviart')

    if V.ufl_element().num_sub_elements():
        elm, = set(V.ufl_element().sub_elements())
    
    vals = u.vector().get_local()
    value_size = 1 if not u.ufl_shape else np.prod(u.ufl_shape)
    vals = vals.reshape((-1, value_size))

    if V.ufl_element().num_sub_elements():
        x = V.sub(0).collapse().tabulate_dof_coordinates().reshape((-1, V.mesh().geometry().dim()))
    else:
        x = V.tabulate_dof_coordinates().reshape((-1, V.mesh().geometry().dim()))
    
    return x, vals
    

def interior_dofs_values(u):
    '''
    Coordinates of dofs of u that are away from the boundary and the values 
    of u there.
    '''
    x, vals = all_dofs_values(u)
    x_shape, vals_shape = x.shape, vals.shape
    x, vals = x.flatten(), vals.flatten()
    
    bdry_dofs = set(df.DirichletBC(u.function_space(),
                                   null(u),
                                   'on_boundary').get_boundary_values().keys())
    interior_dofs = list(set(range(len(x))) - bdry_dofs)

    return (x[interior_dofs].reshape((-1, x_shape[1])),
            vals[interior_dofs].reshape((-1, vals_shape[1])))
    

def boundary_dofs_values(u):
    '''
    Coordinates of dofs of u that are on the boundary and the values 
    of u there.
    '''
    V = u.function_space()
    bdry_dofs = list(df.DirichletBC(u.function_space(),
                                    null(u),
                                    'on_boundary').get_boundary_values().keys())

    x, vals = all_dofs_values(u)
    x_shape, vals_shape = x.shape, vals.shape
    x, vals = x.flatten(), vals.flatten()
    
    return (x[bdry_dofs].reshape((-1, x_shape[1])),
            vals[bdry_dofs].reshape((-1, vals_shape[1])))
