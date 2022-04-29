# PDE extension operators
from nn_atlas.extensions.harmonic import harmonic_extension
from nn_atlas.extensions.elastic import elastic_extension, mixed_elastic_extension
from nn_atlas.extensions.biharmonic import biharmonic_extension
import dolfin as df
import numpy as np


def all_dofs_values(u):
    '''Coordinates of dofs of u the values of u there.'''
    V = u.function_space()
    # Since we don't want to interpolate we are restricted to spaces with
    # dofs being nodal evaluations so ... there are more but
    assert V.ufl_element().family() in ('Lagrange', 'Discontinuous Lagrange', 'Crouzeix-Raviart')
    gdim = V.mesh().geometry().dim()
    
    vals = u.vector().get_local()

    if V.num_sub_spaces() == 0:
        x = V.tabulate_dof_coordinates().reshape((-1, gdim))
        vals = vals.reshape((-1, 1))
        
        return x, vals

    x = V.sub(0).collapse().tabulate_dof_coordinates().reshape((-1, gdim))
    vals = np.column_stack([vals[V.sub(i).dofmap().dofs()] for i in range(V.num_sub_spaces())])
    return x, vals
    

def interior_dofs_values(u):
    '''
    Coordinates of dofs of u that are away from the boundary and the values 
    of u there.
    '''
    x, vals = all_dofs_values(u)

    V = u.function_space()
    if V.num_sub_spaces() > 0:
        V = V.sub(0).collapse()
    
    bdry_dofs = set(df.DirichletBC(V, df.Constant(0), 'on_boundary').get_boundary_values().keys())
    interior_dofs = list(set(range(len(x))) - bdry_dofs)

    return x[interior_dofs], vals[interior_dofs]
    

def boundary_dofs_values(u):
    '''
    Coordinates of dofs of u that are on the boundary and the values 
    of u there.
    '''
    x, vals = all_dofs_values(u)

    V = u.function_space()
    if V.num_sub_spaces() > 0:
        V = V.sub(0).collapse()
    
    bdry_dofs = list(df.DirichletBC(V, df.Constant(0), 'on_boundary').get_boundary_values().keys())

    return x[bdry_dofs], vals[bdry_dofs]

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    mesh = df.UnitSquareMesh(4, 4)
    V = df.FunctionSpace(mesh, 'CG', 2)
    u = df.interpolate(df.Expression('sin(pi*(x[0]-x[1]))', degree=4),
                                      #'cos(pi*(x[0]-x[1]))'), degree=4),
                       V)

    pts, vals = all_dofs_values(u)
    
    pts1, vals1 = boundary_dofs_values(u)
