from nn_atlas.domains.utils import sample_boundary, sample_interior
from nn_atlas.extensions import interior_dofs_values, boundary_dofs_values
import dolfin as df
import numpy as np


def test_idofs():
    '''(0, 1)^2'''
    mesh = df.UnitSquareMesh(32, 32)
    V = df.VectorFunctionSpace(mesh, 'CG', 2)
    u = df.interpolate(df.Expression(('sin(pi*(x[0]-x[1]))',
                                      'cos(pi*(x[0]-x[1]))'), degree=4),
                       V)
    
    pts, vals = interior_dofs_values(u)
    # Points are interior
    # Distance
    x, y = pts.T
    d = np.min(np.c_[x, 1-x, y, 1-y], axis=1)
    assert np.all(d > 1E-10)

    # Values are fine
    vals0 = np.c_[np.sin(np.pi*(x-y)), np.cos(np.pi*(x-y))]
    assert np.linalg.norm(vals - vals0) < 1E-12


def test_bdofs_scalar():
    '''(0, 1)^2'''
    mesh = df.UnitSquareMesh(32, 32)
    V = df.FunctionSpace(mesh, 'CG', 2)
    u = df.interpolate(df.Expression('sin(pi*(x[0]-x[1]))', degree=4),
                                      #'cos(pi*(x[0]-x[1]))'), degree=4),
                       V)
    
    pts, vals = boundary_dofs_values(u)
    # Points are on boundary
    x, y = pts.T
    assert np.all(np.logical_or(np.logical_or(np.abs(x) < 1E-10, np.abs(x-1) < 1E-10),
                                np.logical_or(np.abs(y) < 1E-10, np.abs(y-1) < 1E-10)))

    # Values are fine
    vals0 = np.sin(np.pi*(x-y)).reshape((-1, 1))#, np.cos(np.pi*(x-y))]
    assert np.linalg.norm(vals - vals0) < 1E-12


def test_bdofs_vector():
    '''(0, 1)^2'''
    mesh = df.UnitSquareMesh(32, 32)
    V = df.VectorFunctionSpace(mesh, 'CG', 2)
    u = df.interpolate(df.Expression(('sin(pi*(x[0]-x[1]))',
                                      'cos(pi*(x[0]-x[1]))'), degree=4),
                       V)
    
    pts, vals = boundary_dofs_values(u)
    # Points are on boundary
    x, y = pts.T
    assert np.all(np.logical_or(np.logical_or(np.abs(x) < 1E-10, np.abs(x-1) < 1E-10),
                                np.logical_or(np.abs(y) < 1E-10, np.abs(y-1) < 1E-10)))

    # Values are fine
    vals0 = np.c_[np.sin(np.pi*(x-y)), np.cos(np.pi*(x-y))]
    assert np.linalg.norm(vals - vals0) < 1E-12
    

def test_ipts():
    '''(0, 1)^2'''
    mesh = df.UnitSquareMesh(32, 32)

    pts = sample_interior(mesh, 100)
    x, y = pts.T
    # Distance
    d = np.min(np.c_[x, 1-x, y, 1-y], axis=1)
    assert np.all(d > 1E-10)


def test_bpts():
    '''(0, 1)^2'''
    mesh = df.UnitSquareMesh(32, 32)

    pts = sample_boundary(mesh, 100)
    x, y = pts.T

    assert np.all(np.logical_or(np.logical_or(np.abs(x) < 1E-10, np.abs(x-1) < 1E-10),
                                np.logical_or(np.abs(y) < 1E-10, np.abs(y-1) < 1E-10)))
