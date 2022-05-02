from nn_atlas.extensions import harmonic_extension, elastic_extension, biharmonic_extension
from dolfin import *
import numpy as np
import pytest


@pytest.mark.parametrize('deg', [1, 2, 3])
def test_harmonic(deg):
    import ulfy

    mesh = UnitSquareMesh(1, 1)
    x, y = SpatialCoordinate(mesh)

    u = sin(pi*(x-y**2))
    f = -div(grad(u))

    u, f = ulfy.Expression(u, degree=deg+3), ulfy.Expression(f, degree=deg+4)

    hs, errors = [], []
    for k in range(1, 6):
        n = 2**k
        mesh = UnitSquareMesh(n, n)

        boundaries = MeshFunction('size_t', mesh, 1, 0)
        DomainBoundary().mark(boundaries, 1)
        
        dirichlet_bcs = {1: u}
        V = FunctionSpace(mesh, 'CG', deg)

        uh = harmonic_extension(V, boundaries, dirichlet_bcs=dirichlet_bcs, f=f)

        errors.append(errornorm(u, uh, 'H1'))
        hs.append(mesh.hmin())
    rate, _ = np.polyfit(np.log(hs), np.log(errors), deg=1)

    assert np.round(rate, 2) > 0.9*deg
    
    return errors


@pytest.mark.parametrize('deg', [1, 2, 3])
def test_elastic(deg, mu=1, lmbda=1):
    import sympy as sp
    import ulfy

    mesh = UnitSquareMesh(1, 1)
    x, y = SpatialCoordinate(mesh)

    u = as_vector((sin(pi*(x-y**2)),
                   cos(pi*(x**2+y))))

    mu, lmbda = Constant(mu), Constant(lmbda)
    sigma = mu*grad(u) + lmbda*div(u)*Identity(2)
    f = -div(sigma)

    mu_, lmbda_ = sp.symbols('mu lmbda')
    subs = {mu: mu_, lmbda: lmbda_}
    u, f = ulfy.Expression(u, degree=deg+3, subs=subs), ulfy.Expression(f, degree=deg+3, subs=subs)
    u.mu, f.mu = mu(0), mu(0)
    u.lmbda, f.lmbda = lmbda(0), lmbda(0)
    
    hs, errors = [], []
    for k in range(1, 6):
        n = 2**k
        mesh = UnitSquareMesh(n, n)

        boundaries = MeshFunction('size_t', mesh, 1, 0)
        DomainBoundary().mark(boundaries, 1)
        
        dirichlet_bcs = {1: u}
        V = VectorFunctionSpace(mesh, 'CG', deg)

        uh = elastic_extension(V, boundaries, dirichlet_bcs=dirichlet_bcs, f=f,
                               mu=mu, lmbda=lmbda)

        errors.append(errornorm(u, uh, 'H1'))
        hs.append(mesh.hmin())
    rate, _ = np.polyfit(np.log(hs), np.log(errors), deg=1)

    assert np.round(rate, 2) > 0.9*deg, rate
    
    return errors


def test_biharmonic():
    import sympy as sp
    import ulfy

    mesh = UnitSquareMesh(1, 1)
    x, y = SpatialCoordinate(mesh)

    u = sin(pi*x)*sin(pi*y)
    f = div(grad(div(grad(u))))

    u, f = ulfy.Expression(u, degree=5), ulfy.Expression(f, degree=5)
    
    hs, errors = [], []
    for k in range(1, 7):
        n = 2**k
        mesh = UnitSquareMesh(n, n)

        boundaries = MeshFunction('size_t', mesh, 1, 0)
        DomainBoundary().mark(boundaries, 1)
        
        dirichlet_bcs = {1: u}
        V = FunctionSpace(mesh, 'CG', 2)

        uh = biharmonic_extension(V, boundaries, dirichlet_bcs=dirichlet_bcs, f=f)

        errors.append(errornorm(u, uh, 'H1'))
        hs.append(mesh.hmin())
    rate, _ = np.polyfit(np.log(hs), np.log(errors), deg=1)

    assert np.round(rate, 2) > 1.5
    
    return errors

# --------------------------------------------------------------------

if __name__ == '__main__':

    print(test_biharmonic())

                
