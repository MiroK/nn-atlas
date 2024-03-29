import dolfin as df
import numpy as np
import torch
import torch.nn as nn


def hat_supports(mesh):
    '''[Vertex, (points of the support of the hat function of vertex)]'''
    V = df.FunctionSpace(mesh, 'CG', 1)
    
    mesh = V.mesh()
    tdim = mesh.topology().dim()
    
    mesh.init(0, tdim)
    v2c = mesh.topology()(0, tdim)
    c2v = mesh.topology()(tdim, 0)
    x = mesh.coordinates()

    supports = []
    # We represent a supports as 
    # (vertex, [Cells that are connected to it in terms of their vertices]
    for dof, vertex in enumerate(df.dof_to_vertex_map(V)):
  
        cs = v2c(vertex)

        vertex_cells = []
        for c in cs:
            v = list(set(c2v(c)) - set((vertex, )))
            vertex_cells.append(x[v])
        supports.append((x[vertex], vertex_cells))
        
    return supports


def hat_function(x, support):
    '''
    Neural net encoding of a hat function, 2 layers with 2 neurons 
    in first for the hats
      min(rho((x-x0)/(x1-x0)), rho((x2-x)/(x2-x1)))
    '''
    # NOTE: I assume here that x is three dimensional array
    batchdim, npts, gdim = x.shape
    assert gdim == 2

    dtype = x.dtype
    reduce_ = torch.tensor(np.array([[-1., -1]]).T, dtype=dtype)
    # The cell is
    # A
    # |\
    # | \
    # |  \
    # V0--B and we want to map it to reference A = (0, 1), B = (1, 0), V0 = (0, 0)
    V0, cells = support
    
    ys = []
    # Add contributions from cells that form the support
    for A, B in cells:
        # Map from x, y to reference
        T = np.column_stack([A-V0, B-V0])

        Tinv = torch.tensor(np.linalg.inv(T).T, dtype=dtype)
        b = torch.tensor(-(V0.dot(Tinv)), dtype=dtype)
        y = torch.add(torch.matmul(x, Tinv), b)
        # 1 - X - Y
        y = torch.matmul(y, reduce_)
        # Break on the plane there
        y = torch.nn.ReLU()(torch.add(y, 1.))

        ys.append(y)
    ys = torch.stack(ys, axis=2)
    # As a result we get (batchdim, npts) as 1 scalar for the basis
    # function
    return torch.min(ys, axis=2, keepdim=False)[0].squeeze(2)


class VectorP1FunctionSpace(nn.Module):
    '''Neural net that is ['CG1']^2 function space on mesh'''    
    def __init__(self, mesh, invalidate_cache=True):
        assert mesh.geometry().dim() == 2 and mesh.topology().dim() == 2
        
        super().__init__()
        # How we combine x component ...
        self.lin_x = nn.Linear(mesh.num_vertices(), 1, bias=False)
        # ... and y component
        self.lin_y = nn.Linear(mesh.num_vertices(), 1, bias=False)
        
        self.supports = hat_supports(mesh)
        self.basis_cache = None
        self.invalidate_cache = invalidate_cache

        V = df.VectorFunctionSpace(mesh, 'CG', 1)
        self.dofs_x = V.sub(0).dofmap().dofs()
        self.dofs_y = V.sub(1).dofmap().dofs()
        
    def forward(self, x):
        # Build up "Vandermonde", then each column is what
        ndofs = len(self.lin_x.weight[0])
        if self.basis_cache is None:
            bsize, npts  = x.shape[:2]
            basis_values = torch.zeros(bsize, npts, ndofs, dtype=x.dtype)

            for col, support in enumerate(self.supports):
                a = hat_function(x, support)
                basis_values[..., col] = hat_function(x, support)
            self.basis_cache = basis_values
            
        x_component = self.lin_x(self.basis_cache).squeeze(2)
        y_component = self.lin_y(self.basis_cache).squeeze(2)

        out = torch.stack([x_component, y_component], axis=2)

        self.invalidate_cache and setattr(self, 'basis_cache', None)

        return out

    def set_from_coefficients(self, coefs):
        '''Set the degrees of freedom'''
        with torch.no_grad():
            self.lin_x.weight[0] = torch.tensor([coefs[self.dofs_x]])
            self.lin_y.weight[0] = torch.tensor([coefs[self.dofs_y]])            


class ScalarP1FunctionSpace(nn.Module):
    '''Neural net that is 'CG1' function space on mesh'''
    def __init__(self, mesh, invalidate_cache=True):
        assert mesh.geometry().dim() == 2 and mesh.topology().dim() == 2
        super().__init__()
        # The weights are coefficients that combine the basis functions
        self.lin = nn.Linear(mesh.num_vertices(), 1, bias=False)
        self.ndofs = mesh.num_vertices()
        self.supports = hat_supports(mesh)

        self.basis_cache = None
        self.invalidate_cache = invalidate_cache
        
    def forward(self, x):
        # Build up "Vandermonde", then each column is what
        ndofs = len(self.lin.weight[0])
        if self.basis_cache is None:
            bsize, npts  = x.shape[:2]
            basis_values = torch.zeros(bsize, npts, ndofs, dtype=x.dtype)
            
            for col, support in enumerate(self.supports):
                a = hat_function(x, support)
                basis_values[..., col] = hat_function(x, support)
            self.basis_cache = basis_values

        out = self.lin(self.basis_cache).squeeze(2)
        #  npts x ndofs
        # nvecs x npts x ndofs   .  weigths
        # nvecs x npts <---
        #

        self.invalidate_cache and setattr(self, 'basis_cache', None)

        return out

    def set_from_coefficients(self, coefs):
        '''Set the degrees of freedom'''
        with torch.no_grad():
            self.lin.weight[0] = torch.tensor([coefs])
            
# --------------------------------------------------------------------

if __name__ == '__main__':
    from nn_atlas.nn_extensions.calculus import grad
    
    mesh = df.UnitSquareMesh(10, 10)
    V = df.FunctionSpace(mesh, 'CG', 1)
    
    p1 = ScalarP1FunctionSpace(mesh)
    p1.double()

    f = df.Expression('x[0] + 3*x[1]', degree=1)
    coef = df.interpolate(f, V).vector().get_local()
    p1.set_from_coefficients(coef)
    
    x = torch.rand(1, 10000, 2, dtype=torch.float64)

    timer = df.Timer('first')
    mine = p1(x).detach().numpy().flatten()
    print('Time to first', timer.stop())

    timer = df.Timer('second')
    mine = p1(x).detach().numpy().flatten()    
    print('Time to second', timer.stop())
    true = np.array([f(xi) for xi in x.detach().numpy().reshape(-1, 2)])
    print('Error', np.linalg.norm(mine - true, np.inf))

    y = torch.rand(1, 10, 2, dtype=torch.float64)
    y.requires_grad = True
    print(grad(p1(y), y))
    print(grad(p1(y), y))
    print(grad(p1(y), y))    
    
    # # ---

    Vh = df.VectorFunctionSpace(mesh, 'CG', 1)
    
    p1 = VectorP1FunctionSpace(mesh)
    p1.double()

    f = df.Expression(('x[0] + 3*x[1]', 'x[0] - 2*x[1]'), degree=1)
    coef = df.interpolate(f, Vh).vector().get_local()
    p1.set_from_coefficients(coef)
    
    x = torch.rand(1, 10000, 2, dtype=torch.float64)

    timer = df.Timer('first')
    mine = p1(x).detach().numpy().flatten()
    print('Time to first', timer.stop())

    timer = df.Timer('second')
    mine = p1(x).detach().numpy().flatten()    
    print('Time to second', timer.stop())
    true = np.array([f(xi) for xi in x.detach().numpy().reshape(-1, 2)]).flatten()
    print(np.linalg.norm(mine - true, np.inf), 'Error')

    # Attempt at plot
    x_ = torch.linspace(0, 1, 100, dtype=x.dtype)
    y_ = torch.linspace(0, 1, 100, dtype=x.dtype)
    X, Y = torch.meshgrid(x_, y_)

    grid_points = torch.column_stack([torch.ravel(X), torch.ravel(Y)]).reshape(1, -1, 2)

    values = p1(grid_points)
    U, V = values[..., 0], values[..., 1]
    U = U.reshape(X.shape)
    V = V.reshape(X.shape)

    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 2)
    mappable = df.common.plotting.mplot_function(ax[0], df.interpolate(f, Vh))
    plt.colorbar(mappable, ax=ax[0])

    stride = 10
    X_, Y_, U_, V_ = (t.detach().numpy()[::stride, ::stride] for t in (X, Y, U, V))

    C_ = np.sqrt(U_**2 + V_**2)
    mappable = ax[1].quiver(X_, Y_, U_, V_, C_)
    plt.colorbar(mappable, ax=ax[1])

