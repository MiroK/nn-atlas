import dolfin as df
import numpy as np
import torch
import torch.nn as nn


from nn_atlas.nn_extensions.calculus import grad, logical_and


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


def hat_supports_mat(mesh, dtype=torch.float64):
    '''[Vertex, (points of the support of the hat function of vertex)]'''
    V = df.FunctionSpace(mesh, 'CG', 1)
    
    mesh = V.mesh()
    tdim = mesh.topology().dim()
    assert tdim == 2
    
    mesh.init(0, tdim)
    v2c = mesh.topology()(0, tdim)
    c2v = mesh.topology()(tdim, 0)
    x = mesh.coordinates()

    mappings = []
    # We represent a supports as 
    # (vertex, [Cells that are connected to it in terms of their vertices]
    for dof, vertex in enumerate(df.dof_to_vertex_map(V)):
        V0 = x[vertex]
        cs = v2c(vertex)
        # Bounding box
        xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
        support_mappings = []
        for c in cs:
            A, B = x[list(set(c2v(c)) - set((vertex, )))]

            xmin, xmax = min(xmin, min(A[0], B[0])), max(xmax, max(A[0], B[0]))
            ymin, ymax = min(ymin, min(A[1], B[1])), max(ymax, max(A[1], B[1]))
            
            T = np.column_stack([A-V0, B-V0])
            Tinv = np.linalg.inv(T).T

            b = -np.dot(V0, Tinv)
        
            support_mappings.append((torch.tensor(Tinv, dtype=dtype),
                                     torch.tensor(b, dtype=dtype)))

        mappings.append(((((torch.tensor([xmin, xmax], dtype=dtype)),
                           (torch.tensor([ymin, ymax], dtype=dtype)))), support_mappings))

    return mappings


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


def hat_function_mat(x, support, tol=1E-8):
    '''
    Neural net encoding of a hat function, 2 layers with 2 neurons 
    in first for the hats
      min(rho((x-x0)/(x1-x0)), rho((x2-x)/(x2-x1)))
    '''
    # NOTE: I assume here that x is three dimensional array
    batchdim, npts, gdim = x.shape
    assert gdim == 2

    # Add contributions from cells that form the support
    (bbox_x, bbox_y), support = support

    output = 0*x[..., 0]
    
    in_bbox = torch.where(logical_and(
       logical_and(bbox_x[0]-tol < x[..., 0], x[..., 0] < bbox_x[1]+tol),
       logical_and(bbox_y[0]-tol < x[..., 1], x[..., 1] < bbox_y[1]+tol)
    ))

    dtype = x.dtype
    reduce_ = torch.tensor(np.array([[-1., -1]]).T, dtype=dtype)

    if len(in_bbox[0]) == 0:
        return output
    
    x = x[in_bbox]
    ys = []

    for Tinv, b in support:
        # Map from x, y to reference
        y = torch.add(torch.matmul(x, Tinv), b)
        # 1 - X - Y
        y = torch.matmul(y, reduce_)
        # Break on the plane there
        y = torch.nn.ReLU()(torch.add(y, 1.))

        ys.append(y)
    ys = torch.stack(ys, axis=2)
    thing = torch.min(ys, axis=2, keepdim=False)[0].squeeze(-1)

    output[in_bbox] = thing

    return output

# Inherit from Function
class LinearFunction(torch.autograd.Function):

    foo = None
    
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        # print('>>>', input, weight)
        ctx.save_for_backward(input, weight, bias)
        # print(type(input), type(weight), input.shape, weight.shape)
        output = torch.matmul(input, weight.t())
        print('FOOO', LinearFunction.foo)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        print(grad_output, '<<<')
        
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        print('?', input.shape, weight.shape)        
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(weight, grad_output)
        if ctx.needs_input_grad[1]:
            print(input.shape, grad_output.shape)
            grad_weight = torch.matmul(input, grad_output)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

    
class ScalarP1FunctionSpace(nn.Module):
    '''Neural net that is 'CG1' function space on mesh'''
    def __init__(self, mesh, invalidate_cache=True):
        assert mesh.geometry().dim() == 2 and mesh.topology().dim() == 2
        super().__init__()
        # The weights are coefficients that combine the basis functions
        self.ndofs = mesh.num_vertices()
        
        self.weight = nn.Parameter(torch.Tensor(self.ndofs))
        self.register_parameter('bias', None)

        self.supports = hat_supports_mat(mesh)

        self.basis_cache = None
        self.invalidate_cache = invalidate_cache
        
    def forward(self, x):
        # Build up "Vandermonde", then each column is what
        ndofs = self.ndofs
        
        if self.basis_cache is None:
            # x.requires_grad = True
            
            bsize, npts  = x.shape[:2]
            basis_values = torch.zeros(bsize, npts, ndofs, dtype=x.dtype)
            
            for col, support in enumerate(self.supports):
                # print(support)
                basis_values[..., col] = hat_function_mat(x, support)
            self.basis_cache = basis_values
            # self.dbasis_cache = grad(basis_values, x)
            
        # out = LinearFunction.apply(basis_values, self.weight, None)
        out = torch.matmul(self.basis_cache, self.weight)        
        self.invalidate_cache and setattr(self, 'basis_cache', None)

        return out

    def set_from_coefficients(self, coefs):
        '''Set the degrees of freedom'''
        with torch.no_grad():
            self.weight[:] = torch.tensor([coefs])


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

            
# --------------------------------------------------------------------

if __name__ == '__main__':
    # from nn_atlas.nn_extensions.calculus import grad
    
    mesh = df.UnitSquareMesh(32, 32)

    # supports = hat_supports(mesh)

    # x = torch.rand(1, 1000, 2)
    # y = hat_function(x, support)

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

    # timer = df.Timer('second')
    # mine = p1(x).detach().numpy().flatten()    
    # print('Time to second', timer.stop())
    true = np.array([f(xi) for xi in x.detach().numpy().reshape(-1, 2)])
    print(np.linalg.norm(mine - true, np.inf))

    x.requires_grad = True
    du = grad(p1(x), x)
    
    # y = torch.rand(1, 10, 2, dtype=torch.float64)
    # y.requires_grad = True
    # print(grad(p1(y), y))
    # # print(grad(p1(y), y))
    # # print(grad(p1(y), y))    
    
    # ---

    # V = df.VectorFunctionSpace(mesh, 'CG', 1)
    
    # p1 = VectorP1FunctionSpace(mesh)
    # p1.double()

    # f = df.Expression(('x[0] + x[1]', 'x[0] - 2*x[1]'), degree=1)
    # coef = df.interpolate(f, V).vector().get_local()
    # p1.set_from_coefficients(coef)
    
    # x = torch.rand(1, 10000, 2, dtype=torch.float64)

    # timer = df.Timer('first')
    # mine = p1(x).detach().numpy().flatten()
    # print('Time to first', timer.stop())

    # timer = df.Timer('second')
    # mine = p1(x).detach().numpy().flatten()    
    # print('Time to second', timer.stop())
    # true = np.array([f(xi) for xi in x.detach().numpy().reshape(-1, 2)]).flatten()
    # print(np.linalg.norm(mine - true, np.inf))
