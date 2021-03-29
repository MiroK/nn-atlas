import numpy as np
import torch
import torch.nn as nn
import dolfin as df


class ScalarP1FunctionSpace(nn.Module):
    '''Neural net that is 'CG1' function space on mesh'''
    def __init__(self, mesh):
        assert mesh.geometry().dim() == 2 and mesh.topology().dim() == 2
        super().__init__()
        # The weights are coefficients that combine the basis functions
        self.weight = nn.Parameter(torch.Tensor(mesh.num_vertices()))
        self.register_parameter('bias', None)

        # Supports
        V = df.FunctionSpace(mesh, 'CG', 1)
        dm = V.dofmap()
        d2v = df.dof_to_vertex_map(V)
        x = mesh.coordinates()

        tol = 1E-10
        supports = []
        for cell in range(mesh.num_cells()):
            dofs = dm.cell_dofs(cell)

            tri = x[d2v[dofs]]
            
            xmin, ymin = np.min(tri, axis=0)
            xmax, ymax = np.max(tri, axis=0)
            bbox = ((xmin-tol, xmax+tol), (ymin-tol, ymax+tol))
            
            A, B, C = tri
            M = np.c_[B-A, C-A]
            Minv = np.linalg.inv(M).T

            supports.append((torch.tensor(Minv),
                             torch.tensor(A),
                             bbox,
                             torch.tensor(dofs, dtype=torch.int64)))
            
        self.supports = supports
        
    def forward(self, x):

        tol = 1E-10

        out = torch.zeros_like(x[..., 0])
        # NOTE: I implemented this also in a way that the points found
        # inside the cell won't be searched for further. It seems the 
        # overhead of the bookkeeping (nested indexing) makes things slower
        # Also, doing it the fenics way, i.e. using mesh.bounding_box_tree
        # to find the colliding cell is about 2x slower. So atm we are 10x
        # slower than fenics

        # IDEA: a way to speed up could be caching the cell for point once
        # we find it. Or reusing arrays and don't allocase new ones. Also,
        # in more recent pytorch there is logical_and - I'd hope that it
        # shortcircuits
        for Minv, A, bbox, dofs in self.supports:
            inside_bbox = torch.where(torch.mul((bbox[0][0] < x[..., 0])*(x[..., 0] < bbox[0][1]),
                                                (bbox[1][0] < x[..., 1])*(x[..., 1] < bbox[1][1])))

            x_ = x[inside_bbox]
            # Coordinates with respect to the cell
            st = torch.matmul((x_-A), Minv)

            s, t = st[..., 0], st[..., 1]
            c0, c1, c2 = self.weight[dofs]
            # For interior points ...
            inside  = torch.where((-tol < s)*(-tol < t)*(s + t < 1+tol))
            s, t = s[inside], t[inside]
            # We obtain the value by evaluating the local basis matrix
            # at x and dot with coefficients
            out[(inside_bbox[0][inside],
                 inside_bbox[1][inside])] = (1-s-t)*c0 + s*c1 + t*c2
            # s = x * Minv
            # df_dx =  ds/dx * df/ds    
            # print(torch.matmul(Minv, torch.tensor([[-c0 + c1], [-c0 + c2]])))

        return out

    def set_from_coefficients(self, coefs):
        '''Set the degrees of freedom'''
        with torch.no_grad():
            self.weight[:] = torch.tensor([coefs])


class VectorP1FunctionSpace(nn.Module):
    '''Neural net that is 'CG1' function space on mesh'''
    def __init__(self, mesh):
        assert mesh.geometry().dim() == 2 and mesh.topology().dim() == 2
        super().__init__()
        # The weights are coefficients that combine the basis functions
        self.weight_x = nn.Parameter(torch.Tensor(mesh.num_vertices()))
        self.weight_y = nn.Parameter(torch.Tensor(mesh.num_vertices()))        
        self.register_parameter('bias', None)

        # Supports
        V = df.FunctionSpace(mesh, 'CG', 1)
        dm = V.dofmap()
        d2v = df.dof_to_vertex_map(V)
        x = mesh.coordinates()

        tol = 1E-10
        supports = []
        # Lookup will be based on scalar
        for cell in range(mesh.num_cells()):
            dofs = dm.cell_dofs(cell)

            tri = x[d2v[dofs]]
            
            xmin, ymin = np.min(tri, axis=0)
            xmax, ymax = np.max(tri, axis=0)
            bbox = ((xmin-tol, xmax+tol), (ymin-tol, ymax+tol))
            
            A, B, C = tri
            M = np.c_[B-A, C-A]
            Minv = np.linalg.inv(M).T

            supports.append((torch.tensor(Minv),
                             torch.tensor(A),
                             bbox,
                             torch.tensor(dofs, dtype=torch.int64)))
            
        self.supports = supports
        # We then combine things for vectors
        V = df.VectorFunctionSpace(mesh, 'CG', 1)
        self.dofs_x = V.sub(0).dofmap().dofs()
        self.dofs_y = V.sub(1).dofmap().dofs()
        
    def forward(self, x):

        tol = 1E-10

        out_x, out_y = torch.zeros_like(x[..., 0]), torch.zeros_like(x[..., 0])
        for Minv, A, bbox, dofs in self.supports:
            inside_bbox = torch.where(torch.mul((bbox[0][0] < x[..., 0])*(x[..., 0] < bbox[0][1]),
                                                (bbox[1][0] < x[..., 1])*(x[..., 1] < bbox[1][1])))

            x_ = x[inside_bbox]
            # Coordinates with respect to the cell
            st = torch.matmul((x_-A), Minv)

            s, t = st[..., 0], st[..., 1]
            # For interior points ...
            inside  = torch.where((-tol < s)*(-tol < t)*(s + t < 1+tol))
            s, t = s[inside], t[inside]

            idx0, idx1 = inside_bbox[0][inside], inside_bbox[1][inside]
            # We obtain the value by evaluating the local basis matrix
            # at x and dot with coefficients
            c0x, c1x, c2x = self.weight_x[dofs]
            out_x[(idx0, idx1)] = (1-s-t)*c0x + s*c1x + t*c2x

            c0y, c1y, c2y = self.weight_y[dofs]
            out_y[(idx0, idx1)] = (1-s-t)*c0y + s*c1y + t*c2y
            
            # s = x * Minv
            # df_dx =  ds/dx * df/ds    
            #torch.stack([
            #    torch.matmul(Minv, torch.tensor([[-c0x + c1x], [-c0x + c2x]])),
            #    torch.matmul(Minv, torch.tensor([[-c0y + c1y], [-c0y + c2y]]))                
            #, axis=2))

        return torch.stack([out_x, out_y], axis=2)

    def set_from_coefficients(self, coefs):
        '''Set the degrees of freedom'''
        with torch.no_grad():
            self.weight_x[:] = torch.tensor([coefs[self.dofs_x]])
            self.weight_y[:] = torch.tensor([coefs[self.dofs_y]])

# --------------------------------------------------------------------

if __name__ == '__main__':
    from nn_atlas.nn_extensions.calculus import grad
    
    mesh = df.UnitSquareMesh(32, 32)

    # supports = hat_supports(mesh)

    # x = torch.rand(1, 1000, 2)
    # y = hat_function(x, support)

    V = df.FunctionSpace(mesh, 'CG', 1)
    
    p1 = ScalarP1FunctionSpace(mesh)
    p1.double()

    f = df.Expression('-x[0] + 3*x[1]', degree=1)
    coef = df.interpolate(f, V).vector().get_local()
    p1.set_from_coefficients(coef)
    
    x = torch.rand(1, 30000, 2, dtype=torch.float64)

    timer = df.Timer('first')
    mine = p1(x).detach().numpy().flatten()
    print('Time to first', timer.stop())

    # timer = df.Timer('second')
    # mine = p1(x).detach().numpy().flatten()    
    # print('Time to second', timer.stop())
    timer = df.Timer('fenics')
    true = np.array([f(xi) for xi in x.detach().numpy().reshape(-1, 2)])
    print('Time to first', timer.stop())    
    print(np.linalg.norm(mine - true, np.inf))

    #x = torch.rand(1, 10, 2, dtype=torch.float64)
    #x.requires_grad = True
    #du = grad(p1(x), x)
    #print(du)
    
    # ----------------------------------------------------------------
    
    V = df.VectorFunctionSpace(mesh, 'CG', 1)
    
    p1 = VectorP1FunctionSpace(mesh)
    p1.double()

    f = df.Expression(('x[0] + x[1]', 'x[0] - 2*x[1]'), degree=1)
    coef = df.interpolate(f, V).vector().get_local()
    p1.set_from_coefficients(coef)
    
    x = torch.rand(1, 30000, 2, dtype=torch.float64)

    timer = df.Timer('first')
    mine = p1(x).detach().numpy().flatten()
    print('Time to first', timer.stop())

    timer = df.Timer('first')    
    true = np.array([f(xi) for xi in x.detach().numpy().reshape(-1, 2)]).flatten()
    print('Time to first', timer.stop())    
    print(np.linalg.norm(mine - true, np.inf))

    #x = torch.rand(1, 10, 2, dtype=torch.float64)
    #x.requires_grad = True
    #du = grad(p1(x), x)
    #print(du)

    # print(du)

# FIXME: - wire up vector
#        - say we cache the gradient on forward pass, how to reuse during backward?
#
#
