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
        self.ndofs = mesh.num_vertices()
        
        self.weight = nn.Parameter(torch.Tensor(self.ndofs))
        self.register_parameter('bias', None)

        # Supports
        V = df.FunctionSpace(mesh, 'CG', 1)
        dm = V.dofmap()
        d2v = df.dof_to_vertex_map(V)
        x = mesh.coordinates()

        tol = 1E-10
        supports = {}
        for cell in range(mesh.num_cells()):
            dofs = dm.cell_dofs(cell)

            tri = x[d2v[dofs]]
            
            xmin, ymin = np.min(tri, axis=0)
            xmax, ymax = np.max(tri, axis=0)
            bbox = ((xmin-tol, xmax+tol), (ymin-tol, ymax+tol))
            
            A, B, C = tri
            M = np.c_[B-A, C-A]
            Minv = np.linalg.inv(M).T

            supports[cell] = (torch.tensor(Minv),
                              torch.tensor(A),
                              bbox,
                              torch.tensor(dofs, dtype=torch.int64))
            
        self.supports = supports
        self.tree = mesh.bounding_box_tree()
        
    def forward(self, x):

        tol = 1E-10

        out = torch.zeros_like(x[..., 0])

        for i, xi in enumerate(x.squeeze(0)):
            cell = self.tree.compute_first_collision(df.Point(xi))

            Minv, A, bbox, dofs = self.supports[cell]

            st = torch.matmul((xi - A), Minv)

            s, t = st[..., 0], st[..., 1]
            c0, c1, c2 = self.weight[dofs]

            #out[(0, i)] = (1-s-t)*c0 + s*c1 + t*c2

            # s = x * Minv
            # df_dx =  ds/dx * df/ds         [[-c0+c1;-c0 + c2]]

        return out

    def set_from_coefficients(self, coefs):
        '''Set the degrees of freedom'''
        with torch.no_grad():
            self.weight[:] = torch.tensor([coefs])

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
    timer = df.Timer('fenics')
    true = np.array([f(xi) for xi in x.detach().numpy().reshape(-1, 2)])
    print('Time to first', timer.stop())    
    print(np.linalg.norm(mine - true, np.inf))

    # x = torch.rand(1, 100, 2, dtype=torch.float64)
    # x.requires_grad = True
    # du = grad(p1(x), x)

    # print(du)

# FIXME: - wire up vector
#        - say we cache the gradient on forward pass, how to reuse during backward?
#
#
