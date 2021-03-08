import dolfin as df
import numpy as np


def mesh_interior_points(mesh):
    '''Interior points of the mesh'''
    idx = mesh_intior_points_idx(mesh)
    return mesh.coordinates()[idx]


def mesh_intior_points_idx(mesh):
    '''Auxiliary function that get interior vertices by index in mesh coorfs'''
    V = df.FunctionSpace(mesh, 'CG', 1)
    bc = df.DirichletBC(V, df.Constant(0), 'on_boundary')
    # As incides in dofmap
    all_dofs = np.arange(V.dim())
    interior_dofs = np.delete(all_dofs,
                              list(bc.get_boundary_values().keys()))
    # As indices in vertex map
    return df.dof_to_vertex_map(V)[interior_dofs]


def line_line(ref, target, f=lambda x: x):
    '''Mapping from reference line to target line'''
    tol = 1E-13
    # [A, B] => P = A + t*(B-A)  for t in (0, 1)
    # [a, b] = a + f(t)*(b-a) for t in (0, 1)
    assert np.abs(f(0)) < tol and np.abs(f(1)-1) < tol
    # It should also be invergible on (0, 1) but I don't check
    # for that
    def mapping(P, f=f, ref=ref, target=target):
        A, B = ref
        t = np.dot(P-A, B-A)/np.dot(B-A, B-A)
        assert -tol < t < 1+tol
        
        a, b = target
        return a + (b-a)*f(t)

    return mapping


def line_circleArc(ref, target, f=lambda x: x, orientation=0):
    '''Mapping from reference line to target circle arc'''
    tol = 1E-13

    a, center, b = target
    assert abs(np.linalg.norm(a-center) - np.linalg.norm(b-center)) < tol, \
        (np.linalg.norm(a-center), np.linalg.norm(b-center))
    # Let's build parametrization of cirlce arc
    radius = np.linalg.norm(a-center)
    alpha = -2*np.arcsin(0.5*np.linalg.norm(b-a)/radius)

    # For counterclockwise
    if orientation == 0:
        R = lambda t, alpha=alpha: np.array([[np.cos(t*alpha), np.sin(t*alpha)],
                                             [-np.sin(t*alpha), np.cos(t*alpha)]])
    else:
        R = lambda t, alpha=alpha: np.array([[np.cos(t*alpha), -np.sin(t*alpha)],
                                             [np.sin(t*alpha), np.cos(t*alpha)]])    

    def mapping(P, f=f, ref=ref, center=center, a=a, R=R):
        A, B = ref
        t = np.dot(P-A, B-A)/np.dot(B-A, B-A)
        assert -tol < t < 1+tol

        return center + np.dot(R(t), a - center)

    return mapping

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt


    ref = np.array([[0, 0],
                    [0.5, 0]])
                    
    target = np.array([[0.3, 0.0],
                       [0.25, 0.1],
                       [0.2, 0.0],
                       ])

    d = 0.92
    R1 = 1./d
    
    alpha = np.arccos(np.sqrt(d**2 + 0.5**2)/2./R1)

    P = np.array([0.5, d])
    P = P/np.linalg.norm(P)
    rot = np.array([[np.cos(alpha), -np.sin(alpha)],
                    [np.sin(alpha), np.cos(alpha)]])
    # This is a direction
    center = np.dot(rot, P)
    center = R1*center
    

    target = np.array([[0.0, 0.0],
                       center,
                       [0.5, d]])

    print(center)
    
    print(np.linalg.norm(center - target[0]))
    print(np.linalg.norm(center - target[-1]))    
    print(target[-1] - center)
    
    thetas = np.linspace(0, 2*np.pi, 1000)
    x = center[0] + R1*np.cos(thetas)
    y = center[1] + R1*np.sin(thetas)    

    plt.figure()
    fig, ax = plt.subplots()

    ax.plot(x, y)
    
    ax.plot(0.5*np.ones(10), np.linspace(0, d+0.1, 10))

    ax.plot(*target[0], 'ob')
    ax.plot(*target[1], 'xb')
    ax.plot(*target[2], 'xb')        

    
    plt.axis('equal')
    plt.show()
    
    
    f = line_circleArc(ref, target)

    fig, ax = plt.subplots()
    ax.plot(ref.T[0], ref.T[1], 'r')
    ax.plot(*ref[0], 'ro')    

    ax.plot(*target[0], 'ob')
    ax.plot(*target[1], 'xb')
    ax.plot(*target[2], 'xb')        

    A, B = ref
    t = np.linspace(0, 1, 200)
    for ti in t:
        x_ref = A + ti*(B-A)
        y_ref = f(x_ref)

        seg = np.array([x_ref, y_ref])
        #ax.plot(seg.T[0], seg.T[1], 'k')
        ax.plot(y_ref[0], y_ref[1], marker='x')

    ax.axis('equal')
    plt.show()
