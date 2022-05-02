import numpy as np


def idx2image_map(V, tol=1E-6):
    '''
    Restructe dofs of V into an image. The general idea for usage is 
    something like `Function(V).vector().get_local()[mapping]` would fill 
    an image. If V has subspaces mappings are returned for each subspace 
    and we should get a multichannel image.
    '''
    if V.num_sub_spaces() > 0:
        V0 = V.sub(0).collapse()
    else:
        V0 = V
    # Compute the ordering
    mesh = V0.mesh()
    assert mesh.topology().dim()
    
    gdim = mesh.geometry().dim()
    assert gdim == 2
        
    x = V0.tabulate_dof_coordinates().reshape((-1, gdim))

    index = np.argsort(x[:, 0] + 1j*x[:, 1])
    # Sort to have x constant first
    index = np.array(index)
    # To get some robustness in finite precision we check whether the
    # image is gridded by avoiding np.unique etc
    offsets = np.r_[0, 1 + np.where(np.diff(x[index, 0]) > tol)[0]]
    # So we get lengths of chunks of constant x
    # For each fixed x we have same tal slices
    _, = set(np.diff(offsets))
    # And they are equidistant
    x_ticks = x[index[offsets], 0]
    assert len(x_ticks) > 1
    assert np.all(abs(x_ticks[1] - x_ticks[0] - np.diff(x_ticks)) < tol)
    # So now we have the number of columns ...
    nrows = len(x_ticks)
    # ... and their spacing
    dx = x_ticks[1] - x_ticks[0]
    
    # We check the slicing in the other direction        
    index = index.reshape((nrows, -1))
    for irow in index:
        irow[:] = irow[np.argsort(x[irow, 1])]
        
    assert all(np.linalg.norm(x[index[i], 1]-x[index[i+1], 1]) < tol
               for i in range(nrows-1))
    y_ticks = x[index[0], 1]
    assert len(y_ticks) > 1
    assert np.all(abs(y_ticks[1] - y_ticks[0] - np.diff(y_ticks)) < tol)
    # So now we have the number of columns ...
    ncols = len(x_ticks)
    # ... and their spacing
    dy = y_ticks[1] - y_ticks[0]

    index = index.T

    if V.num_sub_spaces() > 0:
        index = [np.array(V.sub(i).dofmap().dofs())[index] for i in range(V.num_sub_spaces())]
    return (index, dx, dy)


# TODO: image2idx_map(V)
#       image_bdry_mask(V)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from nn_atlas.domains.cusp_square import CuspSquare
    import dolfin as df

    d = 0.9
    domain = CuspSquare(d=d, R1=1/d)
    mesh, _ = domain.get_reference_mesh(structured=True, resolution=0.2)
    df.File('foo.pvd') << mesh
    # mesh = df.UnitSquareMesh(4, 6)
    
    V = df.FunctionSpace(mesh, 'CG', 1)
    mapping, dx, dy = idx2image_map(V)
    x = df.interpolate(df.Expression('sqrt((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.25)*(x[1]-0.25))', degree=1), V)
    
    V = df.VectorFunctionSpace(mesh, 'CG', 1)
    mapping, dx, dy = idx2image_map(V)

    x = df.interpolate(df.Expression(('x[0]', 'x[1]'), degree=1), V)    
    vals = x.vector().get_local()
    vals0 = vals[mapping[0]]
    vals1 = vals[mapping[1]]    

    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(vals0)    
    ax[1].imshow(vals1)
    plt.show()
    
