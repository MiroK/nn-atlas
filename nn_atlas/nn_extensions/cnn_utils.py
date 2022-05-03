import numpy as np


def set_image(u, img=None, tol=1E-6):
    '''u->img'''
    V = u.function_space()
    mapping, dx, dy = idx2image_map(V, tol=tol)

    if img is None:
        img = np.zeros(mapping[0].shape + (len(mapping), ), dtype=float)

    if len(mapping) == 1:
        assert (img.ndim == 3 and img.shape[2] == 1) or img.ndim == 2

        index = mapping[0]
        if img.ndim == 3:
            img[:, :, 0] = u.vector().get_local()[index]
        else:
            img[:, :] = u.vector().get_local()[index]
        return img

    assert img.ndim == 3 and img.shape[2] == len(mapping)

    for channel, index in enumerate(mapping):
        img[:, :, channel] = u.vector().get_local()[index]

    return img
    

def set_function(img, u, tol=1E-6):
    '''img->u'''
    V = u.function_space()
    mapping, dx, dy = idx2image_map(V, tol=tol)

    u_vec = u.vector()
    u_arr = u_vec.get_local()

    if img.ndim == 2:
        index = mapping[0]
        u_arr[index.flatten()] = img.flatten()
    else:
        assert img.shape[2] == len(mapping)
        for channel, index in enumerate(mapping):
            u_arr[index.flatten()] = img[:, :, channel].flatten()
    # Update
    u_vec.set_local(u_arr)
    u_vec.apply('insert')
            
    return u


def idx2image_map(V, tol=1E-6):
    '''
    Restructure dofs of V into an image. The general idea for usage is 
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

    index = np.rot90(index, k=1)
    if V.num_sub_spaces() > 0:
        index = [np.array(V.sub(i).dofmap().dofs())[index] for i in range(V.num_sub_spaces())]
    else:
        index = [index]
    return (index, dx, dy)


# TODO: 
#       image_bdry_mask(V)

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt    
    import dolfin as df

    mesh = df.UnitSquareMesh(32, 30)

    # Scalars 
    V = df.FunctionSpace(mesh, 'CG', 1)
    x = df.interpolate(df.Expression('-3*x[0]+4*x[1]', degree=1), V)

    mapping, dx, dy = idx2image_map(V)
    
    vals0 = set_image(x)
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(vals0)
    df.plot(x)

    # The other way around    
    x.vector()[:] *= 0
    set_function(vals0, x)
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(vals0)
    df.plot(x)
    
    plt.show()

    # Vectors
    V = df.VectorFunctionSpace(mesh, 'CG', 1)
    x = df.interpolate(df.Expression(('-3*x[0]+4*x[1]', '-5*x[0]+2*x[1]'), degree=1), V)
    
    mapping, dx, dy = idx2image_map(V)
    
    vals0 = set_image(x)
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.linalg.norm(vals0, 2, axis=2))
    df.plot(df.sqrt(df.inner(x, x)))

    # The other way around    
    x.vector()[:] *= 0
    set_function(vals0, x)
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.linalg.norm(vals0, 2, axis=2))
    df.plot(df.sqrt(df.inner(x, x)))
    
    plt.show()
