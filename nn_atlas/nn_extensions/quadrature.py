import numpy as np
import itertools
import torch
import ffc


# For mesh based we in general rely on FIAT simplex quadratures
def get_volume_quadrature_mesh(volume_subdomains, degree, subdomain=-1, return_tangent=False):
    '''dV quadrature using underlying triangulation'''
    # NOTE: If topological dim of the mesh is 1 we [return]
    # quad points, weights, and tangent vectors at quad points
    # Get it for FIAT reference elm which is (-1, -1) x (1, -1) x (-1, 1)
    return FIAT_map_from_reference(degree,
                                   volume_subdomains,
                                   subdomain,
                                   return_tangent=return_tangent)


def FIAT_map_from_reference(degree, subdomains, subdomain, return_tangent=False):
    '''
    Make the reference quadrature points and weights to subdomains entities 
    with tag
    '''
    # Convention that -1 is all nonzero
    if subdomain == -1:
        subdomain = tuple(set(subdomains.array()) - set((0, )))
    elif isinstance(subdomain, int):
        subdomain = (subdomain, )

    assert set(subdomain) <= set(subdomains.array())

    tdim = subdomains.mesh().topology().dim()
    # Volume integrals
    if subdomains.dim() == tdim:
        # Pick
        if tdim == 1:
            cell = ffc.fiatinterface.FIAT.quadrature.reference_element.DefaultLine()
            reference_elm_quad = ffc.fiatinterface.FIAT.quadrature.GaussJacobiQuadratureLineRule(cell, degree)
        elif tdim == 2:
            cell = ffc.fiatinterface.FIAT.quadrature.reference_element.DefaultTriangle()
            reference_elm_quad = ffc.fiatinterface.FIAT.quadrature.CollapsedQuadratureTriangleRule(cell, degree)
        elif tdim == 3:
            cell = ffc.fiatinterface.FIAT.quadrature.reference_element.DefaultTetrahedron()
            reference_elm_quad = ffc.fiatinterface.FIAT.quadrature.CollapsedQuadratureTetrahedronRule(cell, degree)
        else:
            raise ValueError
        xq, wq = np.array(reference_elm_quad.pts), np.array(reference_elm_quad.wts)
        
        return map_volume_from_reference(xq, wq, subdomains, subdomain, return_tangent)

    # Facet integrals
    if subdomains.dim() == tdim-1:
        # Pick
        if tdim == 2:
            cell = ffc.fiatinterface.FIAT.quadrature.reference_element.DefaultLine()
            reference_elm_quad = ffc.fiatinterface.FIAT.quadrature.GaussJacobiQuadratureLineRule(cell, degree)
        elif tdim == 3:
            cell = ffc.fiatinterface.FIAT.quadrature.reference_element.DefaultTriangle()
            reference_elm_quad = ffc.fiatinterface.FIAT.quadrature.CollapsedQuadratureTriangleRule(cell, degree)
        else:
            # Boundary of 1d is points
            mesh = subdomains.mesh()
            array = subdomains.array()
            x = mesh.coordinates()
            
            mapped_xq  = None
            for tag in subdomain:
                for facet in np.where(array == tag)[0]:
                    xq = x[facet]
                    if mapped_xq is None:
                        mapped_xq = xq
                    else:
                        mapped_xq = np.row_stack([mapped_xq, xq])
            mapped_wq = np.ones(len(mapped_xq))
                        
            return torch.tensor([mapped_xq]), torch.tensor([mapped_wq])

        # Work for it!
        xq, wq = np.array(reference_elm_quad.pts), np.array(reference_elm_quad.wts)
        
        return map_facet_from_reference(xq, wq, subdomains, subdomain)

    raise ValueError    


def map_volume_from_reference(xq, wq, subdomains, subdomain, return_tangent=False):
    '''Assuming elements of FIAT'''
    tdim = subdomains.mesh().topology().dim()
    # Requested and makes sense
    return_tangent = return_tangent and tdim == 1
    # I guess it only makes sense to speak of the tangent it it is consistent
    # form cell to cell. The way we compute the tangent down, the orientation
    # is ensured iff the cells are "linkder"
    if return_tangent:
        # [1, 2], [2, 3], ... Otherwise it will throw
        link0, link1 = set(np.diff(np.concatenate(subdomains.mesh().cells())))
    
    mesh = subdomains.mesh()
    array = subdomains.array()
    x = mesh.coordinates()

    nvtx = tdim + 1
    
    mapped_xq, mapped_wq, tangents = None, None, None
    for tag in subdomain:
        for cell in mesh.cells()[np.where(array == tag)]:
            cell_x = x[cell]
            o = cell_x[0]
            # In (0, 0)-(1, 0)-(0, 1) the form of trasformation is
            # based on coordinates wrt axis that ara B-A, C-A, ...
            # In coordinates (-1, -1)-(-1, 1)-(1, -1) we basically
            # need to add (1, 1, ...) and divide by 2 first
            T = 0.5*np.column_stack([cell_x[i] - o for i in range(1, nvtx)])
            yq_ = np.array([np.dot(T, xqi+np.ones_like(xqi)) for xqi in xq]) + o
            # NOTE: hopefully this will get us covered in manifolds too
            wq_ = np.sqrt(np.linalg.det(np.dot(T.T, T)))*wq
            # I compute it but might not use
            if return_tangent:
                tau = np.diff(cell_x, axis=0).squeeze(0)
                tau = tau / np.linalg.norm(tau)
            
                tq_ = np.repeat(np.array([tau]), len(xq), axis=0)
            else:
                tq_ = []

            if mapped_xq is None:
                mapped_xq, mapped_wq, tangents = yq_, wq_, tq_
            else:
                mapped_xq = np.row_stack([mapped_xq, yq_])
                mapped_wq = np.r_[mapped_wq, wq_]
                tangents = np.row_stack([tangents, tq_])

    if not return_tangent:
        return torch.tensor([mapped_xq]), torch.tensor([mapped_wq])
    # Full
    return torch.tensor([mapped_xq]), torch.tensor([mapped_wq]), torch.tensor([tangents])


def map_facet_from_reference(xq, wq, subdomains, subdomain):
    '''Assuming elements of FIAT'''
    tdim = subdomains.mesh().topology().dim()
    
    mesh = subdomains.mesh()
    array = subdomains.array()
    x = mesh.coordinates()
    # We shall look up connected cell to orient the facet
    _, f2c = (mesh.init(tdim-1, tdim), mesh.topology()(tdim-1, tdim))
    _, f2v = (mesh.init(tdim-1, 0), mesh.topology()(tdim-1, 0))
    _, c2v = (mesh.init(tdim, 0), mesh.topology()(tdim, 0))

    nvtx = tdim
    
    mapped_xq, mapped_wq, normals = None, None, None
    for tag in subdomain:
        for facet in np.where(array == tag)[0]:
            cell, = f2c(facet)
            facet_x = x[f2v(facet)]
            o = facet_x[0]
            # For compute the normal taking the cell mid as inside
            mid = np.mean(x[c2v(cell)], axis=0)

            nq_ = np.repeat(np.array([normal_vector(facet_x, mid)]),
                            len(xq),
                            axis=0)

            # Transoform is as before
            T = 0.5*np.column_stack([facet_x[i] - o for i in range(1, nvtx)])
            yq_ = np.array([np.dot(T, xqi+np.ones_like(xqi)) for xqi in xq]) + o
            wq_ = np.sqrt(np.linalg.det(np.dot(T.T, T)))*wq

            if mapped_xq is None:
                mapped_xq, mapped_wq, normals = yq_, wq_, nq_
            else:
                mapped_xq = np.row_stack([mapped_xq, yq_])
                mapped_wq = np.r_[mapped_wq, wq_]
                normals = np.row_stack([normals, nq_])

    return torch.tensor([mapped_xq]), torch.tensor([mapped_wq]), torch.tensor([normals])

