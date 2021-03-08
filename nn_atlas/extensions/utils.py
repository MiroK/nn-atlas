from functools import reduce
from operator import or_


def check_bcs(bdry, dirichlet, neumann):
    '''No overlaps in bcs and bcs make sense given data'''
    mesh = bdry.mesh()
    assert mesh.topology().dim() - 1 == bdry.dim()

    found_tags = set(bdry.array())  # Local
    found_tags = reduce(or_, mesh.mpi_comm().allgather(found_tags))

    assert set(dirichlet.keys()) <= found_tags
    assert set(neumann.keys()) <= found_tags
    
    assert not set(dirichlet.keys()) & set(neumann.keys())

    return True
