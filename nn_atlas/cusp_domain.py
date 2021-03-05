from nn_atlas.gmsh_interopt import *
import numpy as np
import gmsh


def reference_square(structured=False, resolution=1.):
    '''This is a nice unit square'''
    #  -1-2-
    #  5   3
    #    4
    gmsh.initialize()

    model = gmsh.model
    factory = model.occ

    pts = [factory.addPoint(0, 0, 0),
           factory.addPoint(0.5, 0, 0),
           factory.addPoint(1, 0, 0),
           factory.addPoint(1, -0.5, 0),           
           factory.addPoint(1, -1, 0),
           factory.addPoint(0.5, -1, 0),           
           factory.addPoint(0, -1, 0),
           factory.addPoint(0, -0.5, 0),           
    ]

    n = len(pts)
    lines = [factory.addLine(pts[i], pts[(i+1)%n])
             for i in range(len(pts))]

    c = factory.addPoint(0.5, -0.5, 0)
    cross = [factory.addLine(pts[-1], c),
             factory.addLine(pts[1], c),
             factory.addLine(c, pts[3]),
             factory.addLine(c, pts[5])]
    
    factory.synchronize()
    # 0|1
    # --
    # 2|3
    loop0 = factory.addCurveLoop([lines[0], cross[1], -1*cross[0], lines[-1]])
    domain0 = factory.addPlaneSurface([loop0])

    loop1 = factory.addCurveLoop([lines[1], lines[2], -cross[2], -1*cross[1]])
    domain1 = factory.addPlaneSurface([loop1])

    loop2 = factory.addCurveLoop([lines[5], lines[6], cross[0], cross[-1]])
    domain2 = factory.addPlaneSurface([loop2])

    loop3 = factory.addCurveLoop([lines[3], lines[4], -cross[-1], cross[2]])
    domain3 = factory.addPlaneSurface([loop3])

    domains = (domain0, domain1, domain2, domain3)
    factory.synchronize()

    if structured:
        [model.mesh.setTransfiniteSurface(d) for d in domains[:-1]]
        model.mesh.setTransfiniteSurface(domains[-1], 'Right')
        
    model.addPhysicalGroup(2, domains, tag=1)

    model.addPhysicalGroup(1, [lines[0]], 1)
    model.addPhysicalGroup(1, [lines[1]], 2)    
    model.addPhysicalGroup(1, [lines[2], lines[3]], 3)
    model.addPhysicalGroup(1, [lines[4], lines[5]], 4)
    model.addPhysicalGroup(1, [lines[6], lines[7]], 5)
    
    factory.synchronize()

    gmsh.fltk.initialize()
    gmsh.fltk.run()
    
    nodes, topologies = msh_gmsh_model(model,
                                       2,
                                       # Globally refine
                                       number_options={'Mesh.CharacteristicLengthFactor': resolution})
    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions


def cusp_domain(d, R1, R2=None, uniform_mesh=True, resolution=1.0):
    '''
    Square [0, 1]^2 with cusp on top edge formed by two circle arcs.
     /\
    /   \
    |    |
    |    |
    |----|
    '''
    assert d > 0
    assert R1 > np.sqrt(d**2 + 0.5**2)/2
    assert R2 > np.sqrt(d**2 + 0.5**2)/2    
    
    gmsh.initialize()

    model = gmsh.model
    factory = model.occ

    ul = factory.addPoint(0, 0, 0)
    ur = factory.addPoint(1, 0, 0)
    lr = factory.addPoint(1, -1, 0)
    ll = factory.addPoint(0, -1, 0)    

    # Auxiliary point to define both
    aux = factory.addPoint(0.5, d, 0)
    
    # Left circle
    if R1 < np.inf:
        alpha = np.arccos(np.sqrt(d**2 + 0.5**2)/2./R1)

        P = np.array([0.5, d])
        P = P/np.linalg.norm(P)
        rot = np.array([[np.cos(alpha), -np.sin(alpha)],
                        [np.sin(alpha), np.cos(alpha)]])
        # This is a direction
        center = np.dot(rot, P)
        center = R1*center
        
        center0 = factory.addPoint(*center, z=0)
        line0 = factory.addCircleArc(ul, center0, aux)
    else:
        line0 = factory.addLine(ul, aux)

    if R2 is None:
        R2 = R1

    if R2 < np.inf:
        #  Right circle    
        beta = np.arccos(np.sqrt(d**2 + 0.5**2)/2./R2)
    
        P = np.array([0.5, d]) - np.array([1, 0])
        P = P/np.linalg.norm(P)
        rot = np.array([[np.cos(beta), -np.sin(beta)],
                        [np.sin(beta), np.cos(beta)]])
        center = R2*np.dot(rot.T, P) + np.array([1, 0])
        
        center1 = factory.addPoint(*center, z=0)
        line1 = factory.addCircleArc(aux, center1, ur)
    else:
        line1 = factory.addLine(aux, ur)
        
    # # Linkt it     1/2\
    # #            5      3
    # #                4
    lines = [line0,
             line1,
             factory.addLine(ur, lr),
             factory.addLine(lr, ll),
             factory.addLine(ll, ul)]
    
    factory.synchronize()

    loop = factory.addCurveLoop(lines)
    domain = factory.addPlaneSurface([loop])

    factory.synchronize()
    
    model.addPhysicalGroup(2, [domain], tag=1)

    for tag, l in enumerate(lines, 1):
        model.addPhysicalGroup(1, [l], tag)

    if not uniform_mesh:
        model.occ.synchronize()
        model.geo.synchronize()
        
        model.mesh.field.add('Distance', 1)
        # Try to be finer in curve part
        model.mesh.field.setNumbers(1, 'CurvesList', [1, 2])
        model.mesh.field.setNumber(1, 'NumPointsPerCurve', 100)
        
        model.mesh.field.add('Threshold', 2)
        model.mesh.field.setNumber(2, 'InField', 1)        
        model.mesh.field.setNumber(2, 'SizeMax', 0.1)
        model.mesh.field.setNumber(2, 'SizeMin', 0.01)
        model.mesh.field.setNumber(2, 'DistMin', 0.1)
        model.mesh.field.setNumber(2, 'DistMax', 0.2)    

        model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.occ.synchronize()
        gmsh.model.geo.synchronize()    
    
    nodes, topologies = msh_gmsh_model(model,
                                       2,
                                       # Globally refine
                                       number_options={'Mesh.CharacteristicLengthFactor': resolution})
    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions


# harmonic extensions (displacement and/or phi)
# visualize - lmin, lmax, det, gridlines
# biharmonic
# elasticity
# harmonic with adaptive smoothing?
# (p laplace)

# How to train a neural network?

# reference_square(structured=True)

reference_square(structured=False, resolution=1.)
