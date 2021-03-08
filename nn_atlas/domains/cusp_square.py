from nn_atlas.domains.utils import line_line, line_circleArc
from nn_atlas.domains.gmsh_interopt import *
from nn_atlas.domains.parametrized_domain import ParametrizedDomain
import numpy as np
import gmsh


class CuspSquare(ParametrizedDomain):
    '''
    This is a square [0, 1] x [-1, 0] with a cusp at the top 
    edge. The edge is halfed and the center point lifted by distance d. 
    The curves that join [0, 0], respectively [1, 0], with [0.5, d] are 
    circle arcs with radius R1 and R2.
   
    The boundary marking is enumerate((arc1, arc2, right, bottom, left), 1)
    '''
    def __init__(self, d, R1, R2=None):
        super().__init__()

        assert d > 0
        assert R1 > np.sqrt(d**2 + 0.5**2)/2
        assert R2 is None or R2 > np.sqrt(d**2 + 0.5**2)/2    
    
        self.d = d
        self.R1 = R1
        self.R2 = R1 if R2 is None else R2

        self._tags = (1, 2, 3, 4, 5)

    def get_reference_mesh(self, *, structured=False, uniform=False, resolution=1.):
        '''Build the reference mesh'''
        d, R1, R2 = self.d, self.R1, self.R2
    
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

        # NOTE: we do this to make transfinite work
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
        else:
            if not uniform:
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
        
        model.addPhysicalGroup(2, domains, tag=1)

        model.addPhysicalGroup(1, [lines[0]], 1)
        model.addPhysicalGroup(1, [lines[1]], 2)    
        model.addPhysicalGroup(1, [lines[2], lines[3]], 3)
        model.addPhysicalGroup(1, [lines[4], lines[5]], 4)
        model.addPhysicalGroup(1, [lines[6], lines[7]], 5)
        
        factory.synchronize()

        # gmsh.fltk.initialize()
        # gmsh.fltk.run()
        
        nodes, topologies = msh_gmsh_model(model,
                                           2,
                                       # Globally refine
                                           number_options={'Mesh.CharacteristicLengthFactor': resolution})
        mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

        gmsh.finalize()

        return mesh, entity_functions

    def get_target_mesh(self, *, structured=False, uniform=False, resolution=1.):
        '''Build the target mesh'''
        d, R1, R2 = self.d, self.R1, self.R2
    
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

        if not uniform:
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


    def ref2def_bdry_deformation(self, tag):    
        '''
        We are after mapping reference domains boundary to the cusp domain. 
        Here we return a function which for boundary of reference square return f: x -> y 
        where y belongs to the boundary of the cusp domain of the same 
        tag.
    '''
        if tag in (3, 4, 5):
            return lambda x: x

        d, R1, R2 = self.d, self.R1, self.R2
        if tag == 1:
            # The reference domain is (0, 0) ->- (0.5, 0)
            if R1 == np.inf:
                # We want to map to (0, 0) ->- (0.5, d) from (0, 0) ->- (0.5, 0)
                return line_line(ref=np.array([[0, 0.], [0.5, 0]]),
                                 target=np.array([[0, 0.], [0.5, d]]))
            else:
                # We want to map to circle arc through (0, 0), (0.5, d)
                # from (0, 0) ->- (0.5, 0)

                # Compute the center
                alpha = np.arccos(np.sqrt(d**2 + 0.5**2)/2./R1)

                P = np.array([0.5, d])
                P = P/np.linalg.norm(P)
                rot = np.array([[np.cos(alpha), -np.sin(alpha)],
                                [np.sin(alpha), np.cos(alpha)]])

                center = np.dot(rot, P)
                center = R1*center

                return line_circleArc(ref=np.array([[0, 0.], [0.5, 0]]),
                                      target=np.array([[0, 0.], center, [0.5, d]]))

        assert tag == 2
        # The reference domain is (0.5, 0) ->- (1, 0)
        if R2 == np.inf:
            # We want to map to (0.5, d) ->- (1, 0) 
            return line_line(ref=np.array([[0.5, 0.], [1.0, 0]]),
                             target=np.array([[0.5, d], [1.0, 0]]))
        else:
            # We want to map to circle arc through (0.5, d) ->- (1, 0)
            beta = np.arccos(np.sqrt(d**2 + 0.5**2)/2./R2)
            
            P = np.array([0.5, d]) - np.array([1, 0])
            P = P/np.linalg.norm(P)
            rot = np.array([[np.cos(beta), -np.sin(beta)],
                            [np.sin(beta), np.cos(beta)]])
            center = R2*np.dot(rot.T, P) + np.array([1, 0])

            return line_circleArc(ref=np.array([[0.5, 0.], [1.0, 0]]),
                                  target=np.array([[0.5, d], center, [1.0, 0]]))

    def ref_bdry_points(self, parameters, tag):
        '''Refeference domain boundary points based on their parametric coordinates'''
        #  ->1>-2-
        #  5     v
        #  ^     3
        #  --4-<-
        points = {1: np.array([[0, 0.], [0.5, 0]]),
                  2: np.array([[0.5, 0.], [1.0, 0]]),
                  3: np.array([[1.0, 0.], [1.0, -1.0]]),
                  4: np.array([[1.0, -1.0], [0, -1.0]]),
                  5: np.array([[0, -1.0], [0., 0.]])}

        if not len(parameters.shape) == 2:
            parameters = parameters.reshape((-1, 1))
        
        A, B = points[tag]

        return A + (B-A)*parameters

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # FIXME: What is the theoretical max value we can allow?
    d = 0.9
    R1 = 1./d
    R2 = 1./d

    domain = CuspSquare(d=d, R1=R1, R2=R2)
    
    rmesh, rentities = domain.get_reference_mesh(structured=False, resolution=1.0)
    rmesh_boundaries = rentities[1]

    tmesh, tentities = domain.get_target_mesh(uniform=False, resolution=1.0)
    tmesh_boundaries = tentities[1]

    X = rmesh.coordinates()
    _, Xe2v = (rmesh.init(1, 0), rmesh.topology()(1, 0))

    Y = tmesh.coordinates()
    _, Ye2v = (tmesh.init(1, 0), tmesh.topology()(1, 0))
    
    fig, ax = plt.subplots()
    for tag in (1, 2, 3, 4, 5):
        f = domain.ref2def_bdry_deformation(tag)

        tag_facets, = np.where(rmesh_boundaries.array() == tag)
        reference_x = X[np.unique(np.hstack([Xe2v(e) for e in tag_facets]))]
        idx = np.argsort(reference_x[:, 0])
        reference_x = reference_x[idx]

        y0 = np.array([f(xi) for xi in reference_x])
        l, = ax.plot(y0[:, 0], y0[:, 1], marker='x', linestyle='none')

        # Compare with deformed
        tag_facets, = np.where(tmesh_boundaries.array() == tag)
        reference_y = Y[np.unique(np.hstack([Ye2v(e) for e in tag_facets]))]
        idx = np.argsort(reference_y[:, 0])
        reference_y = reference_y[idx]

        ax.plot(reference_y[:, 0], reference_y[:, 1], color=l.get_color())

    plt.show()


    from dolfin import *

    Velm = VectorElement('Lagrange', triangle, 1)
    bc_map = domain.set_mapping_bcs(Velm,
                                    boundaries=rmesh_boundaries,
                                    tags=(1, 2, 3, 4, 5),
                                    mode='displacement')

    for tag in (1, 2, 3, 4, 5):
        f = domain.ref2def_bdry_deformation(tag)

        tag_facets, = np.where(rmesh_boundaries.array() == tag)
        reference_x = X[np.unique(np.hstack([Xe2v(e) for e in tag_facets]))]
        idx = np.argsort(reference_x[:, 0])
        reference_x = reference_x[idx]

        for xi in reference_x:
            assert np.linalg.norm(xi + bc_map(xi) - f(xi)) < 1E-15
