import dolfin as df


class ParametrizedDomain(object):
    '''We will look for mapping reference to deformed (paremetrized domain)'''
    def get_reference_mesh(self, *, structured, uniform, resolution):
        '''Mesh of the domain of the mapping'''
        # structured ~ transfinite (takes precedence over uniform)
        # uniform mesh has rougly same mesh size everywhere
        # non-uniform will used gmsh Fields to have non-homogeneity
        # global scaling is controlled by resolution
        raise NotImplemented

    def get_target_mesh(self, *, structured, uniform, resolution):
        '''Mesh of the deformed configuration'''
        raise NotImplemented

    def ref2def_bdry_deformation(self, tag):
        '''
        Reference boundary with tag is mapping to Target boundary with 
        the same tag. Here we compute the function which retuns 
        the map: point of reference boundary -> point on target boundary
        '''
        # In general if t is the paremtrization of reference and we have
        # also paremetrization of target them for x_ref = x_ref(param)
        # is mapped to y_ref(param), i.e. we use the same parameter
        # value
        raise NotImplemented

    def set_mapping_bcs(self, Velm, boundaries, tags, mode):
        '''
        Mapping will be computed in Velm-functionspace (V) over reference 
        mesh (boundaries.mesh()). To set the boundary values we compute here 
        a function in V which set the boundary dofs. For mode "displacement" the 
        boundary values are set using x_target - x_ref, otherwise, the 
        values is x_target.
        '''
        # This will work only for Lagrange* like elements
        mesh = boundaries.mesh()
        gdim = mesh.geometry().dim()
        
        assert Velm.value_shape() == (gdim, )
        assert Velm.family() in ('Lagrange', 'Discontinuous Lagrange', 'Crouzeix-Raviart')

        # The idea is to use DirichletBC to find dofs on tags and based
        # on their coordinates compute
        V = df.FunctionSpace(mesh, Velm)
        X_dofs = V.tabulate_dof_coordinates().reshape((-1, gdim))

        dm = V.dofmap()
        gFirst, gLast = dm.ownership_range()
        is_local = lambda d: gFirst <= dm.local_to_global_index(d) < gLast

        if mode == 'displacement':
            mode = lambda ref, target: target - ref
        else:
            mode = lambda ref, target: target

        # What do want to build is
        f = df.Function(V)
        f_values = f.vector().get_local()
                            
        for tag in tags:
            bdry_map = self.ref2def_bdry_deformation(tag)
            # NOTE: this is a point of inefficiency as dofs of sub are
            # associated with same point
            for sub in range(Velm.num_sub_elements()):
                Vsub_dofs = df.DirichletBC(V.sub(sub),
                                           df.Constant(0),
                                           boundaries,
                                           tag).get_boundary_values().keys()
                
                for dof in filter(is_local, Vsub_dofs):
                    x = X_dofs[dof]    # Reference
                    y = bdry_map(x)    # Target

                    f_values[dof] = mode(x, y)[sub]
    
        f.vector().set_local(f_values)

        return f
