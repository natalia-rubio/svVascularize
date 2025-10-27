import numpy
from copy import deepcopy
from svv.tree.tree import Tree
from svv.tree.data.data import TreeData
from svv.tree.collision.tree_collision import tree_collision
from svv.forest.connect.geodesic import geodesic_constructor
from svv.forest.connect.forest_connection import ForestConnection
from svv.visualize.forest.show import show


class Forest(object):
    def __init__(self, **kwargs):
        """
        The Forest class defines collections of synthetic vascular tree
        structures that are used to abstract the physical representations
        of generated interpenetrating vascular networks.

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments to be passed.
            Keyword arguments include:
                domain : svtoolkit.domain.Domain
                    A domain object that defines the spatial region in which vascular
                    trees are generated.
                n_networks : int
                    The number of vascular networks to be generated.
                n_trees_per_network : list of int
                    The number of trees to be generated per network.
                start_points : list of numpy.ndarray
                    A list of starting points for the trees.
                directions : list of numpy.ndarray
                    A list of directions for the trees.
                physical_clearance : float
                    The physical clearance of any vessel from the domain and other vessels.
                compete : bool
                    A flag that indicates whether trees should compete for space.
        """
        self.networks = []
        self.connections = None
        self.domain = kwargs.get('domain', None)
        self.n_networks = kwargs.get('n_networks', 1)
        self.n_trees_per_network = kwargs.get('n_trees_per_network', [2])
        self.start_points = kwargs.get('start_points', None)
        self.directions = kwargs.get('directions', None)
        self.physical_clearance = kwargs.get('physical_clearance', 0.0)
        self.compete = kwargs.get('compete', False)
        self.geodesic = None
        self.convex = None
        if isinstance(self.start_points, type(None)):
            self.start_points = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
        if isinstance(self.directions, type(None)):
            self.directions = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
        networks = []
        for i in range(self.n_networks):
            network = []
            for j in range(self.n_trees_per_network[i]):
                tree = Tree()
                tree.physical_clearance = self.physical_clearance
                network.append(tree)
            networks.append(network)
        self.networks = networks

    def set_domain(self, domain, convexity_tolerance=1e-2):
        """
        Set the domain for the forest. The domain should be an implicit
        function of the form f(X) = scalar, where X is a cartesian
        coordinate in space.

        Parameters
        ----------
        domain : svtoolkit.domain.Domain
            A domain object that defines the spatial region in which vascular
            trees are generated.
        convexity_tolerance : float
            The tolerance for the convexity of the domain.
        """
        self.domain = domain
        self.geodesic = geodesic_constructor(domain)
        self.convex = numpy.isclose(domain.convexity, 1.0, atol=convexity_tolerance)
        for network in self.networks:
            for tree in network:
                tree.set_domain(domain, convexity_tolerance)
        return None

    def set_roots(self, *args, **kwargs):
        """
        Set the root point of the forest.

        Parameters
        ----------
        start_points : list of numpy.ndarray
            A list of starting points for the trees.
        directions : list of numpy.ndarray
            A list of directions for the trees.
        kwargs : dict
        """
        if len(args) == 0:
            kwargs['start_points'] = self.start_points
            kwargs['directions'] = self.directions
        elif len(args) == 1:
            kwargs['start_points'] = args[0]
            kwargs['directions'] = self.directions
        elif len(args) == 2:
            kwargs['start_points'] = args[0]
            kwargs['directions'] = args[1]
        else:
            raise ValueError("Too many arguments.")
        tmp_roots = []
        tmp_root_maps = []
        tmp_connectivities = []
        tmp_kdtms = []
        tmp_hnsw_trees = []
        tmp_hnsw_tree_ids = []
        tmp_probabilities = []
        tmp_tree_scales = []
        for i in range(self.n_networks):
            for j in range(self.n_trees_per_network[i]):
                success = False
                while not success:
                    tmp_root, tmp_root_map,tmp_connecivity,tmp_kdtm,tmp_hnsw_tree,tmp_hnsw_tree_id,tmp_probability,tmp_tree_scale = self.networks[i][j].set_root(start=kwargs['start_points'][i][j],
                                                                          direction=kwargs['directions'][i][j],
                                                                          inplace=False)
                    collisions = []
                    for k in range(len(tmp_roots)):
                        root_collision = tree_collision(tmp_roots[k], tmp_root, clearance=self.physical_clearance)
                        collisions.append(root_collision)
                        if root_collision:
                            break
                    if not any(collisions):
                        success = True
                tmp_roots.append(tmp_root)
                tmp_root_maps.append(tmp_root_map)
                tmp_connectivities.append(tmp_connecivity)
                tmp_kdtms.append(tmp_kdtm)
                tmp_hnsw_trees.append(tmp_hnsw_tree)
                tmp_hnsw_tree_ids.append(tmp_hnsw_tree_id)
                tmp_probabilities.append(tmp_probability)
                tmp_tree_scales.append(tmp_tree_scale)
        for i in range(self.n_networks):
            for j in range(self.n_trees_per_network[i]):
                self.networks[i][j].preallocate[0, :] = tmp_roots.pop(0)
                self.networks[i][j].data = self.networks[i][j].preallocate[:1, :]
                self.networks[i][j].vessel_map.update(tmp_root_maps.pop(0))
                self.networks[i][j].n_terminals = 1
                self.networks[i][j].max_distal_node = 1
                self.networks[i][j].connectivity = tmp_connectivities.pop(0)
                #self.networks[i][j].kdtm = tmp_kdtms.pop(0)
                self.networks[i][j].hnsw_tree = tmp_hnsw_trees.pop(0)
                self.networks[i][j].hnsw_tree_id = tmp_hnsw_tree_ids.pop(0)
                self.networks[i][j].probability = tmp_probabilities.pop(0)
                self.networks[i][j].tree_scale = tmp_tree_scales.pop(0)
                self.networks[i][j].midpoints = (self.networks[i][j].data[:, 0:3] + self.networks[i][j].data[:, 3:6])/2
                self.networks[i][j].segment_count = 1

    def add(self, *args, **kwargs):
        """
        Add a tree to the forest.
        """
        decay_probability = kwargs.pop('decay_probability', 0.9)
        self.connections = None
        if len(args) == 0:
            n_vessels = 1
        else:
            n_vessels = args[0]
        progress = kwargs.get('progress', False)
        network_id = kwargs.get('network_id', None)
        if isinstance(network_id, type(None)):
            network_id = list(range(self.n_networks))
        for k in range(n_vessels):
            new_vessels = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            added_vessels = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_vessel_maps = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_connectivity = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_inds = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_mesh_cell = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            for i in network_id:
                for j in range(self.n_trees_per_network[i]):
                    success = False
                    while not success:
                        #tmp_new_vessels, tmp_added_vessels, tmp_new_vessel_map, _, _, tmp_nonconvex_count = self.networks[i][j].add(inplace=False)
                        results = self.networks[i][j].add(inplace=False)
                        change_i, change_j, new_tmp_data, old_tmp_data, tmp_new_vessel_map, connectivity, inds, mesh_cell, tmp_added_vessels = results
                        #tmp_new_vessels = numpy.vstack([self.networks[i][j].data, tmp_added_vessels[0], tmp_added_vessels[1]])
                        change_i = numpy.array(change_i, dtype=int)
                        change_j = numpy.array(change_j, dtype=int)
                        self.networks[i][j].preallocate[self.networks[i][j].segment_count, :] = tmp_added_vessels[0]
                        self.networks[i][j].preallocate[self.networks[i][j].segment_count+1, :] = tmp_added_vessels[1]
                        self.networks[i][j].preallocate[change_i, change_j] = numpy.array(new_tmp_data)
                        self.networks[i][j].data = self.networks[i][j].preallocate[:self.networks[i][j].segment_count+2, :]
                        self.networks[i][j].preallocate_midpoints[:self.networks[i][j].segment_count+2, :] = (self.networks[i][j].data[:, 0:3] +
                                                                                                              self.networks[i][j].data[:, 3:6])/2
                        self.networks[i][j].midpoints = self.networks[i][j].preallocate_midpoints[:self.networks[i][j].segment_count + 2, :]
                        #tmp_new_vessels[change_i, change_j] = numpy.array(new_tmp_data)
                        tmp_new_vessels = self.networks[i][j].data
                        collisions = []
                        for h in range(self.n_networks):
                            for w in range(self.n_trees_per_network[h]):
                                if h == i and w == j:
                                    continue
                                for q in range(len(tmp_added_vessels)):
                                    if isinstance(new_vessels[h][w], type(None)):
                                        collision = tree_collision(self.networks[h][w].data, tmp_added_vessels[q],
                                                                   clearance=self.physical_clearance)
                                    else:
                                        collision = tree_collision(new_vessels[h][w], tmp_added_vessels[q],
                                                                   clearance=self.physical_clearance)
                                    collisions.append(collision)
                                    if collision:
                                        print("Collision detected {}-{}.".format(h, q))
                                        break
                                if any(collisions):
                                    break
                            if any(collisions):
                                break
                        if not any(collisions):
                            success = True
                            new_vessels[i][j] = tmp_new_vessels
                            added_vessels[i][j] = tmp_added_vessels
                            new_vessel_maps[i][j] = tmp_new_vessel_map
                            new_connectivity[i][j] = connectivity
                            new_inds[i][j] = inds
                            new_mesh_cell[i][j] = mesh_cell
                        else:
                            success = False
                            self.networks[i][j].preallocate[self.networks[i][j].segment_count, :] = numpy.nan
                            self.networks[i][j].preallocate[self.networks[i][j].segment_count + 1, :] = numpy.nan
                            self.networks[i][j].preallocate[change_i, change_j] = numpy.array(old_tmp_data)
                            self.networks[i][j].data = self.networks[i][j].preallocate[:self.networks[i][j].segment_count + 2,:]
                            self.networks[i][j].preallocate_midpoints[:self.networks[i][j].segment_count + 2, :] = (self.networks[i][j].data[:,0:3] +
                                                                                                                    self.networks[i][j].data[:,3:6]) / 2
                            self.networks[i][j].midpoints = self.networks[i][j].preallocate_midpoints[
                                                            :self.networks[i][j].segment_count, :]
            for i in network_id:
                for j in range(self.n_trees_per_network[i]):
                    #_data = TreeData(new_vessels[i][j].shape)
                    #_data[:, :] = new_vessels[i][j][:, :]
                    #self.networks[i][j].data = TreeData.from_array(new_vessels[i][j])
                    #self.networks[i][j].vessel_map.update(new_vessel_maps[i][j])
                    for key in new_vessel_maps[i][j].keys():
                        if key in self.networks[i][j].vessel_map.keys():
                            self.networks[i][j].vessel_map[key]['upstream'].extend(new_vessel_maps[i][j][key]['upstream'])
                            self.networks[i][j].vessel_map[key]['downstream'].extend(new_vessel_maps[i][j][key]['downstream'])
                        else:
                            self.networks[i][j].vessel_map[key] = deepcopy(new_vessel_maps[i][j][key])
                    self.networks[i][j].connectivity = new_connectivity[i][j]
                    self.networks[i][j].max_distal_node += 2
                    if new_mesh_cell[i][j] >= 0:
                        self.networks[i][j].probability[mesh_cell] *= decay_probability
                        self.networks[i][j].probability = self.networks[i][j].probability / self.networks[i][j].probability.sum()
                        self.networks[i][j].domain.cumulative_probability = numpy.cumsum(self.networks[i][j].probability)
                    self.networks[i][j].hnsw_tree.replace(
                        ((new_vessels[i][j][new_inds[i][j][0], 0:3] + new_vessels[i][j][new_inds[i][j][0], 3:6]) / 2).reshape(1, 3).astype(
                            numpy.float32), numpy.array([new_inds[i][j][0]]))
                    self.networks[i][j].hnsw_tree.add_items(
                        ((new_vessels[i][j][new_inds[i][j][1], 0:3] + new_vessels[i][j][new_inds[i][j][1], 3:6]) / 2).reshape(1, 3).astype(
                            numpy.float32), numpy.array([new_inds[i][j][1]]))
                    self.networks[i][j].hnsw_tree.add_items(
                        ((new_vessels[i][j][new_inds[i][j][2], 0:3] + new_vessels[i][j][new_inds[i][j][2], 3:6]) / 2).reshape(1, 3).astype(
                            numpy.float32), numpy.array([new_inds[i][j][2]]))
                    self.networks[i][j].tree_scale = self.networks[i][j].new_tree_scale
                    self.networks[i][j].n_terminals += 1
                    self.networks[i][j].segment_count += 2

    def connect(self, *args, **kwargs):
        self.connections = ForestConnection(self)
        if len(args) == 0:
            self.connections.solve(1, **kwargs)
        elif len(args) == 1 and isinstance(args[0], int):
            self.connections.solve(*args, **kwargs)
        else:
            raise ValueError("Invalid arguments. Optional integer argument for default Bezier polynomial degree.")

    def show(self, **kwargs):
        return show(self, **kwargs)

    def export_solid(self, outdir=None, shell_thickness=0.0):
        """
        Export the domain as a solid object.
        """
        if isinstance(outdir, type(None)):
            outdir = '3d_tmp'
        for i in range(self.n_networks):
            for j in range(self.n_trees_per_network[i]):
                self.networks[i][j].export_solid(outdir=outdir, shell_thickness=shell_thickness)