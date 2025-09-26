import numpy as np
import os
import uuid
import pyvista
from time import perf_counter
from tqdm import trange
from copy import deepcopy
from types import MappingProxyType
import itertools
from svv.tree.data.data import TreeData, TreeParameters, TreeMap
from svv.tree.branch.root import set_root
from svv.visualize.tree.show import show
from svv.tree.branch.bifurcation import add_vessel, check_tree
from svv.tree.export.export_solid import build_watertight_solid, build_merged_solid
from svv.tree.export.export_centerlines import build_centerlines
from svv.tree.utils.TreeManager import KDTreeManager, HNSWTree, USearchTree
from svv.tree.utils.c_extend import build_c_vessel_map
from collections import ChainMap


class Tree(object):
    def __init__(self):
        """
        The Tree class defines a branching tree structure
        that is used to abstract the physical representation
        of the generated vascular network.
        """
        self.data = TreeData()
        self.parameters = TreeParameters()
        self.vessel_map = TreeMap()
        #self.c_vessel_map = None
        self.physical_clearance = 0.0
        self.domain = None
        self.convex = None
        self.random_seed = None
        self.characteristic_length = None
        self.domain_clearance = None
        self.n_terminals = 0
        self.nonconvex_count = 0
        self.clamped_root = False
        self.rtree = None
        self.preallocation_step = int(4e6)
        self.preallocate = TreeData((self.preallocation_step, 31))
        self.segment_count = 0
        #self.preallocate_connectivity = (np.ones((self.preallocation_step, 3)) * -1).astype(int)
        self.preallocate_midpoints = np.zeros((self.preallocation_step, 3))
        self.times = {'vessels':[],
                      'local_optimization':[],
                      'collision':[],
                      'collision_1':[],
                      'collision_2':[],
                      'get_points':[],
                      'get_points_0':[],
                      'get_points_1':[],
                      'get_points_2':[],
                      'get_points_3':[],
                      'chunk_1':[],
                      'chunk_2':[],
                      'chunk_3':[],
                      'chunk_3_0':[],
                      'chunk_3_1':[],
                      'chunk_3_2':[],
                      'chunk_3_3':[],
                      'chunk_3_4':[],
                      'chunk_3_4_alt':[],
                      'chunk_3_5':[],
                      'chunk_3_6':[],
                      'chunk_3_7':[],
                      'chunk_4':[],
                      'chunk_4_0':[],
                      'chunk_4_1':[],
                      'chunk_4_2':[],
                      'chunk_4_3':[],
                      'chunk_5':[],
                      'all':[]}

    def set_domain(self, domain, convexity_tolerance=1e-2):
        """
        Set the domain for the tree. The domain should be an implicit
        function of the form f(X) = scalar, where X is a cartesian
        coordinate in space.
        """
        self.domain = domain
        self.characteristic_length = domain.characteristic_length
        self.convex = np.isclose(domain.convexity, 1.0, atol=convexity_tolerance)
        if self.physical_clearance > 0.0:
            patch = domain.patches[0]
            pt = (patch.points[0, :] + patch.normals[0, :] * self.physical_clearance).reshape(1, patch.points.shape[1])
            value = domain(pt)
            self.domain_clearance = abs(value).flatten()[0]
        else:
            self.domain_clearance = 0.0
        if self.domain.mesh is not None:
            self.domain.mesh.cell_data['probability'] = self.domain.mesh.cell_data['Normalized_Volume'].copy()
        return None

    def set_root(self, *args, **kwargs):
        """
        Set the root point of the tree.
        """
        if len(args) == 0:
            pass
        elif len(args) == 1:
            kwargs['start'] = args[0]
        elif len(args) == 2:
            kwargs['start'] = args[0]
            kwargs['direction'] = args[1]
            self.clamped_root = True
        else:
            raise ValueError("Too many arguments.")
        inplace = kwargs.pop('inplace', True)
        if not isinstance(kwargs.get('direction', None), type(None)):
            self.clamped_root = True
        if self.physical_clearance > 0.0:
            kwargs['interior_range'] = [-1.0, 0.0-self.domain_clearance]
        root, root_map = set_root(self, **kwargs)
        if inplace:
            self.data = root
            self.preallocate[0, :] = root
            #self.data_copy = self.preallocate[:3, :]
            self.connectivity = np.nan_to_num(root[:, 15:18], nan=-1.0).astype(int).reshape(1, 3)
            #self.preallocate_connectivity[0, :] = self.connectivity
            #self.connectivity_copy = self.connectivity.copy()
            #self.connectivity_copy = self.preallocate_connectivity[:3, :]
            self.preallocate_midpoints[0, :] = (root[:, 0:3] + root[:, 3:6]) / 2
            self.midpoints = self.preallocate_midpoints[0, :]
            #self.midpoints_copy = self.preallocate_midpoints[:1, :]
            self.vessel_map.update(root_map)
            #self.c_vessel_map = build_c_vessel_map(root_map)
            self.vessel_map_copy = deepcopy(self.vessel_map)
            self.n_terminals = 1
            self.kdtm = KDTreeManager(((root[:, 0:3] + root[:, 3:6]) / 2).reshape(1, 3))
            self.hnsw_tree = USearchTree(((root[:, 0:3] + root[:, 3:6]) / 2).reshape(1, 3).astype(np.float32))
            self.hnsw_tree_id = id(self.hnsw_tree)
            self.probability = np.array(self.domain.mesh.cell_data['probability'])
            self.max_distal_node = 1
            self.tree_scale = np.pi * root[0, 21]**self.parameters.radius_exponent*root[0, 20]**self.parameters.length_exponent
            self.segment_count = 1
            #self.rtree = RTree()
            #self.rtree.insert(((root[:, 0:3] + root[:, 3:6]) / 2).reshape(1, 3))
        else:
            connectivity = np.nan_to_num(root[:, 15:18], nan=-1.0).astype(int).reshape(1, 3)
            kdtm = KDTreeManager(((root[:, 0:3] + root[:, 3:6]) / 2).reshape(1, 3))
            hnsw_tree = USearchTree(((root[:, 0:3] + root[:, 3:6]) / 2).reshape(1, 3).astype(np.float32))
            hnsw_tree_id = id(hnsw_tree)
            probability = np.array(self.domain.mesh.cell_data['probability'])
            tree_scale = np.pi * root[0, 21] ** self.parameters.radius_exponent * root[
                0, 20] ** self.parameters.length_exponent
            return root, root_map, connectivity, kdtm, hnsw_tree, hnsw_tree_id, probability, tree_scale

    def add(self, inplace=True, **kwargs):
        all_start = perf_counter()
        decay_probability = kwargs.pop('decay_probability', 0.9)
        new_data, added_vessels, new_vessel_map, history, lines, nonconvex_outside, new_inds, mesh_cell, connectivity, change_i, change_j, new_tmp_data, old_tmp_data = add_vessel(self, **kwargs)
        start = perf_counter()
        #native_volume = np.sum(np.pi*new_data[:, 21]**2.0*new_data[:, 20])
        #assert np.isclose(native_volume,self.new_tree_scale), "native {} cost {} volumes do not match ".format(native_volume, self.new_tree_scale)
        #results = tuple([new_data, added_vessels, new_vessel_map, history, lines])
        #check = check_tree(self, results)
        #if not check[0]:
        #    print("Tree is not valid.")
        #    return check[1], check[2]
        #start = perf_counter()
        if not self.convex:
            if nonconvex_outside:
                self.nonconvex_count = 0
            else:
                self.nonconvex_count += 1
            if self.nonconvex_count > self.parameters.max_nonconvex_count:
                self.convex = True
        if inplace:
            #start_5 = perf_counter()
            #_data = TreeData(new_data.shape)
            #_data[:, :] = new_data[:, :]
            #self.data = _data
            #end_5 = perf_counter()
            #self.times['chunk_5'].append(end_5-start_5)
            start_chunk_4_0 = perf_counter()
            #new_data = np.vstack([self.data, added_vessels[0], added_vessels[1]])
            self.preallocate[self.segment_count,:] = added_vessels[0]
            self.preallocate[self.segment_count+1,:] = added_vessels[1]
            end_chunk_4_0 = perf_counter()
            self.times['chunk_4_0'].append(end_chunk_4_0 - start_chunk_4_0)
            start_chunk_4_1 = perf_counter()
            #new_data[change_i, change_j] = np.array(new_tmp_data)
            change_i = np.array(change_i, dtype=int)
            change_j = np.array(change_j, dtype=int)
            #if change_i.shape[0] > 10000:
            #    #print('parallel')
            #    parallel_scatter_update(self.preallocate, change_i, change_j, np.array(new_tmp_data))
            #else:
            self.preallocate[change_i, change_j] = np.array(new_tmp_data)
            end_chunk_4_1 = perf_counter()
            self.times['chunk_4_1'].append(end_chunk_4_1 - start_chunk_4_1)
            start_chunk_4_2 = perf_counter()
            #self.data = TreeData.from_array(new_data)
            self.data = self.preallocate[:self.segment_count+2,:]
            new_data = self.data
            end_chunk_4_2 = perf_counter()
            self.times['chunk_4_2'].append(end_chunk_4_2 - start_chunk_4_2)
            #np.copyto(self.preallocate[:new_data.shape[0], :], new_data)
            #self.data_copy = self.preallocate[:(new_data.shape[0] + 2), :]
            #self.vessel_map_copy = ChainMap(new_vessel_map, self.vessel_map)
            start_chunk_4_3 = perf_counter()
            for key in new_vessel_map.keys():
                if key in self.vessel_map.keys():
                    #print("new_vessel_map: {}".format(new_vessel_map))
                    #print("vessel_map: {}".format(self.vessel_map))
                    self.vessel_map[key]['upstream'].extend(new_vessel_map[key]['upstream'])
                    #_, counts = np.unique(self.vessel_map[key]['upstream'], return_counts=True)
                    #assert np.all(counts == 1), "Fail in appending upstream idxs.\nkey: {}\nprevious: {}\nextension: {}".format(key, self.vessel_map[key]['upstream'],
                    #                                                                                                   new_vessel_map[key]['upstream'])
                    self.vessel_map[key]['downstream'].extend(new_vessel_map[key]['downstream'])
                    #_, counts = np.unique(self.vessel_map[key]['downstream'], return_counts=True)
                    #assert np.all(counts == 1), "Fail in appending downstream idxs"
                else:
                    self.vessel_map[key] = deepcopy(new_vessel_map[key])
            end_chunk_4_3 = perf_counter()
            self.times['chunk_4_3'].append(end_chunk_4_3 - start_chunk_4_3)
            #self.vessel_map = ChainMap(new_vessel_map, self.vessel_map)
            self.n_terminals += 1
            self.segment_count += 2
            self.connectivity = connectivity #self.connectivity_copy.view()
            #self.connectivity_copy = self.preallocate_connectivity[:(self.connectivity_copy.shape[0] + 2), :]
            self.max_distal_node += 2
            if mesh_cell >= 0:
                self.probability[mesh_cell] *= decay_probability
                self.probability = self.probability / self.probability.sum()
                self.domain.cumulative_probability = np.cumsum(self.probability)
                #self.domain.mesh.cell_data['probability'][mesh_cell] *= decay_probability
                #self.domain.mesh.cell_data['probability'] = self.domain.mesh.cell_data['probability'] / \
                #                                            self.domain.mesh.cell_data['probability'].sum()
                #self.domain.cumulative_probability = np.cumsum(self.domain.mesh.cell_data['probability'])
            #self.kdtm.start_update(((new_data[:, 0:3] + new_data[:, 3:6]) / 2))
            #self.rtree.replace(new_inds[0], ((new_data[new_inds[0], 0:3] + new_data[new_inds[0], 3:6]) / 2).flatten().tolist())
            #self.rtree.rtree_index.insert(new_inds[1], ((new_data[new_inds[1], 0:3] + new_data[new_inds[1], 3:6]) / 2).flatten().tolist())
            #self.rtree.points.append(((new_data[new_inds[1], 0:3] + new_data[new_inds[1], 3:6]) / 2).flatten().tolist())
            #self.rtree.rtree_index.insert(new_inds[2], ((new_data[new_inds[2], 0:3] + new_data[new_inds[2], 3:6]) / 2).flatten().tolist())
            #self.rtree.points.append(((new_data[new_inds[2], 0:3] + new_data[new_inds[2], 3:6]) / 2).flatten().tolist())
            #self.kdtm.wait_for_update()
            #self.kdtm.swap_trees()
            #if self.n_terminals % 10000 == 0:
            #    self.hnsw_tree = USearchTree(((new_data[:, 0:3] + new_data[:, 3:6])/2).astype(np.float32))
            #else:
            self.hnsw_tree.replace(((new_data[new_inds[0], 0:3] + new_data[new_inds[0], 3:6]) / 2).reshape(1, 3).astype(np.float32), np.array([new_inds[0]]))
            self.hnsw_tree.add_items(((new_data[new_inds[1], 0:3] + new_data[new_inds[1], 3:6]) / 2).reshape(1, 3).astype(np.float32), np.array([new_inds[1]]))
            self.hnsw_tree.add_items(((new_data[new_inds[2], 0:3] + new_data[new_inds[2], 3:6]) / 2).reshape(1, 3).astype(np.float32), np.array([new_inds[2]]))
            #self.preallocate_midpoints[new_inds, :] = (new_data[new_inds, 0:3] + new_data[new_inds, 3:6])/2
            #self.midpoints_copy = self.preallocate_midpoints[:new_data.shape[0], :]
            self.preallocate_midpoints[:self.segment_count, :] = (new_data[:, 0:3] + new_data[:, 3:6])/2
            self.tree_scale = self.new_tree_scale
            end = perf_counter()
            self.times['chunk_4'].append(end - start)
            self.times['all'].append(end - all_start)
            return None
        else:
            end = perf_counter()
            self.times['chunk_4'].append(end - start)
            self.times['all'].append(end - all_start)
            #return new_data, added_vessels, new_vessel_map, history, lines, nonconvex_outside, new_inds, mesh_cell
            return change_i, change_j, new_tmp_data, old_tmp_data, new_vessel_map, connectivity, new_inds, mesh_cell, added_vessels

    def n_add(self, n, **kwargs):
        for i in trange(n, desc='Adding vessels', unit='vessel', leave=False):
            self.add(**kwargs)
        return None
    def prune(self):
        pass

    def show(self, **kwargs):
        return show(self, **kwargs)

    def save(self):
        pass

    def export_solid(self, outdir=None, shell_thickness=0.0, watertight=False, 
                    smooth_junctions=True, smoothing_radius_factor=2.0, 
                    smoothing_iterations=5, **kwargs):
        """
        Export the tree as a solid mesh with optional junction smoothing.
        
        Parameters
        ----------
        outdir : str, optional
            Output directory for the mesh files
        shell_thickness : float
            Thickness of the shell (currently unused)
        watertight : bool
            Whether to create a watertight mesh
        smooth_junctions : bool
            Whether to apply junction smoothing
        smoothing_radius_factor : float
            Factor to determine smoothing radius around junctions
        smoothing_iterations : int
            Number of smoothing iterations
        **kwargs
            Additional keyword arguments
            
        Returns
        -------
        model : pyvista.PolyData
            The exported solid mesh
        """
        if isinstance(outdir, type(None)):
            if not os.path.exists(os.getcwd() + os.sep + '3d_tmp'):
                outdir = os.getcwd() + os.sep + '3d_tmp'
            else:
                outdir = os.getcwd() + os.sep + '3d_tmp_' + str(uuid.uuid4())
        else:
            if os.path.exists(outdir):
                raise ValueError("Output directory already exists.")
        if not watertight:
            model = build_merged_solid(self)
        else:
            model = build_watertight_solid(self, smooth_junctions=smooth_junctions,
                                         smoothing_radius_factor=smoothing_radius_factor,
                                         smoothing_iterations=smoothing_iterations)
        return model


    def export_centerlines(self, outdir=None, **kwargs):
        centerlines, polys = build_centerlines(self)
        return centerlines, polys


    def export_gcode(self):
        pass

    def export_centerline(self):
        pass

    def get_junction_statistics(self):
        """
        Get statistics about junctions in the tree.
        
        Returns
        -------
        stats : dict
            Dictionary containing junction statistics
        """
        from svv.tree.utils.junction_smoothing import get_junction_statistics
        return get_junction_statistics(self.data, self.vessel_map, self.connectivity)

