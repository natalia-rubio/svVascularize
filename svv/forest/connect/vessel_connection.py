import numpy
import pyvista as pv
from copy import deepcopy
from svv.forest.connect.base_connection import BaseConnection


class VesselConnection:
    def __init__(self, forest, network_id, tree_0, tree_1, idx, jdx,
                 ctrl_function=None, clamp_first=True, clamp_second=True,
                 point_0=None, point_1=None, curve_type="Bezier"):
        self.forest = forest
        self.tree_0 = tree_0
        self.tree_1 = tree_1
        self.idx = idx
        self.jdx = jdx
        vessel_0 = forest.networks[network_id][tree_0].data[idx, :]
        vessel_1 = forest.networks[network_id][tree_1].data[jdx, :]
        self.proximal_0 = vessel_0[0:3]
        self.distal_0 = vessel_0[3:6]
        self.proximal_1 = vessel_1[0:3]
        self.distal_1 = vessel_1[3:6]
        self.radius_0 = vessel_0[21]
        self.radius_1 = vessel_1[21]
        collision_vessels = []
        min_distance = max(self.radius_0, self.radius_1)*0.5
        #min_distance = (max(numpy.max(forest.networks[network_id][tree_0].data[:,21]),
        #                   numpy.max(forest.networks[network_id][tree_1].data[:,21])) + max(self.radius_0,self.radius_1))
        conn = BaseConnection(vessel_0[0:3], vessel_0[3:6], vessel_1[0:3], vessel_1[3:6], vessel_0[21], vessel_1[21],
                              domain=forest.domain, ctrlpt_function=ctrl_function, clamp_first=clamp_first,
                              clamp_second=clamp_second, point_0=point_0, point_1=point_1, min_distance=min_distance,
                              curve_type=curve_type)
        tree_0_idx = numpy.arange(forest.networks[network_id][tree_0].data.shape[0], dtype=int).tolist()
        if not numpy.isnan(vessel_0[17]):
            parent = int(vessel_0[17])
            daughter_0 = int(forest.networks[network_id][tree_0].data[parent, 15])
            daughter_1 = int(forest.networks[network_id][tree_0].data[parent, 16])
            #tree_0_idx.remove(parent)
            #if daughter_0 == idx:
            #    tree_0_idx.remove(daughter_0)
            #else:
            #    tree_0_idx.remove(daughter_1)
            tree_0_idx.remove(idx)
        if not numpy.isnan(vessel_0[15]):
            daughter_2 = int(vessel_1[15])
            tree_0_idx.remove(daughter_2)
        if not numpy.isnan(vessel_0[16]):
            daughter_3 = int(vessel_1[16])
            tree_0_idx.remove(daughter_3)
        tree_0_idx = numpy.array(tree_0_idx).astype(int)
        tmp = numpy.zeros((tree_0_idx.shape[0], 7))
        tmp[:, 0:3] = forest.networks[network_id][tree_0].data[tree_0_idx, 0:3]
        tmp[:, 3:6] = forest.networks[network_id][tree_0].data[tree_0_idx, 3:6]
        tmp[:, 6] = forest.networks[network_id][tree_0].data[tree_0_idx, 21]
        collision_vessels.append(deepcopy(tmp))
        tree_1_idx = numpy.arange(forest.networks[network_id][tree_1].data.shape[0], dtype=int).tolist()
        if not numpy.isnan(vessel_1[17]):
            parent = int(vessel_1[17])
            daughter_0 = int(forest.networks[network_id][tree_1].data[parent, 15])
            daughter_1 = int(forest.networks[network_id][tree_1].data[parent, 16])
            #tree_1_idx.remove(parent)
            #if daughter_0 == jdx:
            #    tree_1_idx.remove(daughter_0)
            #else:
            #    tree_1_idx.remove(daughter_1)
            #tree_1_idx.remove(daughter_0)
            #tree_1_idx.remove(daughter_1)
            tree_1_idx.remove(jdx)
        if not numpy.isnan(vessel_1[15]):
            daughter_2 = int(vessel_1[15])
            tree_1_idx.remove(daughter_2)
        if not numpy.isnan(vessel_1[16]):
            daughter_3 = int(vessel_1[16])
            tree_1_idx.remove(daughter_3)
        tree_1_idx = numpy.array(tree_1_idx).astype(int)
        tmp = numpy.zeros((tree_1_idx.shape[0], 7))
        tmp[:, 0:3] = forest.networks[network_id][tree_1].data[tree_1_idx, 0:3]
        tmp[:, 3:6] = forest.networks[network_id][tree_1].data[tree_1_idx, 3:6]
        tmp[:, 6] = forest.networks[network_id][tree_1].data[tree_1_idx, 21]
        collision_vessels.append(deepcopy(tmp))
        for i in range(forest.n_networks):
            for j in range(forest.n_trees_per_network[i]):
                if i == network_id and (j == tree_0 or j == tree_1):
                    continue
                tmp = numpy.zeros((forest.networks[i][j].data.shape[0], 7))
                tmp[:, 0:3] = forest.networks[i][j].data[:, 0:3]
                tmp[:, 3:6] = forest.networks[i][j].data[:, 3:6]
                tmp[:, 6] = forest.networks[i][j].data[:, 21]
                collision_vessels.append(deepcopy(tmp))
        collision_vessels = numpy.vstack(collision_vessels)
        if collision_vessels.shape[0] > 0:
            conn.set_collision_vessels(collision_vessels)
        conn.set_physical_clearance(self.forest.networks[network_id][tree_0].domain_clearance)
        bounds = numpy.zeros((3, 2))
        bounds[:, 0] = numpy.min(forest.domain.points, axis=0).T
        bounds[:, 1] = numpy.max(forest.domain.points, axis=0).T
        conn.clamp_first = clamp_first
        conn.clamp_second = clamp_second
        self.connection = conn
        self.curve = None
        self.result = None
        self.vessels_1 = None
        self.vessels_2 = None
        self.vessel_1_meshes = None
        self.vessel_2_meshes = None
        self.seperate = None
        self.plotting_vessels = None

    def solve(self, *args, **kwargs):
        result, curve, objective, constraints = self.connection.solve(*args, **kwargs)
        self.result = result
        self.curve = curve
        self.objective = objective
        self.constraints = constraints

    def build_vessels(self, num, seperate=True, build_meshes=True):
        t = numpy.linspace(0, 1, num)
        pts = self.curve.evaluate(t)
        if seperate:
            sep = (pts.shape[0]-1)//2
            vessels_1 = numpy.zeros((sep,7))
            vessels_2 = numpy.zeros((pts.shape[0]-1-sep,7))
            vessels_1_pts = pts[:sep+1,:]
            vessels_2_pts = numpy.flip(pts[sep:,:],axis=0)
            vessels_1[:, 0:3] = vessels_1_pts[:-1,:]
            vessels_1[:, 3:6] = vessels_1_pts[1:,:]
            vessels_2[:, 0:3] = vessels_2_pts[:-1,:]
            vessels_2[:, 3:6] = vessels_2_pts[1:,:]
            vessels_1[:, 6] = self.radius_0
            vessels_2[:, 6] = self.radius_1
        else:
            vessels_1 = numpy.zeros((pts.shape[0]-1,7))
            vessels_1[:,0:3] = pts[:-1,:]
            vessels_1[:,3:6] = pts[1:,:]
            vessels_1[:,6] = self.radius_0
            pts_reverse = numpy.flip(pts,axis=0)
            vessels_2 = numpy.zeros((pts.shape[0]-1,7))
            vessels_2[:,0:3] = pts_reverse[:-1,:]
            vessels_2[:,3:6] = pts_reverse[1:,:]
            vessels_2[:,6] = self.radius_1
        self.vessels_1 = vessels_1
        self.vessels_2 = vessels_2
        self.seperate = seperate
        self.vessel_1_meshes = []
        if build_meshes:
            for i in range(vessels_1.shape[0]):
                center = (self.vessels_1[i, 0:3] + self.vessels_1[i, 3:6]) / 2
                direction = self.vessels_1[i, 3:6] - self.vessels_1[i, 0:3]
                length = numpy.linalg.norm(direction)
                direction = direction / length
                radius = self.vessels_1[i, 6]
                cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                self.vessel_1_meshes.append(cylinder)
            self.vessel_2_meshes = []
            for i in range(self.vessels_2.shape[0]):
                center = (self.vessels_2[i, 0:3] + self.vessels_2[i, 3:6]) / 2
                direction = self.vessels_2[i, 3:6] - self.vessels_2[i, 0:3]
                length = numpy.linalg.norm(direction)
                direction = direction / length
                radius = self.vessels_2[i, 6]
                cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                self.vessel_2_meshes.append(cylinder)
        return vessels_1, vessels_2

    def show(self):
        plotter = pv.Plotter()
        if self.seperate:
            for i in range(self.vessels_1.shape[0]):
                center = (self.vessels_1[i,0:3]+self.vessels_1[i,3:6])/2
                direction = self.vessels_1[i,3:6] - self.vessels_1[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_1[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='red')
            for i in range(self.vessels_2.shape[0]):
                center = (self.vessels_2[i,0:3]+self.vessels_2[i,3:6])/2
                direction = self.vessels_2[i,3:6] - self.vessels_2[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_2[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='blue')
        else:
            for i in range(self.vessels_1.shape[0]):
                center = (self.vessels_1[i,0:3]+self.vessels_1[i,3:6])/2
                direction = self.vessels_1[i,3:6] - self.vessels_1[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_1[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='red')
        if self.plotting_vessels is not None:
            for i in range(self.plotting_vessels.shape[0]):
                center = (self.plotting_vessels[i,0:3]+self.plotting_vessels[i,3:6])/2
                direction = self.plotting_vessels[i,3:6] - self.plotting_vessels[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.plotting_vessels[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='black')
        if numpy.all(self.proximal_0 != self.distal_0):
            center = (self.proximal_0+self.distal_0)/2
            direction = (self.distal_0-self.proximal_0)
            length = numpy.linalg.norm(direction)
            direction = direction/length
            radius = self.radius_0
            cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
            plotter.add_mesh(cylinder,color='yellow')
        if numpy.all(self.proximal_1 != self.distal_1):
            center = (self.proximal_1+self.distal_1)/2
            direction = (self.distal_1-self.proximal_1)
            length = numpy.linalg.norm(direction)
            direction = direction/length
            radius = self.radius_1
            cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
            plotter.add_mesh(cylinder,color='yellow')
        return plotter
