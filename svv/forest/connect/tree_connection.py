import numpy as np
import pyvista as pv
from scipy.interpolate import splprep, splev
from tqdm import trange
from copy import deepcopy
import pymeshfix

from svv.tree.tree import Tree
from svv.forest.connect.vessel_connection import VesselConnection
from svv.forest.connect.assign import assign_network, assign_network_vector
from svv.tree.branch.bifurcation import naive, naive_radius, naive_radius_scaling
from svv.tree.export.export_solid import (get_branches, get_points, get_radii, get_normals,
                                                polyline_from_points, generate_polylines, generate_tube,
                                                generate_tubes, union_tubes)
from svv.forest.export.export_solid import smooth_junctions


class TreeConnection:
    def __init__(self, forest, network_id, **kwargs):
        self.assignments, self.ctrlpts_functions = assign_network(forest, network_id, **kwargs)
        self.forest = forest
        self.network_id = network_id
        self.connections = []
        self.vessels = []
        self.lengths = []
        self.other_vessels = []
        self.meshes = []
        self.plotting_vessels = None
        self.network = []
        self.curve_type = kwargs.get("curve_type", "Bezier")
        for i in range(forest.n_trees_per_network[network_id]):
            tree_object = Tree()
            tree_object.parameters = forest.networks[network_id][i].parameters
            tree_object.data = forest.networks[network_id][i].data
            tree_object.domain = forest.networks[network_id][i].domain
            self.network.append(tree_object)
        self.connected_network = deepcopy(self.network)

    def solve(self, *args, num_vessels=20, attempts=5):
        tree_0 = 0
        tree_1 = 1
        tree_connections = []
        midpoints = []
        self.vessels = []
        self.vessels.append([])
        self.vessels.append([])
        self.lengths.append([])
        self.lengths.append([])
        self.connected_network = deepcopy(self.network)
        for j in trange(len(self.ctrlpts_functions[0]), desc=f"Tree {tree_0} to Tree {tree_1}", leave=False):
            conn = VesselConnection(self.forest, self.network_id, tree_0, tree_1,
                                    self.assignments[tree_0][j], self.assignments[tree_1][j],
                                    ctrl_function=self.ctrlpts_functions[0][j],
                                    clamp_first=True, clamp_second=True, curve_type=self.curve_type)
            collisions = []
            collisions.append(conn.connection.other_line_segments)
            if len(self.vessels) > 0:
                for i in range(len(self.vessels)):
                    if len(self.vessels[i]) > 0:
                        collisions.extend(self.vessels[i])
            if len(self.other_vessels) > 0:
                for i in range(len(self.other_vessels)):
                    if len(self.other_vessels[i]) > 0:
                        collisions.extend(self.other_vessels[i])
            if len(collisions) > 0:
                collisions = np.vstack(collisions)
                conn.connection.set_collision_vessels(collisions)
            index_0 = self.assignments[tree_0][j]
            index_1 = self.assignments[tree_1][j]
            degree = args[0]
            for i in trange(attempts, desc=f"Curve Order: {degree}, Connect {index_0} <--> {index_1}",
                            leave=False):
                if i > 0:
                    degree += 1
                conn.solve(degree)
                if conn.result.success:
                    break
            conn.build_vessels(num_vessels, seperate=True, build_meshes=False)
            self.vessels[tree_0].append(conn.vessels_1)
            lengths_0 = np.sum(np.linalg.norm(conn.vessels_1[:, 3:6] - conn.vessels_1[:, 0:3], axis=1))
            lengths_0 += np.linalg.norm(conn.vessels_1[0, 0:3] - self.network[tree_0].data[self.assignments[tree_0][j], 0:3])
            self.connected_network[tree_0].data[self.assignments[tree_0][j], 3:6] = conn.vessels_1[0, 0:3]
            self.connected_network[tree_0].data[self.assignments[tree_0][j], 20] = np.linalg.norm(self.connected_network[tree_0].data[self.assignments[tree_0][j], 3:6] -
                                                                                                  self.connected_network[tree_0].data[self.assignments[tree_0][j], 0:3])
            self.lengths[tree_0].append(lengths_0)
            self.vessels[tree_1].append(conn.vessels_2)
            lengths_1 = np.sum(np.linalg.norm(conn.vessels_2[:, 3:6] - conn.vessels_2[:, 0:3], axis=1))
            lengths_1 += np.linalg.norm(conn.vessels_2[0, 0:3] - self.network[tree_1].data[self.assignments[tree_1][j], 0:3])
            self.connected_network[tree_1].data[self.assignments[tree_1][j], 3:6] = conn.vessels_2[0, 0:3]
            self.connected_network[tree_1].data[self.assignments[tree_1][j], 20] = np.linalg.norm(self.connected_network[tree_1].data[self.assignments[tree_1][j], 3:6] -
                                                                                                  self.connected_network[tree_1].data[self.assignments[tree_1][j], 0:3])
            self.lengths[tree_1].append(lengths_1)
            tree_connections.append(conn)
            midpoints.append(conn.vessels_1[-1, 3:6])
        midpoints = np.array(midpoints)
        self.connections.append(tree_connections)
        if self.forest.n_trees_per_network[self.network_id] > 2:
            remaining_assignments, remaining_connections = assign_network_vector(self.forest, self.network_id,
                                                                                 midpoints)
            self.assignments.extend(remaining_assignments)
            self.ctrlpts_functions.extend(remaining_connections)
            for n in range(2, self.forest.n_trees_per_network[self.network_id]):
                tree_0 = 0
                tree_n = n
                tree_connections = []
                self.vessels.append([])
                self.lengths.append([])
                for j in trange(len(remaining_connections[n - 2]), desc=f"Tree {tree_0} to Tree {tree_n} ", leave=False):
                    conn = VesselConnection(self.forest, self.network_id, tree_n, tree_0,
                                            remaining_assignments[n - 2][j], self.assignments[tree_0][j],
                                            ctrl_function=remaining_connections[n - 2][j],
                                            clamp_first=True, clamp_second=False, point_1=midpoints[j],
                                            curve_type=self.curve_type)
                    collisions = []
                    for i in range(len(self.vessels)):
                        for c in range(len(self.vessels[i])):
                            if c == j:
                                collisions.append(self.vessels[i][c][:-1, :])
                            else:
                                collisions.append(self.vessels[i][c])
                    if len(self.other_vessels) > 0:
                        for i in range(len(self.other_vessels)):
                            if len(self.other_vessels[i]) > 0:
                                collisions.extend(self.other_vessels[i])
                    collisions.append(conn.connection.other_line_segments)
                    collisions = np.vstack(collisions)
                    conn.connection.set_collision_vessels(collisions)
                    index_0 = remaining_assignments[n - 2][j]
                    index_1 = self.assignments[tree_0][j]
                    degree = args[0]
                    for i in trange(attempts, desc=f"Curve Order: {degree}, Connect {index_0} <--> {index_1}", leave=False):
                        if i > 0:
                            degree += 1
                        conn.solve(degree)
                        if conn.result.success:
                            break
                    conn.build_vessels(num_vessels, seperate=False, build_meshes=False)
                    self.vessels[tree_n].append(conn.vessels_1)
                    lengths_n = np.sum(np.linalg.norm(conn.vessels_1[:, 3:6] - conn.vessels_1[:, 0:3], axis=1))
                    lengths_n += np.linalg.norm(conn.vessels_1[0, 0:3] - self.network[tree_n].data[remaining_assignments[n - 2][j], 0:3])
                    self.connected_network[tree_n].data[remaining_assignments[n - 2][j], 3:6] = conn.vessels_1[0, 0:3]
                    self.connected_network[tree_n].data[remaining_assignments[n - 2][j], 20] = np.linalg.norm(self.connected_network[tree_n].data[remaining_assignments[n - 2][j], 3:6] -
                                                                                                              self.connected_network[tree_n].data[remaining_assignments[n - 2][j], 0:3])
                    self.lengths[tree_n].append(lengths_n)
                    tree_connections.append(conn)
                self.connections.append(tree_connections)
        # Adjust the radii of the trees to ensure pressure uniformity
        for i in range(len(self.connected_network)):
            for j in range(len(self.assignments[i])):
                self.connected_network[i].data[self.assignments[i][j], 20] = self.lengths[i][j]
        for i in range(len(self.network)):
            naive(self.connected_network[i], self.connected_network[i].data, 0)
            root_pressure = self.connected_network[i].parameters.root_pressure
            terminal_pressure = self.connected_network[i].parameters.terminal_pressure
            root_radius = ((self.connected_network[i].data[0, 22]*self.connected_network[i].data[0, 25])/(root_pressure-terminal_pressure)) ** (1.0/4.0)
            self.connected_network[i].data[0, 21] = root_radius
            naive_radius(self.connected_network[i].data)
            naive_radius_scaling(self.connected_network[i].data)
        # Propagate the radii to the connecting vessels
        for i in range(len(self.connected_network)):
            for j in range(len(self.assignments[i])):
                self.connected_network[i].data[self.assignments[i][j], 20] = np.linalg.norm(self.connected_network[i].data[self.assignments[i][j], 3:6] -
                                                                                            self.connected_network[i].data[self.assignments[i][j], 0:3])
                self.vessels[i][j][:, 6] = self.connected_network[i].data[self.assignments[i][j], 21]

    def show(self, **kwargs):
        colors = kwargs.get('colors', ['red', 'blue', 'green', 'yellow', 'purple',
                                       'orange', 'cyan', 'magenta', 'white', 'black'])
        plotter = pv.Plotter(**kwargs)
        count = 0
        for i in range(len(self.connected_network)):
            for j in range(self.connected_network[i].data.shape[0]):
                if j in self.assignments[i]:
                    conn_id = self.assignments[i].index(j)
                    center = (self.connected_network[i].data[j, 0:3] + self.vessels[i][conn_id][0, 0:3]) / 2
                    length = np.linalg.norm(self.vessels[i][conn_id][0, 0:3] - self.connected_network[i].data[j, 0:3])
                    direction = (self.vessels[i][conn_id][0, 0:3] - self.connected_network[i].data[j, 0:3]) / length
                    radius = self.connected_network[i].data[j, 21]
                    vessel = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                else:
                    center = (self.connected_network[i].data[j, 0:3] + self.connected_network[i].data[j, 3:6]) / 2
                    direction = self.connected_network[i].data.get('w_basis', j)
                    radius = self.connected_network[i].data.get('radius', j)
                    length = np.linalg.norm(self.connected_network[i].data[j, 3:6] - self.connected_network[i].data[j, 0:3])
                    vessel = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                plotter.add_mesh(vessel, color=colors[count % len(colors)], opacity=0.25)
            count += 1
        count = 0
        for conn in range(len(self.vessels)):
            for i in range(len(self.vessels[conn])):
                for j in range(self.vessels[conn][i].shape[0]):
                    center = (self.vessels[conn][i][j, 0:3] + self.vessels[conn][i][j, 3:6]) / 2
                    direction = self.vessels[conn][i][j, 3:6] - self.vessels[conn][i][j, 0:3]
                    length = np.linalg.norm(direction)
                    direction = direction / length
                    radius = self.vessels[conn][i][j, 6]
                    cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                    plotter.add_mesh(cylinder, color=colors[count % len(colors)])
            count += 1
        plotter.add_mesh(self.forest.domain.boundary, color='grey', opacity=0.15)
        return plotter

    def export_solid(self, cap_resolution=10, extrude_roots = False):
        network_branches = []
        network_points = []
        network_radii = []
        network_normals = []
        network_terminals = []
        if extrude_roots:
            for i in range(len(self.connected_network)):
                direction = (self.connected_network[i].data[0, 3:6] - self.connected_network[i].data[0, 0:3])
                direction = direction / np.linalg.norm(direction)
                root_extension = self.connected_network[i].data[0, 21] * 4
                start = self.connected_network[i].data[0, 0:3].copy()
                for j in range(10):
                    new_start = start - direction * root_extension * (j + 1)
                    if self.forest.domain(new_start.reshape(1, 3)).flatten() > 0:
                        self.connected_network[i].data[0, 0:3] = new_start
                        break
        for i in range(len(self.connected_network)):
            branches = get_branches(self.connected_network[i].data)
            network_branches.append(branches)
            network_points.append(get_points(self.connected_network[i].data, branches))
            network_radii.append(get_radii(self.connected_network[i].data, branches))
            network_normals.append(get_normals(self.connected_network[i].data, branches))
            terminals = []
            for branch in branches:
                terminals.append(branch[-1])
            network_terminals.append(np.array(terminals))
        # Match the branch terminals to the connection assignments
        reordered_branches = []
        reordered_points = []
        reordered_radii = []
        reordered_normals = []
        for i in range(len(self.connected_network)):
            tmp_reordered_branches = []
            tmp_reordered_points = []
            tmp_reordered_radii = []
            tmp_reordered_normals = []
            for a in range(len(self.assignments[i])):
                ind = np.argwhere(network_terminals[i] == self.assignments[i][a]).flatten()[0]
                tmp_reordered_branches.append(network_branches[i][ind])
                tmp_reordered_points.append(network_points[i][ind])
                tmp_reordered_radii.append(network_radii[i][ind])
                tmp_reordered_normals.append(network_normals[i][ind])
            reordered_branches.append(tmp_reordered_branches)
            reordered_points.append(tmp_reordered_points)
            reordered_radii.append(tmp_reordered_radii)
            reordered_normals.append(tmp_reordered_normals)
        # For trees != 0 the branches, points, radii, and normals need to be reversed
        connection_vessels = deepcopy(self.vessels)
        for i in range(1, len(reordered_branches)):
            for j in range(len(reordered_branches[i])):
                reordered_branches[i][j] = list(reversed(reordered_branches[i][j]))
                reordered_points[i][j] = list(reversed(reordered_points[i][j]))
                reordered_radii[i][j] = list(reversed(reordered_radii[i][j]))
                reordered_normals[i][j] = list(reversed(reordered_normals[i][j]))
                connection_vessels[i][j] = np.flip(connection_vessels[i][j], axis=0)
                proximal_points = connection_vessels[i][j][:, 0:3].copy()
                distal_points = connection_vessels[i][j][:, 3:6].copy()
                connection_vessels[i][j][:, 0:3] = distal_points
                connection_vessels[i][j][:, 3:6] = proximal_points
        # Take intermediate points along the connecting vessels
        connection_points = []
        connection_radii = []
        connection_normals = []
        for i in range(len(connection_vessels)):
            tmp_tree_points = []
            tmp_tree_radii = []
            tmp_tree_normals = []
            for j in range(len(connection_vessels[i])):
                tmp_branch_points = []
                tmp_branch_radii = []
                tmp_branch_normals = []
                for k in range(connection_vessels[i][j].shape[0]):
                    if i > 1:
                        tmp_branch_points.append(connection_vessels[i][j][k, 0:3].tolist())
                    one_fourth = connection_vessels[i][j][k, 0:3] * (3/4) + connection_vessels[i][j][k, 3:6] * (1/4)
                    tmp_branch_points.append(one_fourth.tolist())
                    mid = connection_vessels[i][j][k, 0:3] * (1/2) + connection_vessels[i][j][k, 3:6] * (1/2)
                    tmp_branch_points.append(mid.tolist())
                    three_fourths = connection_vessels[i][j][k, 0:3] * (1/4) + connection_vessels[i][j][k, 3:6] * (3/4)
                    tmp_branch_points.append(three_fourths.tolist())
                    if i > 1:
                        tmp_branch_radii.append(connection_vessels[i][j][k, 6])
                    tmp_branch_radii.append(connection_vessels[i][j][k, 6])
                    tmp_branch_radii.append(connection_vessels[i][j][k, 6])
                    tmp_branch_radii.append(connection_vessels[i][j][k, 6])
                    normal = connection_vessels[i][j][k, 3:6] - connection_vessels[i][j][k, 0:3]
                    normal = normal / np.linalg.norm(normal)
                    if i > 1:
                        tmp_branch_normals.append(normal.tolist())
                    tmp_branch_normals.append(normal.tolist())
                    tmp_branch_normals.append(normal.tolist())
                    tmp_branch_normals.append(normal.tolist())
                tmp_tree_points.append(tmp_branch_points)
                tmp_tree_radii.append(tmp_branch_radii)
                tmp_tree_normals.append(tmp_branch_normals)
            connection_points.append(tmp_tree_points)
            connection_radii.append(tmp_tree_radii)
            connection_normals.append(tmp_tree_normals)
        # Combine the reordered points, radii, and normals with the connection points, radii, and normals
        all_points = []
        all_radii = []
        all_normals = []
        # Because the first and second vessels are lofted together
        # to ensure smoothness, the first vessel is not included
        for i in range(1, len(self.connected_network)):
            network_vessel_points = []
            network_vessel_radii = []
            network_vessel_normals = []
            for j in range(len(connection_points[i])):
                vessel = []
                radii = []
                normals = []
                if i == 1:
                    vessel.extend(reordered_points[0][j])
                    vessel.extend(connection_points[0][j])
                    vessel.extend(connection_points[i][j])
                    vessel.extend(reordered_points[i][j])
                    radii.extend(reordered_radii[0][j])
                    radii.extend(connection_radii[0][j])
                    radii.extend(connection_radii[i][j])
                    radii.extend(reordered_radii[i][j])
                    normals.extend(reordered_normals[0][j])
                    normals.extend(connection_normals[0][j])
                    normals.extend(connection_normals[i][j])
                    normals.extend(reordered_normals[i][j])
                else:
                    vessel.extend(connection_points[i][j])
                    vessel.extend(reordered_points[i][j])
                    radii.extend(connection_radii[i][j])
                    radii.extend(reordered_radii[i][j])
                    normals.extend(connection_normals[i][j])
                    normals.extend(reordered_normals[i][j])
                network_vessel_points.append(vessel)
                network_vessel_radii.append(radii)
                network_vessel_normals.append(normals)
            all_points.append(network_vessel_points)
            all_radii.append(network_vessel_radii)
            all_normals.append(network_vessel_normals)
        # Now we have complete sets of points, radii, and normals
        # defining the solids for each vessel
        # Create the solid
        interp_xyz = []
        interp_radii = []
        interp_normals = []
        for i in range(len(all_points)):
            network_xyz = []
            network_r = []
            network_n = []
            network_xyzr = []
            for j in range(len(all_points[i])):
                p = np.array(all_points[i][j]).T
                r = np.array(all_radii[i][j]).T
                n = np.array(all_normals[i][j]).T
                if p.shape[1] == 2:
                    network_xyz.append(splprep(p, k=1, s=0))
                    rr = np.vstack((network_xyz[-1][1], r))
                    network_r.append(splprep(rr, k=1, s=0))
                    xyzr = np.vstack((p, r))
                    network_xyzr.append(splprep(xyzr, k=1, s=0))
                    # interp_n.append(splprep(n, k=1, s=0))
                elif p.shape[1] == 3:
                    interp_xyz.append(splprep(p, k=2, s=0))
                    rr = np.vstack((network_xyz[-1][1], r))
                    network_r.append(splprep(rr, k=1, s=0))
                    xyzr = np.vstack((p, r))
                    network_xyzr.append(splprep(xyzr, k=2, s=0))
                    # interp_n.append(splprep(n, k=2, s=0))
                else:
                    network_xyz.append(splprep(p, s=0))
                    rr = np.vstack((network_xyz[-1][1], r))
                    network_r.append(splprep(rr, k=1, s=0))
                    xyzr = np.vstack((p, r))
                    network_xyzr.append(splprep(xyzr, s=0))
                    # interp_n.append(splprep(n, s=0))
            interp_xyz.append(network_xyz)
            interp_radii.append(network_r)
            interp_normals.append(network_n)
        network_solids = []
        network_lines = []
        network_tubes = []
        for i in range(len(interp_xyz)):
            xyz = interp_xyz[i]
            r = interp_radii[i]
            lines = generate_polylines(xyz, r)
            network_lines.append(lines)
            tubes = generate_tubes(lines)
            network_tubes.append(tubes)
            model = union_tubes(tubes, lines, cap_resolution=cap_resolution)
            cell_quality = model.compute_cell_quality(quality_measure='scaled_jacobian')
            keep = cell_quality.cell_data["CellQuality"] > 0.1
            if not np.all(keep):
                print("Removing poor quality elements from the mesh.")
                keep = np.argwhere(keep).flatten()
                non_manifold_model = model.extract_cells(keep)
                non_manifold_model = non_manifold_model.extract_surface()
                fix = pymeshfix.MeshFix(non_manifold_model)
                fix.repair(verbose=True)
                hsize = model.hsize
                model = fix.mesh.compute_normals(auto_orient_normals=True)
                model.hsize = hsize
            try:
                smooth_model, smooth_wall, smooth_caps = smooth_junctions(model)
            except:
                smooth_model = None
                smooth_wall = None
                smooth_caps = None
            if not isinstance(smooth_model, type(None)):
                if smooth_model.is_manifold:
                    network_solids.append(smooth_model)
                else:
                    network_solids.append(model)
            else:
                network_solids.append(model)
        return network_solids, network_lines, network_tubes