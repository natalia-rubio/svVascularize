import numpy

from svv.tree.data.data import TreeData, TreeMap

from svv.domain.routines.c_sample import pick_from_tetrahedron, pick_from_triangle, pick_from_line
from svv.tree.utils.c_basis import basis
from svv.tree.utils.c_update import update_resistance, update_radii


def set_root(tree, **kwargs):
    """
    Set the root of the tree.
    """
    start = kwargs.get('start', None)
    direction = kwargs.get('direction', None)
    homogeneous = kwargs.get('homogeneous', True)
    volume_fraction = kwargs.get('volume_fraction', 1.0)
    threshold_adjuster = kwargs.get('threshold_adjuster', 0.9)
    layer = kwargs.get('layer', 0.0)
    start_on = kwargs.get('start_on', 'boundary')
    interior_range = kwargs.get('interior_range', [-1.0, -tree.domain_clearance])
    exterior_range = kwargs.get('exterior_range', [0.0, 1.0])
    max_attempts = kwargs.get('max_attempts', 20)
    nonconvex_sampling = kwargs.get('nonconvex_sampling', 10)
    ball_radius_scale = kwargs.get('ball_radius_scale', 2.0)
    attempts = kwargs.get('attempts', 100)
    count = 0
    finished = False
    threshold = tree.characteristic_length * volume_fraction
    root_vessel = TreeData()
    root_map = TreeMap()
    if not homogeneous:
        raise NotImplementedError("Non-homogeneous trees are not supported.")
    else:
        if tree.convex:
            while not finished:
                if count >= max_attempts:
                    threshold *= threshold_adjuster
                if start is None:
                    if start_on == 'boundary':
                        _start = tree.domain.get_boundary_points(1)
                        start_within = True
                    elif start_on == 'interior':
                        _start, _ = tree.domain.get_interior_points(1)
                        start_within = True
                    elif start_on == 'exterior':
                        raise NotImplementedError("Exterior start points are not supported for non-convex domains.")
                    else:
                        raise ValueError("Invalid start_on value: {}.".format(start_on))
                else:
                    _start = start
                    start_within = tree.domain.within(start, layer=layer)
                if direction is None:
                    if start_within:
                        _end, _ = tree.domain.get_interior_points(attempts, implicit_range=interior_range)
                        lengths = numpy.linalg.norm(_end - _start, axis=1)
                        _end = _end[lengths >= threshold, :]
                        if _end.shape[0] == 0:
                            count += 1
                            continue
                        else:
                            _end = _end[0, :]
                            finished = True
                    else:
                        closest_cell = tree.domain.boundary.find_closest_cell(_start)
                        simplices = tree.domain.boundary_nodes[tree.domain.boundary_vertices[[closest_cell], :], :]
                        rdx = tree.domain.random_generator.random((1, tree.domain.points.shape[1], 1))
                        if tree.domain.points.shape[1] == 2:
                            _end = pick_from_line(simplices, rdx)
                        elif tree.domain.points.shape[1] == 3:
                            _end = pick_from_triangle(simplices, rdx)
                        else:
                            raise ValueError("Only 2D and 3D domains are supported.")
                        finished = True
                else:
                    _end = _start + direction * threshold * numpy.linspace(0.75, 1.5, attempts).reshape(-1, 1)
                    values = tree.domain.evaluate(_end[:, :tree.domain.points.shape[1]]).flatten()
                    if len(values[values < interior_range[1]]) > 0:
                        _end = _end[values < interior_range[1], :]
                        _end_idx = tree.domain.random_generator.choice(numpy.arange(_end.shape[0]).tolist(), 1,
                                                                   replace=False)
                        _end = _end[_end_idx, :]
                    else:
                        print('Warning: No points found in domain for root endpoint.')
                        idx = numpy.argmin(values).flatten()
                        _end = _end[idx[0], :]
                    finished = True
        else:
            while not finished:
                if count >= max_attempts:
                    threshold *= threshold_adjuster
                if start is None:
                    if start_on == 'boundary':
                        _start = tree.domain.get_boundary_points(1)
                        start_within = True
                    elif start_on == 'interior':
                        _start, _ = tree.domain.get_interior_points(1, implicit_range=interior_range)
                        start_within = True
                    elif start_on == 'exterior':
                        raise NotImplementedError("Exterior start points are not supported for non-convex domains.")
                    else:
                        raise ValueError("Invalid start_on value: {}.".format(start_on))
                else:
                    _start = start
                    start_within = tree.domain.within(start, layer=layer)
                if direction is None:
                    if start_within:
                        upper = set(tree.domain.mesh_tree.query_ball_point(_start,
                                                                           threshold + threshold * ball_radius_scale)[0])
                        lower = set(tree.domain.mesh_tree.query_ball_point(_start, threshold)[0])
                        cells = list(upper.difference(lower))
                        if tree.domain.points.shape[1] == 2:
                            cells = tree.domain.random_generator.choice(cells, min(attempts, len(cells)),
                                                                        p=(tree.domain.mesh['Area'][cells] /
                                                                           tree.domain.mesh['Area'][cells].sum()),
                                                                        replace=False)
                        elif tree.domain.points.shape[1] == 3:
                            print("Cells: ", cells)
                            print("Mesh Volume: ", tree.domain.mesh['Volume'][cells])
                            print("Mesh Volume Sum: ", tree.domain.mesh['Volume'][cells].sum())
                            print("Mesh Volume Fraction: ", tree.domain.mesh['Volume'][cells] / tree.domain.mesh['Volume'][cells].sum())
                            cells = tree.domain.random_generator.choice(cells, min(attempts, len(cells)),
                                                                        p=(tree.domain.mesh['Volume'][cells] /
                                                                           tree.domain.mesh['Volume'][cells].sum()),
                                                                        replace=False)
                        else:
                            raise ValueError("Only 2D and 3D domains are supported.")
                        simplices = tree.domain.mesh_nodes[tree.domain.mesh_vertices[cells, :], :]
                        rdx = tree.domain.random_generator.random((simplices.shape[0], simplices.shape[1], 1))
                        if tree.domain.points.shape[1] == 2:
                            _end = pick_from_triangle(simplices, rdx)
                        elif tree.domain.points.shape[1] == 3:
                            _end = pick_from_tetrahedron(simplices, rdx)
                        else:
                            raise ValueError("Only 2D and 3D domains are supported.")
                        _end = _end[:, :tree.domain.points.shape[1]]
                        values = tree.domain.evaluate(_end).flatten()
                        _end = _end[values < interior_range[1], :]
                        scale = numpy.linspace(0.0, 1.0, nonconvex_sampling).reshape((1, -1, 1))
                        _mid = _start*(1-scale) + _end.reshape((_end.shape[0], 1, _end.shape[1]))*scale
                        values = tree.domain.evaluate(_mid.reshape((-1, _mid.shape[2]))).reshape((-1, nonconvex_sampling))
                        _end = _end[~numpy.any(values > 0.0, axis=1)]
                        values = values[~numpy.any(values > 0.0, axis=1), :]
                        _end = _end[~numpy.any(values[:, nonconvex_sampling//2:] > interior_range[1], axis=1)]
                        if _end.shape[0] == 0:
                            count += 1
                            continue
                        else:
                            _end = _end[0, :]
                            finished = True
                    else:
                        closest_cell = tree.domain.boundary.find_closest_cell(_start)
                        simplices = tree.domain.boundary_nodes[tree.domain.boundary_vertices[[closest_cell], :], :]
                        rdx = tree.domain.random_generator.random((1, tree.domain.points.shape[1], 1))
                        if tree.domain.points.shape[1] == 2:
                            _end = pick_from_line(simplices, rdx)
                        elif tree.domain.points.shape[1] == 3:
                            _end = pick_from_triangle(simplices, rdx)
                        else:
                            raise ValueError("Only 2D and 3D domains are supported.")
                        finished = True
                else:
                    _end = _start + direction * threshold * numpy.linspace(0.75, 1.5, attempts).reshape(-1, 1)
                    values = tree.domain.evaluate(_end[:, :tree.domain.points.shape[1]]).flatten()
                    if len(values[values < interior_range[1]]) > 0:
                        _end = _end[values < interior_range[1], :]
                        _end_idx = tree.domain.random_generator.choice(numpy.arange(_end.shape[0]).tolist(), 1,
                                                                       replace=False)
                        _end = _end[_end_idx, :]
                    else:
                        print('Warning: No points found in domain for root endpoint.')
                        idx = numpy.argmin(values).flatten()
                        _end = _end[idx[0], :]
                    finished = True
    _length = numpy.linalg.norm(_end - _start)
    if hasattr(tree.parameters.terminal_flow, '__call__'):
        _flow = tree.parameters.terminal_flow(_end[:, :tree.domain.points.shape[1]])
    else:
        _flow = tree.parameters.terminal_flow
    if len(_start.shape) == 1:
        _start = _start.reshape((1, -1))
    if len(_end.shape) == 1:
        _end = _end.reshape((1, -1))
    u, v, w = basis(_start.astype(numpy.float64), _end.astype(numpy.float64))
    root_vessel[0, 0:3] = _start
    root_vessel[0, 3:6] = _end
    root_vessel[0, 6:9] = u
    root_vessel[0, 9:12] = v
    root_vessel[0, 12:15] = w
    root_vessel[0, 18] = 0.0
    root_vessel[0, 19] = 1.0
    root_vessel[0, 20] = _length
    root_vessel[0, 22] = _flow
    root_vessel[0, 26] = 0.0
    root_vessel[0, 28] = 1.0
    connectivity = numpy.nan_to_num(tree.data[:, 15:18], nan=-1.0).astype(numpy.int64)
    update_resistance(root_vessel, connectivity, tree.parameters.murray_exponent, tree.parameters.kinematic_viscosity*tree.parameters.fluid_density)
    update_radii(root_vessel, connectivity, tree.parameters.root_pressure, tree.parameters.terminal_pressure)
    root_map[0] = {'upstream': [], 'downstream': []}
    return root_vessel, root_map
