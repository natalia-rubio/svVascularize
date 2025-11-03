from boundary_layer import *
from extract_faces import extract_faces
import tetgen
import svtoolkit
from scipy.spatial import cKDTree
from itertools import permutations
import vtk


cyl = svtoolkit.domain.remeshing.remesh.remesh_surface(pv.Cylinder().triangulate())
cyl = svtoolkit.domain.remeshing.remesh.remesh_surface(cyl)

tet = tetgen.TetGen(cyl)
tet.tetrahedralize()

res = extract_faces(tet.grid.extract_surface(), tet.grid)
wall = res[1][0]

boundary = BoundaryLayer(wall, layer_thickness=1E-2)
combined, interior, layers = boundary.generate()

"""
def tetra_filter(self, layers):
    
    This function filters the boundary layer mesh to yield a tetrahedral mesh.
    We assume that the volume elements of the unstructured grid are well-posed
    in that they define a closed volume and represent a valid element template.
    However, we do not make assumptions regarding orientation of the volume
    elements.
    :param layers:
    :return:
    
    decomposition = {pv.CellType.TETRA: {'ordering': list(permutations([1, 2, 3, 4])),
                                         'tet_number': 1},
                     pv.CellType.WEDGE: {'ordering': [(1, 2, 3, 5), (1, 4, 5, 6), (1, 3, 5, 6)],
                                         'tet_number': 3},
                     pv.CellType.HEXAHEDRON: {'ordering': [],
                                              'tet_number': 6}}
    output_points = np.array(layers.points, dtype=np.float64)
    celltypes = layers.celltypes
    unique_celltypes = np.unique(celltypes)
    if unique_celltypes.shape[0] == 1 and unique_celltypes[0] == pv.CellType.TETRA:
        return layers
    linear_volume_cell_types = [pv.CellType.TETRA, pv.CellType.WEDGE, pv.CellType.HEXAHEDRON]
    # iterate through all of the cells in the boundary layer an get the node ordering for
    # each of the different cell types
    # Supported Volume Elements
    tetra_nodes = []
    tetra_entites = []
    wedge_nodes = []
    wedge_entities = []
    hex_nodes = []
    hex_entities = []
    # Supported Surface Elements
    tri_nodes = []
    tri_entities = []
    quad_nodes = []
    quad_entities = []
    for i in tqdm.trange(layers.n_cells, desc="Filtering Boundary Layer Volume Elements", leave=False):
        cell = layers.get_cell(i)
        nodes = cell.point_ids
        nodes.insert(0, len(nodes))
        entities = layers.cell_data['EntityID'][i]
        if cell.dimension == 3:
            if cell.type == pv.CellType.TETRA:
                tetra_nodes.append(nodes)
                tetra_entites.append(entities)
            elif cell.type == pv.CellType.WEDGE:
                wedge_nodes.append(nodes)
                wedge_entities.append(entities)
            elif cell.type == pv.CellType.HEXAHEDRON:
                hex_nodes.append(nodes)
                hex_entities.append(entities)
            else:
                raise ValueError("Unsupported volume cell type: {}.".format(cell.type))
        elif cell.dimension == 2:
            if cell.type == pv.CellType.TRIANGLE:
                tri_nodes.append(nodes)
                tri_entities.append(entities)
            elif cell.type == pv.CellType.QUAD:
                quad_nodes.append(nodes)
                quad_entities.append(entities)
            else:
                raise ValueError("Unsupported surface cell type: {}.".format(cell.type))
    tetra_nodes = np.array(tetra_nodes, dtype=int)
    tetra_entites = np.array(tetra_entites, dtype=int)
    wedge_nodes = np.array(wedge_nodes, dtype=int)
    wedge_entities = np.array(wedge_entities, dtype=int)
    hex_nodes = np.array(hex_nodes, dtype=int)
    hex_entities = np.array(hex_entities, dtype=int)
    tri_nodes = np.array(tri_nodes, dtype=int)
    tri_entities = np.array(tri_entities, dtype=int)
    quad_nodes = np.array(quad_nodes, dtype=int)
    quad_entities = np.array(quad_entities, dtype=int)
    tetra_cells = []
    all_tetra_entities = []
    # Filter all types of cells into tetrahedra
    for i in range(len(linear_volume_cell_types)):
        volume_cell_type = linear_volume_cell_types[i]
        if volume_cell_type == pv.CellType.TETRA:
            nodes = tetra_nodes
            if nodes.shape[0] == 0:
                continue
            print("Number of Tetrahedra: ", nodes.shape[0])
            tetra_cells.extend(nodes.tolist())
        elif volume_cell_type == pv.CellType.WEDGE:
            nodes = wedge_nodes
            if nodes.shape[0] == 0:
                continue
            print("Number of Wedges: ", nodes.shape[0])
            decomposed_tetra_nodes = nodes[:, decomposition[pv.CellType.WEDGE]['ordering']].copy()
            decomposed_entities = np.zeros((decomposed_tetra_nodes.shape[0], decomposed_tetra_nodes.shape[1], 1),
                                           dtype=int)
            decomposed_entities[:, :, 0] = wedge_entities.reshape(-1, 1)
            decomposed_tetra_nodes = decomposed_tetra_nodes.reshape(-1, 4)
            new_tetra_nodes = np.zeros((decomposed_tetra_nodes.shape[0], 5), dtype=int)
            new_tetra_nodes[:, 0] = 4
            new_tetra_nodes[:, 1:] = decomposed_tetra_nodes
            tetra_cells.extend(new_tetra_nodes.tolist())
            all_tetra_entities.extend(decomposed_entities.ravel().tolist())
        elif volume_cell_type == pv.CellType.HEXAHEDRON:
            nodes = hex_nodes
            if nodes.shape[0] == 0:
                continue
            print("Number of Hexahedra: ", nodes.shape[0])
            decomposed_tetra_nodes = nodes[:, decomposition[pv.CellType.HEXAHEDRON]['ordering']].copy()
            decomposed_entities = np.zeros((decomposed_tetra_nodes.shape[0], decomposed_tetra_nodes.shape[1], 1),
                                           dtype=int)
            decomposed_entities[:, :, 0] = hex_entities.reshape(-1, 1)
            decomposed_tetra_nodes = decomposed_tetra_nodes.reshape(-1, 4)
            new_tetra_nodes = np.zeros((decomposed_tetra_nodes.shape[0], 5), dtype=int)
            new_tetra_nodes[:, 0] = 4
            new_tetra_nodes[:, 1:] = decomposed_tetra_nodes
            tetra_cells.extend(new_tetra_nodes.tolist())
            all_tetra_entities.extend(decomposed_entities.ravel().tolist())
    tetra_cells = np.array(tetra_cells, dtype=int)
    tetra_cell_type = [pv.CellType.TETRA for i in range(tetra_cells.shape[0])]
    if tetra_cells.max() > output_points.shape[0]:
        raise ValueError("Invalid node index.")
    tet_layers = pv.UnstructuredGrid(tetra_cells.ravel(), tetra_cell_type, output_points)
    quality = tet_layers.compute_cell_quality().cell_data['CellQuality'].flatten()
    tetra_ordering = set(decomposition[pv.CellType.TETRA]['ordering']) - set(
        decomposition[pv.CellType.WEDGE]['ordering'])
    for i in tqdm.trange(len(tetra_ordering), desc="Reordering Tetrahedra", leave=False):
        ordering = np.array(tetra_ordering.pop())
        ids = np.argwhere(quality < 0)
        changed = tetra_cells[ids, ordering].copy()
        tetra_cells[ids.flatten(), 1:] = changed
        tet_layers = pv.UnstructuredGrid(tetra_cells.ravel(), tetra_cell_type, output_points)
        quality = tet_layers.compute_cell_quality().cell_data['CellQuality'].flatten()
        if not np.any(quality < 0):
            break
    if np.any(quality < 0):
        raise ValueError(
            "Tetrahedrons at indices {} have negative jacobian.".format(np.argwhere(quality < 0).flatten()))
    # Add surface cells to the tetrahedral mesh
    all_cells = []
    all_entities = []
    all_cell_types = []
    all_cells.extend(tetra_cells.ravel().tolist())
    all_entities.extend(all_tetra_entities)
    print("Number of Tetrahedra Entities: ", len(all_entities))
    all_cell_types.extend([pv.CellType.TETRA for i in range(tetra_cells.shape[0])])
    all_cells.extend(tri_nodes.ravel().tolist())
    all_entities.extend(tri_entities.ravel().tolist())
    print("Number of Triangle Entities: ", len(all_entities))
    all_cell_types.extend([pv.CellType.TRIANGLE for i in range(tri_nodes.shape[0])])
    all_cells.extend(quad_nodes.ravel().tolist())
    all_entities.extend(quad_entities.ravel().tolist())
    print("Number of Quad Entities: ", len(quad_entities))
    all_cell_types.extend([pv.CellType.QUAD for i in range(quad_nodes.shape[0])])
    tet_layers = pv.UnstructuredGrid(all_cells, all_cell_types, output_points)
    tet_layers.cell_data['EntityID'] = np.array(all_entities)
    return tet_layers

def compute_wedge_quality(points):

    v0 = points[1] - points[0]  # Bottom edge
    v1 = points[2] - points[0]  # Bottom edge
    v2 = points[3] - points[0]  # Vertical edge
    v3 = points[4] - points[1]  # Top edge (parallel to v0)
    v4 = points[5] - points[2]  # Top edge (parallel to v1)

    bottom_cross = np.cross(v0, v1)
    top_cross = np.cross(v3, v4)
    parallelism = np.dot(bottom_cross / np.linalg.norm(bottom_cross), top_cross / np.linalg.norm(top_cross))

    quality = np.linalg.norm(bottom_cross) * np.linalg.norm(top_cross) / (
                np.linalg.norm(v0) * np.linalg.norm(v1) * np.linalg.norm(v2))
    quality *= parallelism

    return min(quality, 1.0)


def assess_wedge_quality(mesh):

    qualities = []
    for cell in mesh.cells.reshape(-1, 7)[:, 1:]:  # Skip the cell size (7), extract point indices
        points = mesh.points[cell]
        quality = compute_wedge_quality(points)
        qualities.append(quality)
    return qualities


def quality(mesh, measure='scaled_jacobian'):
    vols = np.argwhere(mesh.celltypes == 13).flatten()
    submesh = mesh.extract_cells(vols)
    orders = np.array(list(permutations([1, 2, 3, 4, 5, 6])))
    old_cells = deepcopy(submesh.cells).reshape(-1, 7)
    new_cells = deepcopy(old_cells)
    test = deepcopy(submesh)
    for i in range(orders.shape[0]):
        ordering = orders[i, :].reshape(1, -1)
        print(ordering.shape)
        qual = np.array(assess_wedge_quality(test))
        ids = set(np.argwhere(qual < 0).flatten().tolist())
        test_surf = test.extract_surface()
        non_manifold = test_surf.extract_feature_edges(boundary_edges=False, manifold_edges=False, feature_edges=False,
                                                       non_manifold_edges=True)
        bad_cells = set(non_manifold.cell_data['vtkOriginalCellIds'].tolist())
        ids = ids.union(bad_cells)
        ids = np.array(list(ids)).reshape(-1, 1)
        print(len(ids))
        new_cells[ids.flatten(), 1:] = old_cells[ids, ordering]
        test = pv.UnstructuredGrid(new_cells.ravel(), submesh.celltypes, submesh.points)
        #qual = np.array(assess_wedge_quality(test))
        #test_surf = test.extract_surface()
        #non_manifold = test_surf.extract_feature_edges(boundary_edges=False, manifold_edges=False, feature_edges=False, non_manifold_edges=True)
        # bad_cells = non_manifold.cell_data['vtkOriginalCellIds']
        qual = np.array(assess_wedge_quality(test))
        ids = set(np.argwhere(qual < 0).flatten().tolist())
        test_surf = test.extract_surface()
        non_manifold = test_surf.extract_feature_edges(boundary_edges=False, manifold_edges=False, feature_edges=False,
                                                       non_manifold_edges=True)
        bad_cells = set(non_manifold.cell_data['vtkOriginalCellIds'].tolist())
        ids = ids.union(bad_cells)
        ids = np.array(list(ids)).reshape(-1, 1)
        if ids.shape[0] == 0:
            break
        #new_cells[:, 1:] = old_cells[:, orders[i, :]]
        #print('{}: {} -> {}'.format(i, old_cells[0,:], new_cells[0,:]))
        #test = pv.UnstructuredGrid(new_cells.ravel(), submesh.celltypes, submesh.points)
        #qual = test.compute_cell_quality(quality_measure=measure, null_value=np.nan).cell_data["CellQuality"]
        #qual = test.compute_cell_sizes().cell_data["Volume"]
        #qual = np.array(assess_wedge_quality(test))
        #if np.any(qual > 0):
        #    print(i)
        #    break
        #if np.any(qual == np.nan):
        #    print('nan')
        #    break
    return test

def wedge_decomp(mesh):
    vols = np.argwhere(mesh.celltypes == 13).flatten()
    submesh = mesh.extract_cells(vols)
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    triangulator = vtk.vtkOrderedTriangulator()
    triangulator.SetUseTemplates(0)
    for i in range(submesh.n_cells):
        cell = submesh.extract_cells(i)
        cell_points = cell.points
        triangulator.InitTriangulation(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        triangulator.PreSortedOff()
        for j, pt in enumerate(cell_points):
            pid = points.InsertNextPoint(pt)  # Insert point into global points list
            triangulator.InsertPoint(pid, pt, None, 0.0)
        triangulator.Update()
        triangulator.Triangulate()
        new_tetras = vtk.vtkUnstructuredGrid()
        triangulator.AddTetras(0, new_tetras)
        for k in range(new_tetras.GetNumberOfCells()):
            tetra = new_tetras.GetCell(k)
            ids = tetra.GetPointIds()
            cell_ids = [ids.GetId(l) for l in range(4)]  # Get the point IDs for the tetra
            cells.InsertNextCell(4, cell_ids)  # Insert the tetrahedron cell
    tetra_grid = pv.UnstructuredGrid(cells, np.full(cells.GetNumberOfCells(), 10), points)
    return tetra_grid
    
"""
#outer_cells = np.argwhere(layers.cell_data["EntityID"] == 2).flatten()
#outer = layers.extract_cells(outer_cells).extract_surface()
#boundaries = outer.extract_feature_edges(boundary_edges=True,manifold_edges=False,feature_edges=False,non_manifold_edges=False)
#boundaries = boundaries.split_bodies()
#cap_1 = svtoolkit.domain.remeshing.remesh.remesh_surface_2d(boundaries[0])
#cap_2 = svtoolkit.domain.remeshing.remesh.remesh_surface_2d(boundaries[1])
#outer_tree = cKDTree(outer.points)
#cap_1_dists, outer_cap_1_inds = outer_tree.query(cap_1.points)
#cap_2_dists, outer_cap_2_inds = outer_tree.query(cap_2.points)
#cap_1.points[:boundaries[0].n_points, :] = outer.points[outer_cap_1_inds[:boundaries[0].n_points], :]
#cap_2.points[:boundaries[1].n_points, :] = outer.points[outer_cap_2_inds[:boundaries[1].n_points], :]
#outer_surface = pv.merge([outer, cap_1, cap_2])
#layer_tree = cKDTree(layers.points)
#layers = boundary.tetra_filter(layers)

"""
cube = svtoolkit.Domain(pv.Cube())
cube.create()
cube.create()
cube.solve()
cube.build()

t = svtoolkit.Tree()
t.parameters.terminal_pressure = 1333*99.5
t.set_domain(cube)
t.set_root()
t.add()

sim = svtoolkit.Simulation(t)
sim.build_meshes(tissue=False)
sim.extract_faces()
sim.construct_3d_fluid_simulation()
sim.write_fluid_simulation()

wall = sim.fluid_domain_faces[0]['walls'][0]
boundary = BoundaryLayer(wall, layer_thickness=5E-2, remesh_vol=True)
combined, interior, layers = boundary.generate()
combined, interior = boundary.combine_layers(layers)
print(combined)
"""