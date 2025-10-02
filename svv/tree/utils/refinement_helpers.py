
import numpy as np
import pyvista as pv
import tetgen
import pymeshfix

try:
    from scipy.spatial import cKDTree as KDTree
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _ensure_surface_polydata(mesh):
    """Convert an UnstructuredGrid or PolyData to a triangulated PolyData surface."""
    if isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.extract_surface()
    elif not isinstance(mesh, pv.PolyData):
        raise TypeError("Mesh must be a pyvista.UnstructuredGrid or pyvista.PolyData.")
    # Ensure triangular faces
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    return mesh


def _average_edge_length(poly):
    """Estimate average edge length for a surface PolyData."""
    try:
        edges = poly.extract_edges()
        if edges.n_cells > 0:
            # Each line cell in PolyData lines array: [2, i, j]
            lines = edges.lines.reshape(-1, 3)[:, 1:]
            pts = edges.points
            seg_len = np.linalg.norm(pts[lines[:, 0]] - pts[lines[:, 1]], axis=1)
            if len(seg_len) > 0:
                return float(np.mean(seg_len))
    except Exception:
        pass
    # Fallback: nearest neighbor distance among points
    pts = poly.points
    if pts.shape[0] > 1:
        if _HAS_SCIPY:
            kd = KDTree(pts)
            dists, _ = kd.query(pts, k=2)
            # dists[:, 0] is zero (self); take the second column
            nn = dists[:, 1]
        else:
            # Simple O(N^2) fallback for small meshes
            nn = []
            for i in range(pts.shape[0]):
                di = np.linalg.norm(pts - pts[i], axis=1)
                di[i] = np.inf
                nn.append(np.min(di))
            nn = np.asarray(nn)
        return float(np.mean(nn))
    # Final fallback: use bbox diagonal scaled
    bounds = poly.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    diag = np.sqrt((bounds[1] - bounds[0])**2 + (bounds[3] - bounds[2])**2 + (bounds[5] - bounds[4])**2)
    return float(diag / max(10, np.sqrt(max(poly.n_points, 1))))


def refine_tetgen_mesh_in_regions(
    volume_or_surface,
    junction_regions,
    subdivide_n=1,
    subdivide_method="linear",  # "linear" or "loop" or "butterfly"
    distance_tol=None,
    tetgen_switches="pq1.2/20MVYSJ",
    meshfix_verbose=False
):
    """
    Locally refine the fluid surface near junction_regions and re-tetrahedralize.

    Parameters
    ----------
    volume_or_surface : pyvista.UnstructuredGrid or pyvista.PolyData
        Fluid volume mesh (UnstructuredGrid) or fluid surface (PolyData).
        If a volume mesh is provided, its surface is extracted and refined.
    junction_regions : list of pyvista.UnstructuredGrid
        List of region patches (surface-like UnstructuredGrid) marking junction neighborhoods.
    subdivide_n : int
        Number of subdivision iterations for selected triangles.
    subdivide_method : str
        Subdivision method: "linear", "loop", or "butterfly".
    distance_tol : float or None
        Vertex-to-region proximity threshold for selecting triangles to refine.
        If None, an adaptive tolerance is computed per region from average edge length.
    tetgen_switches : str
        TetGen switches controlling quality and refinement. Example: "pq1.2/20MVYSJ".
    meshfix_verbose : bool
        If True, makes mesh repair step verbose.

    Returns
    -------
    fluid_volume_mesh : pyvista.UnstructuredGrid
        New tetrahedral volume mesh refined near junction regions.
    nodes : ndarray
        Node coordinates returned by TetGen.
    elements : ndarray
        Tetrahedral connectivity returned by TetGen.
    """
    # Ensure we have a triangulated fluid surface
    fluid_surface = _ensure_surface_polydata(volume_or_surface)

    # Prepare face connectivity (triangles only)
    faces = fluid_surface.faces.reshape(-1, 4)  # [3, i, j, k]
    tri_conn = faces[:, 1:]
    n_faces = tri_conn.shape[0]

    # Union mask for cells to refine
    refine_mask = np.zeros(n_faces, dtype=bool)

    # Precompute fluid surface points once
    fluid_pts = fluid_surface.points

    # Iterate through region patches
    for i, region in enumerate(junction_regions):
        region_surface = _ensure_surface_polydata(region)
        region_pts = region_surface.points

        # Determine proximity threshold
        tol = distance_tol
        if tol is None:
            # Use region-local average edge length as a reasonable scale
            tol = max(1e-8, 0.5 * _average_edge_length(region_surface))

        # Nearest-neighbor distances from fluid surface points to this region
        if _HAS_SCIPY:
            kd = KDTree(region_pts)
            dists, _ = kd.query(fluid_pts, k=1)
        else:
            # Fallback: simple nearest distances (O(N*M), ok for modest sizes)
            dists = np.empty(fluid_pts.shape[0], dtype=float)
            for j in range(fluid_pts.shape[0]):
                dists[j] = np.min(np.linalg.norm(region_pts - fluid_pts[j], axis=1))

        near_point_mask = dists <= tol

        # Mark triangles with at least one vertex near the region
        near_cells = near_point_mask[tri_conn].any(axis=1)
        refine_mask |= near_cells

    # Split into coarse and refined sets
    refine_ids = np.where(refine_mask)[0]
    coarse_ids = np.where(~refine_mask)[0]

    # Extract patches
    refined_patch = fluid_surface.extract_cells(refine_ids) if refine_ids.size > 0 else None
    coarse_patch = fluid_surface.extract_cells(coarse_ids) if coarse_ids.size > 0 else None

    # Subdivide only the refined patch
    if refined_patch is not None and refined_patch.n_cells > 0 and subdivide_n > 0:
        print("Refining", refined_patch.n_cells, "triangles with", subdivide_n, "subdivisions using", subdivide_method)
        refined_patch = refined_patch.extract_surface().triangulate().subdivide(subdivide_n).triangulate().clean()

    elif refined_patch is not None and refined_patch.n_cells > 0:
        refined_patch = refined_patch.extract_surface().triangulate().clean()
        print("No subdivisions applied; retaining", refined_patch.n_cells, "triangles.")
    else:
        print("No triangles selected for refinement.")
    if coarse_patch is not None and coarse_patch.n_cells > 0:
        coarse_patch  =  coarse_patch.extract_surface().triangulate().clean()
        
    import pdb; pdb.set_trace()
    # Merge back together
    if refined_patch is not None and coarse_patch is not None:
        #refined_surface = coarse_patch.merge(refined_patch, merge_points = True, main_has_priority = False)
        refined_surface = refined_patch.merge(coarse_patch, merge_points = True, main_has_priority = True)
        #refined_surface = refined_patch | coarse_patch
        #refined_surface = pv.merge([refined_patch, coarse_patch], merge_points=True)
    elif refined_patch is not None:
        refined_surface = refined_patch
    elif coarse_patch is not None:
        refined_surface = coarse_patch
    else:
        # No cells; return original tetrahedralization
        refined_surface = fluid_surface.triangulate().clean()

    # Ensure triangulated (subdivide should give triangles, but be safe)
    if not refined_surface.is_all_triangles:
        refined_surface = refined_surface.triangulate()

    # Repair manifold/watertightness
    fixer = pymeshfix.MeshFix(refined_surface)
    fixer.repair(verbose=meshfix_verbose, joincomp=True, remove_smallest_components=True)
    repaired_surface = fixer.mesh
    repaired_surface = refined_surface

    # edges = refined_surface.extract_feature_edges(boundary_edges=True, non_manifold_edges=True) 
    # if edges.n_cells == 0: 
    #     repaired_surface = refined_surface # already closed/manifold 
    # else: # Try lighter fixes first 
    #     refined_surface = refined_surface.triangulate().clean(tolerance=0.0) 
    #     refined_surface = refined_surface.compute_normals(auto_orient_normals=True) 
    #     refined_surface = refined_surface.fill_holes(hole_size=1000)
    # repaired_surface = refined_surface
    # Tetrahedralize
    
    tet = tetgen.TetGen(repaired_surface)
    nodes, elements = tet.tetrahedralize(switches=tetgen_switches)
    fluid_volume_mesh = tet.grid

    return fluid_volume_mesh, nodes, elements