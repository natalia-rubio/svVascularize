import os
import re
import subprocess
import numpy as np

def write_node(path_base, points):
    node_file = path_base + ".node"
    npoints = points.shape[0]
    with open(node_file, "w") as f:
        f.write(f"{npoints} 3 0 0\n")
        for i in range(npoints):
            nid = i + 1
            x, y, z = points[i]
            f.write(f"{nid} {x:.17g} {y:.17g} {z:.17g}\n")

def read_node(basepath):
    """
    Read a TetGen .node file.
    Returns:
    points: numpy array of shape (npoints, 3)
    ids: numpy array of shape (npoints,) with 1-based node ids
    """
    node_file = basepath + ".node"
    with open(node_file, "r") as f:
        lines = [ln for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]

    header = lines[0].strip().split()
    npoints = int(header[0])
    dim = int(header[1])
    if dim != 3:
        raise ValueError(f"Expected 3D nodes, got dim={dim}")
    # header[2] num node attributes (ignored), header[3] markers (ignored)

    points = np.zeros((npoints, 3), dtype=float)
    ids = np.zeros((npoints,), dtype=int)
    for i in range(npoints):
        parts = lines[i + 1].strip().split()
        nid = int(parts[0])
        x, y, z = map(float, parts[1:4])
        ids[i] = nid
        points[i] = [x, y, z]

    # Ensure ids are 1..n and sorted accordingly
    # TetGen typically uses 1-based consecutive ids
    order = np.argsort(ids)
    points = points[order]
    ids = ids[order]
    return points, ids

def read_ele(basepath):
    """
    Read a TetGen .ele file.
    Returns:
    tets: numpy array of shape (ntets, 4) with node ids (1-based)
    tet_ids: numpy array of shape (ntets,) with 1-based tet ids
    existing_attrs: optional numpy array of existing attributes per tet or None
    """
        
    ele_file = basepath + ".ele"
    with open(ele_file, "r") as f:
        lines = [ln for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]
    header = lines[0].strip().split()
    ntets = int(header[0])
    nnodes_per_tet = int(header[1])
    if nnodes_per_tet != 4:
        raise ValueError(f"Expected 4 nodes per tetra, got {nnodes_per_tet}")
    nattrs = int(header[2]) if len(header) >= 3 else 0
         

    tets = np.zeros((ntets, 4), dtype=int)
    tet_ids = np.zeros((ntets,), dtype=int)
    existing_attrs = None
    if nattrs > 0:
        existing_attrs = np.zeros((ntets, nattrs), dtype=float)

    for i in range(ntets):
        parts = lines[i + 1].strip().split()
        tid = int(parts[0])
        tet_ids[i] = tid
        tets[i] = list(map(int, parts[1:5]))
        if nattrs > 0:
            existing_attrs[i] = list(map(float, parts[5:5 + nattrs]))

    # Sort by tet id to be safe
    order = np.argsort(tet_ids)
    tets = tets[order]
    tet_ids = tet_ids[order]
    if existing_attrs is not None:
        existing_attrs = existing_attrs[order]

    return tets, tet_ids, existing_attrs

def write_ele_with_volume_constraints(basepath, tets, tet_ids, vol_constraints):
    """
    Write a TetGen .ele file with one attribute per tet: the max volume constraint.
    vol_constraints should be length ntets.
    """
    ele_file = basepath + ".ele"
    ntets = tets.shape[0]
    with open(ele_file, "w") as f:
    # Header: ntets, 4 nodes per tet, 1 attribute
        f.write(f"{ntets} 4 1\n")
        for i in range(ntets):
        # TetGen expects 1-based ids
            tid = int(tet_ids[i])
            n1, n2, n3, n4 = tets[i]
            vc = float(vol_constraints[i])
            f.write(f"{tid} {n1} {n2} {n3} {n4} {vc}\n")

def compute_tet_volumes(points, tets):
    """
    Compute volumes of tetrahedra.
    points: (npoints,3), indexed by 0-based but tets contain 1-based indices.
    tets: (ntets,4) with 1-based node ids.
    Returns: (ntets,) volumes (positive).
    """
    # Convert to 0-based indices
    T = tets - 1
    p1 = points[T[:, 0], :]
    p2 = points[T[:, 1], :]
    p3 = points[T[:, 2], :]
    p4 = points[T[:, 3], :]
    # Volume = |det([p2-p1, p3-p1, p4-p1])| / 6
    v = np.abs(np.linalg.det(np.stack([p2 - p1, p3 - p1, p4 - p1], axis=2))) / 6.0
    return v

def compute_tet_centroids(points, tets):
    """
    Compute centroids of tets.
    Returns: (ntets,3)
    """
    T = tets - 1
    p1 = points[T[:, 0], :]
    p2 = points[T[:, 1], :]
    p3 = points[T[:, 2], :]
    p4 = points[T[:, 3], :]
    return (p1 + p2 + p3 + p4) / 4.0

def parse_bounds_from_text_blob(text):
    """
    Parse a region list text like the one provided to extract bounds.
    Returns: list of (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    # Regex for lines like: X Bounds:   -1.616e+00, -1.150e+00
    # Handle scientific notation and spaces
    def grab_bounds(axis):
        pattern = rf"{axis}\sBounds:\s([-\deE.+]+),\s*([-\deE.+]+)"
        return [tuple(map(float, m)) for m in re.findall(pattern, text)]
    
    xs = grab_bounds("X")
    ys = grab_bounds("Y")
    zs = grab_bounds("Z")
    # They should appear in groups; assume same count
    n = min(len(xs), len(ys), len(zs))
    bounds = []
    for i in range(n):
        xmin, xmax = xs[i]
        ymin, ymax = ys[i]
        zmin, zmax = zs[i]
        bounds.append((xmin, xmax, ymin, ymax, zmin, zmax))
    return bounds

def tets_inside_bounds(centroids, bounds_list):
    """
    Mark which tets have centroids inside any of the given bounds.
    bounds_list: list of (xmin, xmax, ymin, ymax, zmin, zmax)
    Returns: boolean array (ntets,)
    """
    if not bounds_list:
        return np.zeros((centroids.shape[0],), dtype=bool)
    inside = np.zeros((centroids.shape[0],), dtype=bool)
    x = centroids[:, 0]
    y = centroids[:, 1]
    z = centroids[:, 2]
    for bounds in bounds_list:
        (xmin, xmax, ymin, ymax, zmin, zmax) = bounds.bounds
        mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & (z >= zmin) & (z <= zmax)
        inside |= mask
    return inside

def extract_points_tets(mesh_obj):
    """
    Return points (Nx3 float) and tets (Mx4 int, 0-based) from various mesh object types.
    """
    # tuple or dict
    if isinstance(mesh_obj, (tuple, list)) and len(mesh_obj) >= 2:
        points, tets = mesh_obj[0], mesh_obj[1]
        points = np.asarray(points, dtype=float)
        tets = np.asarray(tets, dtype=int)
        # normalize indexing to 0-based
        if tets.min() == 1:
            tets = tets - 1
        return points, tets
    else:
        print("Unsupported mesh object type for extraction")
        return None, None

def refine_tetgen_mesh_in_regions(
    basepath,
    regions_bounds,
    target_volume_in_region=None,
    refine_factor=None,
    outside_volume=1e30,
    tetgen_exe="tetgen",
    dry_run=False
    ):
    """
    Refine an existing TetGen mesh (.node/.ele) inside specified regions.
    Inputs:
    basepath: path without extension to the TetGen mesh, e.g., 'mesh' for mesh.node and mesh.ele
    regions_bounds: list of (xmin, xmax, ymin, ymax, zmin, zmax)
    target_volume_in_region: absolute max volume to enforce inside regions (float), optional
    refine_factor: if provided, per-tet target inside = original_volume * refine_factor
                   e.g., 0.25 to aim for ~4x refinement locally
    outside_volume: large max volume outside regions (default very large so no refinement outside)
    tetgen_exe: name/path of the tetgen executable
    dry_run: if True, do not call TetGen; just write updated .ele

    Behavior:
        - If both target_volume_in_region and refine_factor are provided, refine_factor wins.
        - If neither is provided, refine_factor defaults to 0.25.
    """
    points, node_ids = read_node(basepath)
    tets, tet_ids, existing_attrs = read_ele(basepath)

    centroids = compute_tet_centroids(points, tets)
    volumes = compute_tet_volumes(points, tets)
    mask_inside = tets_inside_bounds(centroids, regions_bounds)

    if refine_factor is None and target_volume_in_region is None:
        refine_factor = 0.25  # Default: 4x smaller average element volume locally

    vol_constraints = np.full((tets.shape[0],), outside_volume, dtype=float)

    if refine_factor is not None:
        target_vols = volumes * refine_factor
        vol_constraints[mask_inside] = target_vols[mask_inside]
    else:
        vol_constraints[mask_inside] = float(target_volume_in_region)

    # Write updated .ele with one attribute (max volume constraint)
    write_ele_with_volume_constraints(basepath, tets, tet_ids, vol_constraints)

    # Call TetGen to refine: -r refine existing mesh, -a use attributes as max volumes
    if not dry_run:
        cmd = [tetgen_exe, "-r", "-a", basepath]
        print("Running:", " ".join(cmd))
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print("TetGen failed:")
            print(res.stdout)
            print(res.stderr)
            raise RuntimeError("TetGen refine failed")
        else:
            print("TetGen output:")
            print(res.stdout)
            # Refined files are typically basepath.1.node, basepath.1.ele, etc.
            print(f"Refined mesh written as {basepath}.1.node and {basepath}.1.ele")