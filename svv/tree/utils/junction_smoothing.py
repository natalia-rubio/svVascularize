"""
Junction smoothing utilities for vascular tree meshes.

This module provides functionality to detect vessel junctions and apply
smoothing algorithms to improve mesh quality at intersection points.
"""
from __future__ import annotations
import pdb
import numpy as np
import pyvista as pv
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pymeshfix
from svv.utils.remeshing.remesh import remesh_surface
from svv.simulation.utils.extract_faces import extract_faces
import pyvista as pv
from pyvista import examples
mesh = examples.download_bunny_coarse().triangulate().clean()

cpos = [
    (-0.02788175062966399, 0.19293295656233056, 0.4334449972621349),
    (-0.053260899930287015, 0.08881197167521734, -9.016948161029588e-05),
    (-0.10170607813337212, 0.9686438023715356, -0.22668272496584665),
]
def plot_subdivisions(mesh, a, b):
    display_args = dict(show_edges=True, color=True)
    p = pv.Plotter(shape=(3, 3))

    for i in range(3):
        p.subplot(i, 0)
        p.add_mesh(mesh, **display_args)
        p.add_text('Original Mesh')

    def row_plot(row, subfilter):
        subs = [a, b]
        for i in range(2):
            p.subplot(row, i + 1)
            p.add_mesh(mesh.subdivide(subs[i], subfilter=subfilter), **display_args)
            p.add_text(f'{subfilter} subdivision of {subs[i]}')

    row_plot(0, 'linear')
    row_plot(1, 'butterfly')
    row_plot(2, 'loop')

    p.link_views()
    p.view_isometric()
    return p

def detect_junctions(tree_data, vessel_map, connectivity, tolerance=1e-6):
    """
    Detect vessel junctions in the tree structure.
    
    Parameters
    ----------
    tree_data : TreeData
        The tree data containing vessel information
    vessel_map : TreeMap
        Mapping of vessel connections
    connectivity : np.ndarray
        Connectivity array with parent-child relationships
    tolerance : float
        Tolerance for point matching
        
    Returns
    -------
    junctions : Dict[int, List[int]]
        Dictionary mapping junction points to vessel indices
    junction_points : np.ndarray
        Array of junction point coordinates
    """
    junctions = defaultdict(list)
    junction_radii = {}
    junction_coords = []
    
    # Get all vessel endpoints
    proximal_points = tree_data.get('proximal')
    distal_points = tree_data.get('distal')
    radii = tree_data.get('radius')

    # Create a mapping from coordinates to vessel indices
    point_to_vessels = defaultdict(list)
    
    for i in range(len(tree_data)):
        # Add proximal point
        prox_coord = tuple(proximal_points[i])
        point_to_vessels[prox_coord].append(i)
        
        # Add distal point
        dist_coord = tuple(distal_points[i])
        point_to_vessels[dist_coord].append(i)
    
    # Find points where multiple vessels meet (junctions)
    junction_id = 0
    for coord, vessel_list in point_to_vessels.items():
        if len(vessel_list) > 1:  # Multiple vessels meet at this point
            junctions[junction_id] = vessel_list
            junction_radii[junction_id] = max([radii[v] for v in vessel_list])
            junction_coords.append(coord)
            junction_id += 1
    
    return dict(junctions), junction_radii, np.array(junction_coords)


def identify_junction_regions(mesh, junctions, junction_radii, junction_points,radius_factor=2.0):
    """
    Identify mesh regions around junction points for smoothing.
    
    Parameters
    ----------
    mesh : pv.PolyData
        The surface mesh
    junction_points : np.ndarray
        Array of junction point coordinates
    radius_factor : float
        Factor to determine smoothing radius around junctions
        
    Returns
    -------
    junction_regions : List[pv.PolyData]
        List of mesh regions around each junction
    """
    junction_regions = []
    
    for i, junction_point in enumerate(junction_points):
        # Find points within a certain radius of the junction
        distances = np.linalg.norm(mesh.points - junction_point, axis=1)
        
        # Use the minimum radius of vessels at this junction as base radius
        min_radius = np.min(distances[distances > 0])  # Exclude exact matches
        min_radius = junction_radii[i]
        smoothing_radius = min_radius * radius_factor
        
        # Get points within smoothing radius
        region_mask = distances <= smoothing_radius
        region_point_ids = np.where(region_mask)[0]
        
        if len(region_point_ids) > 0:
            # Extract the region around the junction
            region_mesh = mesh.extract_points(region_point_ids)
            junction_regions.append(region_mesh)
    
    return junction_regions


def smooth_junction_region(region_mesh, iterations=5, relaxation_factor=0.1):
    """
    Apply smoothing to a junction region.
    
    Parameters
    ----------
    region_mesh : pv.PolyData
        Mesh region around a junction
    iterations : int
        Number of smoothing iterations
    relaxation_factor : float
        Relaxation factor for smoothing
        
    Returns
    -------
    smoothed_mesh : pv.PolyData
        Smoothed mesh region
    """
    import pdb; pdb.set_trace()
    if region_mesh.n_points < 4:  # Need at least 4 points for meaningful smoothing
        return region_mesh
    
    # Apply Taubin smoothing with boundary preservation
    # smoothed = region_mesh.smooth_taubin(
    #     n_iter= iterations,
    #     pass_band=relaxation_factor,
    #     boundary_smoothing=True,
    #     normalize_coordinates=True
    # )

    smoothed = region_mesh.smooth(n_iter=1000,
    #feature_smoothing=True
    )
    
    return smoothed


def apply_junction_smoothing(mesh, tree_data, vessel_map, connectivity, 
                           smoothing_radius_factor=2.0, smoothing_iterations=30,
                           relaxation_factor=0.1):
    """
    Apply junction smoothing to a vascular mesh.
    
    Parameters
    ----------
    mesh : pv.PolyData
        The surface mesh to smooth
    tree_data : TreeData
        Tree data containing vessel information
    vessel_map : TreeMap
        Vessel connectivity mapping
    connectivity : np.ndarray
        Connectivity array
    smoothing_radius_factor : float
        Factor to determine smoothing radius around junctions
    smoothing_iterations : int
        Number of smoothing iterations
    relaxation_factor : float
        Relaxation factor for smoothing
        
    Returns
    -------
    smoothed_mesh : pv.PolyData
        Mesh with smoothed junctions
    """
    # Detect junctions
    import pdb; pdb.set_trace()
    junctions, junction_radii, junction_points = detect_junctions(tree_data, vessel_map, connectivity)
    
    if len(junction_points) == 0:
        print("No junctions detected for smoothing.")
        return mesh
    
    print(f"Detected {len(junction_points)} junctions for smoothing.")
    
    # Create a copy of the mesh for smoothing
    smoothed_mesh = mesh.copy()
    
    # Identify junction regions
    smoothing_radius_factor = 5
    junction_regions = identify_junction_regions(mesh, junctions, junction_radii, junction_points, smoothing_radius_factor)
    
    # Apply smoothing to each junction region
    for i, region in enumerate(junction_regions):
        if region.n_points > 0:
            # Smooth the region
            smoothed_region = smooth_junction_region(
                region, 
                iterations=smoothing_iterations,
                relaxation_factor=relaxation_factor
            )
            
            # Update the main mesh with smoothed region
            # This is a simplified approach - in practice, you might want to
            # implement a more sophisticated mesh update strategy
            region_point_ids = np.where(
                np.linalg.norm(mesh.points - junction_points[i], axis=1) <= 
                smoothing_radius_factor * np.min(np.linalg.norm(mesh.points - junction_points[i], axis=1)[
                    np.linalg.norm(mesh.points - junction_points[i], axis=1) > 0
                ])
            )[0]
            
            if len(region_point_ids) > 0 and len(smoothed_region.points) > 0:
                # Update points in the main mesh
                smoothed_mesh.points[region_point_ids[:len(smoothed_region.points)]] = smoothed_region.points
    
    # Recompute normals after smoothing
    smoothed_mesh = smoothed_mesh.compute_normals(auto_orient_normals=True)
    
    # Repair any mesh issues
    fix = pymeshfix.MeshFix(smoothed_mesh)
    fix.repair()
    smoothed_mesh = fix.mesh
    
    return smoothed_mesh


def smooth_junctions_advanced(mesh, tree_data, vessel_map, connectivity, 
                            hsize=None, cap_resolution=10):
    """
    Advanced junction smoothing that integrates with the existing mesh processing pipeline.
    
    Parameters
    ----------
    mesh : pv.PolyData
        The surface mesh to smooth
    tree_data : TreeData
        Tree data containing vessel information
    vessel_map : TreeMap
        Vessel connectivity mapping
    connectivity : np.ndarray
        Connectivity array
    hsize : float
        Mesh element size for remeshing
    cap_resolution : int
        Resolution for cap remeshing
        
    Returns
    -------
    smoothed_mesh : pv.PolyData
        Mesh with smoothed junctions
    """
    try:
        # Detect junctions
        junctions, junction_radii, junction_points = detect_junctions(tree_data, vessel_map, connectivity)
        
        if len(junction_points) == 0:
            print("No junctions detected for smoothing.")
            return mesh
    except Exception as e:
        print(f"Warning: Junction detection failed: {e}")
        print("Returning original mesh without smoothing.")
        return mesh
    
    print(f"Detected {len(junction_points)} junctions for smoothing.")
    
    try:
        # Extract faces to identify wall surfaces
        faces, walls, caps, shared_boundaries = extract_faces(mesh, None)
    except Exception as e:
        print(f"Warning: Face extraction failed: {e}")
        print("Returning original mesh without smoothing.")
        return mesh
    
    if len(walls) == 0:
        print("No wall faces found for smoothing.")
        return mesh
    
    # Apply smoothing to wall surfaces
    smoothed_walls = []
    for wall in walls:
        # Apply Taubin smoothing with boundary preservation
        
        smoothed_wall = wall.smooth_taubin(
            n_iter=10,
            pass_band=0.05,
            #boundary_smoothing=True,
            normalize_coordinates=True
        )
        

        # smoothed_wall = wall.smooth(n_iter=100,
        # feature_smoothing=True,
        # boundary_smoothing=True,
        # )
        smoothed_walls.append(smoothed_wall)
    
    # Reconstruct the mesh with smoothed walls

    if len(smoothed_walls) == 1:
        smoothed_mesh = smoothed_walls[0]
        
        junction_regions = identify_junction_regions(smoothed_wall, junctions, junction_radii, junction_points, 2)
        for region in junction_regions:
            smoothed_walls.append(region.extract_surface().subdivide(1,'linear'))
        import pdb; pdb.set_trace()
        # Extract boundaries and remesh caps
        boundaries = smoothed_mesh.extract_feature_edges(
            non_manifold_edges=False, 
            feature_edges=False,
            manifold_edges=False, 
            boundary_edges=True
        )
        boundaries = boundaries.split_bodies()
        
        # Remesh caps
        caps = []
        for i, boundary in enumerate(boundaries):
            if hsize is None:
                hsize = mesh.hsize if hasattr(mesh, 'hsize') else 0.1
            
            try:
                # Convert to PolyData if needed for remeshing
                if hasattr(boundary, 'faces'):
                    # Already PolyData
                    boundary_polydata = boundary
                else:
                    # Convert UnstructuredGrid to PolyData
                    boundary_polydata = boundary.extract_surface()
                
                # Check if the boundary has enough points for remeshing
                if boundary_polydata.n_points < 3:
                    print(f"Warning: Boundary {i} has too few points ({boundary_polydata.n_points}), skipping remeshing")
                    caps.append(boundary_polydata)
                    continue
                
                cap = remesh_surface(boundary_polydata, nosurf=True, hsiz=hsize)
                caps.append(cap)
                
            except Exception as e:
                print(f"Warning: Failed to remesh boundary {i}: {e}")
                print(f"Using original boundary without remeshing")
                # Use the original boundary if remeshing fails
                if hasattr(boundary, 'faces'):
                    caps.append(boundary)
                else:
                    caps.append(boundary.extract_surface())
        
        # Merge smoothed walls and caps
        try:
            caps.insert(0, smoothed_mesh)
            smoothed_mesh = pv.merge(caps)
            smoothed_mesh.hsize = hsize
            import pdb; pdb.set_trace()
            return smoothed_mesh, junction_regions
        except Exception as e:
            print(f"Warning: Failed to merge smoothed mesh: {e}")
            print("Returning original mesh without smoothing.")
            return mesh, junction_regions
    else:
        # For multiple walls, merge them
        smoothed_mesh = pv.merge(smoothed_walls)
        smoothed_mesh.hsize = hsize if hsize is not None else mesh.hsize
        
        return smoothed_mesh, None


def get_junction_statistics(tree_data, vessel_map, connectivity):
    """
    Get statistics about junctions in the tree.
    
    Parameters
    ----------
    tree_data : TreeData
        Tree data containing vessel information
    vessel_map : TreeMap
        Vessel connectivity mapping
    connectivity : np.ndarray
        Connectivity array
        
    Returns
    -------
    stats : Dict
        Dictionary containing junction statistics
    """
    junctions, junction_radii, junction_points = detect_junctions(tree_data, vessel_map, connectivity)
    
    if len(junctions) == 0:
        return {
            'total_junctions': 0,
            'junction_types': {},
            'average_vessels_per_junction': 0
        }
    
    # Count junction types (number of vessels meeting at each junction)
    junction_types = defaultdict(int)
    for vessel_list in junctions.values():
        junction_types[len(vessel_list)] += 1
    
    stats = {
        'total_junctions': len(junctions),
        'junction_types': dict(junction_types),
        'average_vessels_per_junction': np.mean([len(vessels) for vessels in junctions.values()]),
        'junction_coordinates': junction_points
    }
    
    return stats
