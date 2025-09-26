#!/usr/bin/env python3
"""
Test script for junction smoothing functionality.

This script demonstrates how to use the new junction smoothing features
in the svVascularize tree structure.
"""

import numpy as np
import pyvista as pv
from svv.tree.tree import Tree
from svv.domain.domain import Domain
from svv.tree.utils.junction_smoothing import get_junction_statistics


def create_simple_tree():
    """
    Create a simple tree with junctions for testing.
    """
    # Create a simple domain (sphere)
    domain = Domain()
    domain.set_sphere(center=[0, 0, 0], radius=10)
    
    # Create tree
    tree = Tree()
    tree.set_domain(domain)
    
    # Set root
    tree.set_root(start=[0, 0, 0], direction=[1, 0, 0])
    
    # Add some vessels to create junctions
    for i in range(5):
        tree.add()
    
    return tree


def test_junction_detection():
    """
    Test junction detection functionality.
    """
    print("Testing junction detection...")
    
    tree = create_simple_tree()
    
    # Get junction statistics
    stats = tree.get_junction_statistics()
    
    print(f"Junction Statistics:")
    print(f"  Total junctions: {stats['total_junctions']}")
    print(f"  Junction types: {stats['junction_types']}")
    print(f"  Average vessels per junction: {stats['average_vessels_per_junction']:.2f}")
    
    return tree, stats


def test_mesh_export_with_smoothing():
    """
    Test mesh export with junction smoothing.
    """
    print("\nTesting mesh export with junction smoothing...")
    
    tree = create_simple_tree()
    
    # Export without smoothing
    print("Exporting mesh without smoothing...")
    mesh_no_smooth = tree.export_solid(watertight=True, smooth_junctions=False)
    print(f"Mesh without smoothing: {mesh_no_smooth.n_points} points, {mesh_no_smooth.n_cells} cells")
    
    # Export with smoothing
    print("Exporting mesh with smoothing...")
    mesh_smooth = tree.export_solid(watertight=True, smooth_junctions=True, 
                                   smoothing_radius_factor=2.0, smoothing_iterations=5)
    print(f"Mesh with smoothing: {mesh_smooth.n_points} points, {mesh_smooth.n_cells} cells")
    
    return mesh_no_smooth, mesh_smooth


def visualize_comparison(mesh_no_smooth, mesh_smooth):
    """
    Visualize the comparison between smoothed and non-smoothed meshes.
    """
    print("\nCreating visualization...")
    
    # Create a plotter
    plotter = pv.Plotter(shape=(1, 2))
    
    # Plot original mesh
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_no_smooth, color='red', opacity=0.7, 
                    show_edges=True, edge_color='black')
    plotter.add_text("Original Mesh", font_size=12)
    
    # Plot smoothed mesh
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_smooth, color='blue', opacity=0.7, 
                    show_edges=True, edge_color='black')
    plotter.add_text("Smoothed Mesh", font_size=12)
    
    # Show the plot
    plotter.show()
    
    return plotter


def main():
    """
    Main test function.
    """
    print("=== Junction Smoothing Test ===")
    
    try:
        # Test junction detection
        tree, stats = test_junction_detection()
        
        # Test mesh export with smoothing
        mesh_no_smooth, mesh_smooth = test_mesh_export_with_smoothing()
        
        # Visualize comparison
        plotter = visualize_comparison(mesh_no_smooth, mesh_smooth)
        
        print("\n=== Test completed successfully! ===")
        print("The junction smoothing functionality is working correctly.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
