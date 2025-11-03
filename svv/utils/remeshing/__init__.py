"""
Remeshing utility based on MMG executables.

This module provides functions to remesh surfaces and volumes using MMG, a software for remeshing.

Functions
---------
- remesh_surface_2d : Remeshes 2D surface boundaries using MMG2D.
- remesh_surface : Remeshes 3D surfaces using MMGS.
- remesh_volume : Remeshes 3D volumes using MMG3D.
- add_required : Adds required triangles to a .mesh file.
- clean_medit : Cleans up a .mesh file by removing unsupported keywords.
"""