import pytest
import numpy as np
import pyvista as pv

from svv.domain.routines.boolean import (
    convert_to_trimesh,
    convert_to_pyvista,
    boolean
)


###############################################################################
# Fixtures for test shapes
###############################################################################

@pytest.fixture
def cube_pv():
    """
    A unit cube from (0,0,0) to (1,1,1).
    Triangulate so that every face is made of two triangles.
    """
    box = pv.Box(bounds=(0, 1, 0, 1, 0, 1))
    return box.triangulate()

@pytest.fixture
def sphere_pv():
    """
    A sphere centered at (0.5, 0.5, 0.5) with radius 0.5.
    Triangulate for safety (it should already be triangular, but just to be sure).
    """
    sph = pv.Sphere(radius=0.5, center=(0.5, 0.5, 0.5), phi_resolution=24, theta_resolution=24)
    return sph.triangulate()

@pytest.fixture
def big_sphere_pv():
    """
    A sphere that fully encompasses the cube: center (0.5,0.5,0.5) and radius=2
    """
    sph = pv.Sphere(radius=2.0, center=(0.5, 0.5, 0.5), phi_resolution=24, theta_resolution=24)
    return sph.triangulate()

@pytest.fixture
def shifted_cube_pv():
    """
    A cube that only partially intersects with the standard unit cube
    by shifting it by 0.5 in x, y, z.
    """
    box = pv.Box(bounds=(0.5, 1.5, 0.5, 1.5, 0.5, 1.5))
    return box.triangulate()

###############################################################################
# Tests for conversion functions
###############################################################################

def test_convert_to_trimesh_triangle():
    """
    Test converting a minimal triangular PyVista mesh to a Trimesh.
    """
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])
    faces = np.hstack([[3], [0, 1, 2]])  # single triangle
    pv_mesh = pv.PolyData(points, faces, force_float=False).triangulate()
    tm_mesh = convert_to_trimesh(pv_mesh)
    assert tm_mesh.faces.shape[0] == 1, "Should have exactly 1 face (triangle)."


def test_convert_to_trimesh_quads():
    points = np.array([
        [0,0,0],
        [1,0,0],
        [1,1,0],
        [0,1,0]
    ], dtype=float)
    faces = np.hstack([[4],[0,1,2,3]])  # single quad
    pv_mesh = pv.PolyData(points, faces)
    # Triangulate here:
    pv_mesh = pv_mesh.triangulate()
    # Now this should be 2 triangles => 8 ints in faces
    tm_mesh = convert_to_trimesh(pv_mesh)
    # Should pass, as code can do the standard reshape
    assert tm_mesh.faces.shape[0] == 2, "Quad should become 2 triangles after triangulation."

###############################################################################
# Tests for boolean() function
###############################################################################

def test_boolean_union(cube_pv, sphere_pv):
    """
    Union of a unit cube and a sphere (radius=0.5 inside it).
    The sphere is centered at (0.5,0.5,0.5) so it partially intersects.
    """
    result = boolean(cube_pv, sphere_pv, operation='union')
    assert result.n_cells > 0, "Result should have faces."
    assert result.n_points > 0, "Result should have vertices."
    assert result.is_manifold, "Union result should be manifold if fix_mesh is True."

    # Volume check
    vol_cube = cube_pv.volume
    vol_sphere = sphere_pv.volume
    vol_union = result.volume
    if vol_cube >= vol_sphere:
        test_vol = vol_cube
        test_geo = 'cube'
    else:
        test_vol = vol_sphere
        test_geo = 'sphere'
    # The union should be bigger than or equal to the larger of the two
    assert vol_union >= vol_cube, "Union volume should at least be the {}'s volume.".format(test_geo)
    # The union should be bigger than or equal to the larger of the two
    assert vol_union >= vol_sphere, "Union volume should at least be the {}'s volume.".format(test_geo)
    if not np.isclose(vol_union, test_vol, atol=1e-7):
        # If it's not equal within tolerance, then we expect it to exceed the cube volume
        assert vol_union > test_vol



def test_boolean_intersection(cube_pv, sphere_pv):
    """
    Intersection of a unit cube and a small sphere (radius=0.5).
    The sphere is inside the cube, so intersection ~ the sphere region that
    lies within the cube's boundaries. Should be roughly the sphere volume.
    """
    result = boolean(cube_pv, sphere_pv, operation='intersection')
    assert result.n_cells > 0, "Result should have faces."
    assert result.n_points > 0, "Result should have vertices."
    assert result.is_manifold, "Intersection result should be manifold."

    vol_cube = cube_pv.volume
    vol_sphere = sphere_pv.volume
    vol_intersect = result.volume
    # Because the sphere is entirely inside the cube (since radius=0.5, center at (0.5,0.5,0.5)),
    # the intersection volume should be about the sphere's volume.
    # But to allow for triangulation or mesh approximations, we allow some tolerance
    assert abs(vol_intersect - vol_sphere) / vol_sphere < 0.05, (
        "Intersection volume should be close to the sphere's volume."
    )


def test_boolean_difference(cube_pv, sphere_pv):
    """
    The difference (cube - sphere).
    Because the sphere is fully inside, the difference is the cube shell minus
    that smaller sphere region in the center.
    """
    result = boolean(cube_pv, sphere_pv, operation='difference')
    assert result.n_cells > 0, "Result should have faces."
    assert result.n_points > 0, "Result should have vertices."
    assert result.is_manifold, "Difference result should be manifold."

    vol_cube = cube_pv.volume
    vol_sphere = sphere_pv.volume
    vol_diff = result.volume
    # We removed the inside part from the cube
    # So the difference volume should be about (cube - sphere).
    expected = vol_cube - vol_sphere
    assert abs(vol_diff - expected) / vol_cube < 0.05, (
        "Difference volume should be close to cube - sphere volume."
    )


def test_boolean_difference_outside(cube_pv, big_sphere_pv):
    """
    If we do (cube - big_sphere), and the sphere is much bigger than the cube,
    the difference is effectively empty (the sphere covers the entire cube).
    """
    result = boolean(cube_pv, big_sphere_pv, operation='difference')
    # Possibly the result might be entirely empty. Some boolean engines produce an empty mesh.
    # pyvista might handle an "empty mesh" in different ways (0 faces, 0 points).
    # Let’s see if the code returns a valid but empty PolyData or raises an error.
    assert result.n_cells == 0 or result.n_points == 0, (
        "Expected empty or near-empty result if the sphere fully envelops the cube."
    )
    # If the engine tries to fix the mesh, it might produce a degenerate polydata with no volume
    assert result.volume == 0.0, "Result volume should be zero (fully subtracted)."


def test_boolean_shared_edge(cube_pv, shifted_cube_pv):
    """
    The union of two cubes that share only a partial overlap (shifted by 0.5).
    We can check volumes, bounding box, and face counts more explicitly.
    """
    # They intersect from [0.5..1.0] in each axis, so the overlap region is a sub-cube of size 0.5^3=0.125
    # So total union volume = 1.0 (cube) + 1.0 (shifted cube) - 0.125 (overlap) = 1.875
    result_union = boolean(cube_pv, shifted_cube_pv, operation='union')
    vol_union = result_union.volume
    assert abs(vol_union - 1.875) < 0.05, (
        f"Union volume ~1.875, got {vol_union}"
    )

    # Intersection volume should be ~0.125
    result_intersect = boolean(cube_pv, shifted_cube_pv, operation='intersection')
    vol_intersect = result_intersect.volume
    assert abs(vol_intersect - 0.125) < 0.01, (
        f"Intersection volume ~0.125, got {vol_intersect}"
    )

    # difference (cube - shifted_cube) => volume should be 1.0 - 0.125 = 0.875
    result_diff = boolean(cube_pv, shifted_cube_pv, operation='difference')
    vol_diff = result_diff.volume
    assert abs(vol_diff - 0.875) < 0.05, (
        f"Difference volume ~0.875, got {vol_diff}"
    )


def test_boolean_touching_corner():
    """
    Create two cubes that only share one corner (1,1,1)
    and test intersection is basically that point (which is degenerate).
    """
    cube1 = pv.Box(bounds=(0,1,0,1,0,1)).triangulate()
    # Shift so the second cube goes from (1,2,1,2,1,2),
    # so the only "overlap" is that corner at (1,1,1).
    cube2 = pv.Box(bounds=(1,2,1,2,1,2)).triangulate()

    inter = boolean(cube1, cube2, operation='intersection')
    # Possibly the intersection is an empty/degenerate mesh or a single vertex.
    # Some boolean engines might treat a corner contact as no intersection volume.
    # Let's see if it's empty.
    assert inter.n_cells == 0 or inter.n_points <= 1, (
        "Touching corners produce no real intersection volume."
    )
    # Union should have volume=2
    union = boolean(cube1, cube2, operation='union')
    vol_union = union.volume
    assert abs(vol_union - 2.0) < 0.05, "Two unit cubes union => volume ~2.0 if they only share a corner."


def test_boolean_invalid_operation(cube_pv, sphere_pv):
    """
    Test that an invalid operation raises a ValueError.
    """
    with pytest.raises(ValueError):
        boolean(cube_pv, sphere_pv, operation='bogus_op')
