import pytest
import numpy as np

# Import all the functions from your compiled module
# e.g. if the compiled module is called my_cython_module, do:
from svv.domain.routines.c_allocate import (
    norm,
    argwhere_nonzeros,
    argwhere_value_double,
    argwhere_value_int,
    any_value_double,
    any_value_int,
    c_dict,
    duplicate_map,
    _allocate_patch,
    closest_point,
    get_angle,
    _allocate_angle
)


def test_norm():
    # Test the norm function with a simple 2D array
    data = np.array([[3.0, 4.0], [1.0, 2.0]], dtype=np.float64)
    result = norm(data)
    # Expected: sqrt(3^2 + 4^2) = 5, sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.23607
    expected = np.array([[5.0], [np.sqrt(5.0)]])
    assert np.allclose(result, expected), f"norm({data}) != expected"


def test_argwhere_nonzeros():
    data = np.array([0.0, 1.0, 0.0, 2.0, 3.0], dtype=np.float64)
    result = argwhere_nonzeros(data)
    # Nonzero elements are at indices 1, 3, and 4
    expected = np.array([1, 3, 4], dtype=np.int64)
    assert np.array_equal(result, expected), f"argwhere_nonzeros({data}) != {expected}"


def test_argwhere_value_double():
    data = np.array([1.0, 2.0, 2.0, 4.0], dtype=np.float64)
    result = argwhere_value_double(data, 2.0)
    # The value 2.0 occurs at indices 1 and 2
    expected = np.array([1, 2], dtype=np.int64)
    assert np.array_equal(result, expected), f"argwhere_value_double({data}, 2.0) != {expected}"


def test_argwhere_value_int():
    data = np.array([10, 20, 20, 30, 40], dtype=np.int64)
    result = argwhere_value_int(data, 20)
    # The value 20 occurs at indices 1 and 2
    expected = np.array([1, 2], dtype=np.int64)
    assert np.array_equal(result, expected), f"argwhere_value_int({data}, 20) != {expected}"


def test_any_value_double():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert any_value_double(data, 2.0) is True, "any_value_double should have found 2.0"
    assert any_value_double(data, 5.0) is False, "any_value_double should NOT have found 5.0"


def test_any_value_int():
    data = np.array([1, 2, 3], dtype=np.int32)
    assert any_value_int(data, 3) is True, "any_value_int should have found 3"
    assert any_value_int(data, 4) is False, "any_value_int should NOT have found 4"


def test_c_dict():
    # Test building a "dictionary" from a 2D array
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = c_dict(data)
    # Depending on how it is returned, if Cython automatically converts it:
    #   Expect something like {0: [1.0, 2.0], 1: [3.0, 4.0]}
    # If it’s a normal Python dict, test likewise:
    #   result might be a dict with integer keys and list-of-floats as values.
    assert isinstance(result, dict), "Returned object from c_dict should be a Python dict"
    assert 0 in result and 1 in result, "Keys 0 and 1 should be in the dictionary"
    assert result[0] == [1.0, 2.0], "Dictionary key 0 should map to [1.0, 2.0]"
    assert result[1] == [3.0, 4.0], "Dictionary key 1 should map to [3.0, 4.0]"


def test_duplicate_map():
    # Example: unique_inverse tells which "cluster index" each element belongs to,
    # and unique_counts is how many elements belong to each cluster index.
    unique_inverse = np.array([0, 1, 1, 2, 2, 2], dtype=np.int64)  # 6 elements
    # unique_counts might be:
    #   cluster 0 has 1 element (index 0),
    #   cluster 1 has 2 elements (indices 1,2),
    #   cluster 2 has 3 elements (indices 3,4,5).
    unique_counts = np.array([1, 2, 3], dtype=np.int64)
    duplicate_dict, duplicate_set = duplicate_map(unique_inverse, unique_counts)

    # Because cluster 0 has only 1 element, it should be skipped.
    # cluster 1 has 2 elements => indices [1,2]
    # cluster 2 has 3 elements => indices [3,4,5]
    # So we expect:
    #   duplicate_dict[1] = [1,2]
    #   duplicate_dict[2] = [3,4,5]
    #   duplicate_set = {1,2,3,4,5}
    assert isinstance(duplicate_dict, dict), "duplicate_map should return a Python dict"
    assert len(duplicate_dict[1]) == 2, "Cluster 1 has 2 duplicates"
    assert len(duplicate_dict[2]) == 3, "Cluster 2 has 3 duplicates"
    assert duplicate_set == {1, 2, 3, 4, 5}, "All duplicates should be in the set"


def test_allocate_patch():
    # Indices array
    indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    overlap = 0.5  # 50% overlap
    point_set = {0, 1, 2, 3, 4}  # All points present initially
    duplicates_set = {2}  # Suppose 2 is a 'duplicate' to preserve, or so

    # _allocate_patch modifies point_set based on overlap
    # overlap_indicies = int(5 * 0.5) => 2
    # for i in range(1, 2): => i = 1
    #   if point_set.count(indices[1]) ...
    # This example is trivial, but let's just run it
    updated = _allocate_patch(indices, overlap, point_set, duplicates_set)
    # Check which points remain
    # Because we only remove indices[1] if it's not in duplicates_set. It's not (duplicates_set = {2}),
    # so we remove `1` from the set.
    expected = {0, 2, 3, 4}
    assert updated == expected, f"Expected {expected}, got {updated}"


def test_closest_point():
    # We have points in 2D
    points = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0],
        [2.0, 2.0],
    ], dtype=np.float64)
    included = {1, 2, 3}
    # We want the closest point to index=0 among those in `included`
    # Distances:
    #   to 1 => sqrt((0-1)^2 + (0-1)^2) = sqrt(1+1)= sqrt(2)
    #   to 2 => sqrt((0-5)^2 + (0-5)^2) = sqrt(25+25)= sqrt(50)
    #   to 3 => sqrt((0-2)^2 + (0-2)^2) = sqrt(4+4)= sqrt(8)
    # The minimum distance is sqrt(2), i.e. index=1
    index = 0
    c_idx, c_dist = closest_point(index, included, points)
    assert c_idx == 1, f"Expected closest index=1, got {c_idx}"
    # sqrt(2) ~ 1.41421
    assert abs(c_dist - np.sqrt(2)) < 1e-7, f"Expected distance ~1.414, got {c_dist}"


def test_get_angle():
    # normals shape = (N, 3) or (N, 2). Let’s do 3D for example
    normals = np.array([
        [1.0, 0.0, 0.0],  # index_0
        [0.0, 1.0, 0.0],  # index_1
    ], dtype=np.float64)
    # The angle between the x-axis and y-axis vectors is 90 degrees
    angle = get_angle(0, 1, normals)
    assert abs(angle - 90.0) < 1e-7, f"Expected angle=90, got {angle}"


def test_allocate_angle():
    """
    This function is more complex, but let's do a small contrived example
    to ensure it runs and returns something of the correct type.
    """
    idx = 1
    indices = np.array([1, 1, 2, 3], dtype=np.int64)
    points = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [2.0, 2.0],
        [3.0, 3.0],
    ], dtype=np.float64)
    normals = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=np.float64)
    feature_angle = 30.0

    # Just run _allocate_angle to see if we get a vector-like return
    allocated = _allocate_angle(idx, indices, points, normals, feature_angle)
    # allocated should be a list of cluster indices that matched the angle condition
    # You can adapt this test once you know the intended logic.
    assert isinstance(allocated, list), "Expected a Python list from _allocate_angle"
    # At minimum, 'idx' (1) should show up in allocated
    assert 1 in allocated, "At least one entry with '1' expected in allocated indices"
    # You can add more precise checks based on the geometry and angle logic.
