import pytest
import numpy as np
from svv.domain.core.n_matrix import n_matrix


def test_n_matrix_basic_2d():
    """
    Test the N matrix for a simple 2D input.
    """
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    expected_shape = (12, 3)  # (D+1)*N rows, D+1 columns where N=4, D=2
    result = n_matrix(points)

    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Check some specific values in the matrix
    # First row corresponds to [0, 0, 1] (point with a 1 appended)
    assert np.allclose(result[0], [0.0, 0.0, 1.0])
    # Second row corresponds to [1, 0, 1]
    assert np.allclose(result[1], [1.0, 0.0, 1.0])
    # Fifth row (derivative constraint) should have [1, 0, 0]
    assert np.allclose(result[4], [1.0, 0.0, 0.0])


def test_n_matrix_random_3d():
    """
    Test the N matrix for random 3D points.
    """
    np.random.seed(0)
    points = np.random.rand(5, 3)  # 5 points in 3D
    expected_shape = (20, 4)  # (D+1)*N rows, D+1 columns where N=5, D=3
    result = n_matrix(points)

    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Check if the first N rows have correct point values with a 1 appended
    for i in range(points.shape[0]):
        assert np.allclose(result[i, :-1], points[i])
        assert result[i, -1] == 1.0, f"Last column in row {i} should be 1."

    # Check derivative constraints rows
    for i in range(points.shape[0]):
        derivative_block = result[points.shape[0] + i * points.shape[1]:points.shape[0] + (i + 1) * points.shape[1], :3]
        assert np.allclose(derivative_block, np.eye(3)), f"Derivative block for point {i} is incorrect."


def test_n_matrix_edge_case_empty_points():
    """
    Test the N matrix for an empty input.
    """
    points = np.empty((0, 3))  # No points in 3D
    expected_shape = (0, 4)  # No rows, (D+1) columns
    result = n_matrix(points)

    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    assert result.size == 0, "Result should be an empty matrix."


def test_n_matrix_edge_case_single_point():
    """
    Test the N matrix for a single point.
    """
    points = np.array([[1.0, 2.0, 3.0]])  # Single point in 3D
    expected_shape = (4, 4)  # (D+1)*N rows, D+1 columns where N=1, D=3
    result = n_matrix(points)

    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Check the first row corresponds to the point with 1 appended
    assert np.allclose(result[0], [1.0, 2.0, 3.0, 1.0])
    # Check the derivative rows
    derivative_block = result[1:, :3]
    assert np.allclose(derivative_block, np.eye(3)), "Derivative block is incorrect for a single point."


def test_n_matrix_high_dimensional_points():
    """
    Test the N matrix for points in a high-dimensional space (5D).
    """
    points = np.random.rand(3, 5)  # 3 points in 5D
    expected_shape = (18, 6)  # (D+1)*N rows, D+1 columns where N=3, D=5
    result = n_matrix(points)

    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Check the first N rows have correct point values with a 1 appended
    for i in range(points.shape[0]):
        assert np.allclose(result[i, :-1], points[i])
        assert result[i, -1] == 1.0, f"Last column in row {i} should be 1."

    # Check derivative constraints rows
    for i in range(points.shape[0]):
        derivative_block = result[points.shape[0] + i * points.shape[1]:points.shape[0] + (i + 1) * points.shape[1], :5]
        assert np.allclose(derivative_block, np.eye(5)), f"Derivative block for point {i} is incorrect."
