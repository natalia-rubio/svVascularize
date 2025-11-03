import pytest
import numpy as np
from svv.domain.core.m_matrix import m00, m01, m11, m_matrix


def test_m00_basic():
    """
    Test m00 function for basic 2D points.
    """
    points = np.array([[0, 0], [1, 0], [0, 1]])
    rbf_degree = 3
    result = m00(points, rbf_degree)
    expected_shape = (3, 3)  # N x N matrix
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    # Verify diagonal elements (distance to self is zero)
    assert np.allclose(np.diag(result), 0.0), "Diagonal elements of m00 should be zero."


def test_m01_basic():
    """
    Test m01 function for basic 2D points.
    """
    points = np.array([[0, 0], [1, 0], [0, 1]])
    rbf_degree = 3
    result = m01(points, rbf_degree)

    # Verify shape
    expected_shape = (3, 6)  # N x (D*N) matrix
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Verify self-interaction elements are zero
    n, d = points.shape
    for i in range(n):
        for k in range(d):
            assert result[i, i * d + k] == 0.0, f"Element {i, i * d + k} (self-interaction) should be zero."

    # Verify specific interactions between points
    # For i=0, j=1 (point [0, 0] and [1, 0]), M[0, 1D + k] = -p * (x_i[k] - x_j[k]) * ||x_i - x_j||^{p-2}
    dist_01 = np.linalg.norm(points[0] - points[1])  # ||x_i - x_j||
    for k in range(d):
        expected = -rbf_degree * (points[0, k] - points[1, k]) * (dist_01 ** (rbf_degree - 2))
        assert np.isclose(result[0, 1 * d + k], expected), (
            f"Element {0, 1 * d + k} should be {expected}, got {result[0, 1 * d + k]}"
        )


def test_m11_basic():
    """
    Test m11 function for basic 2D points.
    """
    points = np.array([[0, 0], [1, 0], [0, 1]])
    rbf_degree = 3
    result = m11(points, rbf_degree)  # By default, m11 only returns the upper triangular part
    result += result.T

    # Verify shape
    expected_shape = (6, 6)  # (D*N) x (D*N) matrix
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Verify symmetry
    assert np.allclose(result, result.T), "m11 matrix should be symmetric."

    # Verify self-interaction elements
    n, d = points.shape
    for i in range(n):
        for k in range(d):
            idx = i * d + k
            assert result[idx, idx] == 0.0, f"Self-interaction element {idx, idx} should be zero."

    # Verify specific interactions between points
    # For i=0, j=1 (points [0,0] and [1,0])
    dist_01 = np.linalg.norm(points[0] - points[1])  # ||x_i - x_j||
    for k in range(d):
        for l in range(d):
            idx_i = 0 * d + k
            idx_j = 1 * d + l
            if k == l:
                expected = (
                    -2 * (rbf_degree / 2 - 1) * rbf_degree * (points[0, k] - points[1, k]) ** 2 * (dist_01 **
                                                                                                   (rbf_degree - 4))
                    - rbf_degree * (dist_01 ** (rbf_degree - 2))
                )
            else:
                expected = (
                    -2 * (rbf_degree / 2 - 1) * rbf_degree * (points[0, k] - points[1, k]) * (points[0, l] -
                                                                                              points[1, l]) *
                    (dist_01 ** (rbf_degree - 4))
                )
            assert np.isclose(result[idx_i, idx_j], expected), (
                f"Element {idx_i, idx_j} should be {expected}, got {result[idx_i, idx_j]}"
            )


def test_m_matrix_basic():
    """
    Test m_matrix function for basic 2D points.
    """
    points = np.array([[0, 0], [1, 0], [0, 1]])
    rbf_degree = 3
    result = m_matrix(points, rbf_degree)
    n = points.shape[0]
    d = points.shape[1]
    expected_shape = (n * (d + 1), n * (d + 1))  # N*(D+1) x N*(D+1) matrix
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"


def test_m00_edge_case_empty_points():
    """
    Test m00 function for an empty input.
    """
    points = np.empty((0, 2))
    rbf_degree = 3
    result = m00(points, rbf_degree)
    assert result.shape == (0, 0), "m00 with empty input should return an empty matrix."


def test_m01_edge_case_empty_points():
    """
    Test m01 function for an empty input.
    """
    points = np.empty((0, 2))
    rbf_degree = 3
    result = m01(points, rbf_degree)
    assert result.shape == (0, 0), "m01 with empty input should return an empty matrix."


def test_m11_edge_case_empty_points():
    """
    Test m11 function for an empty input.
    """
    points = np.empty((0, 2))
    rbf_degree = 3
    result = m11(points, rbf_degree)
    assert result.shape == (0, 0), "m11 with empty input should return an empty matrix."


def test_m_matrix_edge_case_empty_points():
    """
    Test m_matrix function for an empty input.
    """
    points = np.empty((0, 2))
    rbf_degree = 3
    result = m_matrix(points, rbf_degree)
    assert result.shape == (0, 0), "m_matrix with empty input should return an empty matrix."


def test_m_matrix_high_dimensional_points():
    """
    Test m_matrix function for 3D points.
    """
    points = np.random.rand(4, 3)  # 4 points in 3D
    rbf_degree = 3
    result = m_matrix(points, rbf_degree)
    n = points.shape[0]
    d = points.shape[1]
    expected_shape = (n * (d + 1), n * (d + 1))  # N*(D+1) x N*(D+1) matrix
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"


def test_m_matrix_symmetry():
    """
    Verify symmetry of the full M matrix.
    """
    points = np.random.rand(4, 2)  # 4 points in 2D
    rbf_degree = 3
    result = m_matrix(points, rbf_degree)
    assert np.allclose(result, result.T), "Full M matrix should be symmetric."
