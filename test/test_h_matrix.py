import pytest
import numpy as np
from svv.domain.core.h_matrix import h_matrix


def test_h_matrix_shape():
    """
    Test that the H matrix and its components have the correct shape.
    """
    n, d, lam = 4, 2, 0.1  # 4 points in 2D, regularization parameter
    size = (n + 1) * (d + 1)  # Shape of the interpolation matrix A
    a_ = np.random.rand(size, size)

    h, j00, j01, j11, a_inv = h_matrix(a_, n, d, lam)

    # H matrix shape should be (d * n, d * n)
    assert h.shape == (d * n, d * n), f"Expected H matrix shape {(d * n, d * n)}, got {h.shape}"

    # j00_ should have shape (n, n)
    assert j00.shape == (n, n), f"Expected j00 shape {(n, n)}, got {j00.shape}"

    # j01_ should have shape (n, d * n)
    assert j01.shape == (n, d * n), f"Expected j01 shape {(n, d * n)}, got {j01.shape}"

    # j11_ should have shape (d * n, d * n)
    assert j11.shape == (d * n, d * n), f"Expected j11 shape {(d * n, d * n)}, got {j11.shape}"

    # a_inv should have the same shape as a_
    assert a_inv.shape == a_.shape, f"Expected a_inv shape {a_.shape}, got {a_inv.shape}"

def test_h_matrix_no_regularization():
    """
    Test H matrix computation when lam = 0 (no regularization).
    """
    n, d, lam = 3, 3, 0.0  # 3 points in 3D, no regularization
    size = (n + 1) * (d + 1)
    a_ = np.random.rand(size, size)
    h, j00, j01, j11, a_inv = h_matrix(a_, n, d, lam)

    # H matrix should equal j11_ when lam = 0
    assert np.allclose(h, j11), "H matrix should equal j11 when lam = 0."


def test_h_matrix_with_regularization():
    """
    Test H matrix computation when lam > 0 (with regularization).
    """
    n, d, lam = 3, 2, 0.1  # 3 points in 2D, regularization parameter
    size = (n + 1) * (d + 1)
    a_ = np.random.rand(size, size)
    h, j00, j01, j11, a_inv = h_matrix(a_, n, d, lam)

    # Compute the expected H matrix with regularization
    j00_inv = np.linalg.inv(np.eye(j00.shape[0]) + lam * j00)
    expected_h = j11 - lam * j01.T @ j00_inv @ j01

    assert np.allclose(h, expected_h), "H matrix computation with regularization is incorrect."


def test_h_matrix_a_inv_consistency():
    """
    Test that the computed inverse (a_inv) is consistent with the original matrix (a_).
    """
    n, d, lam = 3, 2, 0.1  # 3 points in 2D, regularization parameter
    size = (n + 1) * (d + 1)
    a_ = np.random.rand(size, size)

    _, _, _, _, a_inv = h_matrix(a_, n, d, lam)

    # Verify that A @ A_inv is approximately the identity matrix
    identity = np.eye(a_.shape[0])
    assert np.allclose(a_ @ a_inv, identity), "a_inv is not the correct inverse of a_."


def test_h_matrix_positive_semidefinite():
    """
    Test that the computed H matrix is positive semi-definite.
    """
    n, d, lam = 3, 2, 0.1  # 3 points in 2D, regularization parameter
    size = (n + 1) * (d + 1)
    a_ = np.random.rand(size, size)
    a_ = a_ @ a_.T  # Make a_ symmetric positive definite

    h, _, _, _, _ = h_matrix(a_, n, d, lam)

    # Eigenvalues of H should be non-negative
    eigenvalues = np.linalg.eigvalsh(h)
    assert np.all(eigenvalues >= 0), "H matrix is not positive semi-definite."


def test_h_matrix_edge_case():
    """
    Test edge case when n or d is zero.
    """
    a_ = np.array([[1]])  # Minimal valid A matrix
    h, j00, j01, j11, a_inv = h_matrix(a_, n=0, d=1, lam=0.1)

    # H matrix should be empty
    assert h.size == 0, "H matrix should be empty when n=0."

    # j00_, j01_, j11_, and a_inv should have appropriate shapes
    assert j00.size == 0, "j00 should be empty when n=0."
    assert j01.size == 0, "j01 should be empty when n=0."
    assert j11.size == 0, "j11 should be empty when n=0."
    assert a_inv.shape == (1, 1), f"a_inv shape should be (1, 1), got {a_inv.shape}."
