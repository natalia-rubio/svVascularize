import pytest
import numpy as np

# If your compiled module is named something else, replace the import accordingly:
from svv.domain.routines.c_sample import (
    pick_from_tetrahedron,
    pick_from_triangle,
    pick_from_line
)

def test_pick_from_line():
    """
    A line can be defined by two 3D points. For example:
        point A = [0, 0, 0]
        point B = [1, 1, 1]
    If the barycentric coordinate for B is x, then for A is (1 - x).
    The resulting point is (1 - x)*A + x*B = [x, x, x].
    """
    # We'll test a single sample (i=1)
    simplices = np.array([
        [[0.0, 0.0, 0.0],  # Corner A
         [1.0, 1.0, 1.0]]  # Corner B
    ], dtype=np.float64).reshape(1, 2, 3)

    # rdx is shape (i, 2, 1). We'll choose x=0.2, so rdx[0,1,0] = 0.2, rdx[0,0,0] is ignored initially
    rdx = np.array([
        [[0.0],
         [0.2]]
    ], dtype=np.float64).reshape(1, 2, 1)

    # Expected result => [0.2, 0.2, 0.2]
    result = pick_from_line(simplices, rdx)
    expected = np.array([[0.2, 0.2, 0.2]], dtype=np.float64)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_pick_from_triangle():
    """
    A triangle can be defined by 3 points in 3D. For example:
        A = [0, 0, 0]
        B = [1, 0, 0]
        C = [0, 1, 0]
    We'll pick barycentric coordinates (r2, r3). The first coordinate is r1 = 1 - r2 - r3.
    The resulting point is r1*A + r2*B + r3*C.
    """
    simplices = np.array([
        [[0.0, 0.0, 0.0],  # A
         [1.0, 0.0, 0.0],  # B
         [0.0, 1.0, 0.0]]  # C
    ], dtype=np.float64).reshape(1, 3, 3)

    # rdx shape (1, 3, 1).
    # Suppose r2=0.2, r3=0.3 => r1=0.5 => final point = [0.2, 0.3, 0.0]
    rdx = np.array([
        [[0.0],
         [0.2],
         [0.3]]
    ], dtype=np.float64).reshape(1, 3, 1)

    # Expected => [0.2, 0.3, 0.0]
    result = pick_from_triangle(simplices, rdx)
    expected = np.array([[0.2, 0.3, 0.0]], dtype=np.float64)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_pick_from_tetrahedron():
    """
    A tetrahedron can be defined by 4 points in 3D. For example:
        A = [0, 0, 0]
        B = [1, 0, 0]
        C = [0, 1, 0]
        D = [0, 0, 1]
    We'll specify barycentric coords (r2, r3, r4). The function
    modifies them if sums exceed 1, but let's choose a case that
    doesn't trigger many conditionals, for clarity.
    Suppose r2=0.2, r3=0.2, r4=0.3 => r1=0.3 => final point = [0.2, 0.2, 0.3].
    """
    simplices = np.array([[
        [0.0, 0.0, 0.0],  # A
        [1.0, 0.0, 0.0],  # B
        [0.0, 1.0, 0.0],  # C
        [0.0, 0.0, 1.0]   # D
    ]], dtype=np.float64)  # shape (1, 4, 3)

    # shape (1, 4, 1). We'll manually set r2=0.2, r3=0.2, r4=0.3 => r1=0
    # The function itself sets r1 = 1 - sum(r2..r4), so we can initialize r1=0.0 for convenience.
    rdx = np.array([
        [[0.0],
         [0.2],
         [0.2],
         [0.3]]
    ], dtype=np.float64).reshape(1, 4, 1)

    result = pick_from_tetrahedron(simplices, rdx)
    # We expect the final barycentric coords to remain r1=0.3, r2=0.2, r3=0.2, r4=0.3 =>
    # => [0.2, 0.2, 0.3]
    expected = np.array([[0.2, 0.2, 0.3]], dtype=np.float64)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
