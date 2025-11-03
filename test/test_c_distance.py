# File: tests/test_minimum_segment_distance.py

import pytest
import numpy as np
from math import sqrt

from svv.utils.spatial.c_distance import minimum_segment_distance

def test_both_degenerate_same_point():
    """
    Both segments are degenerate at the exact same point.
    Distance should be zero.
    """
    data0 = np.array([
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    ], dtype=np.float64)
    data1 = np.array([
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    ], dtype=np.float64)

    dist = minimum_segment_distance(data0, data1)
    assert dist.shape == (1, 1)
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_one_degenerate_one_nondegenerate():
    """
    One segment is a point, the other is a real segment.
    Scenario: point (1,1,1), segment from (2,2,2) to (3,2,2).
    The distance is sqrt((2-1)^2 + (2-1)^2 + (2-1)^2) = sqrt(3)
    if the closest approach is at the segment start.
    """
    data0 = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # degenerate
    ], dtype=np.float64)
    data1 = np.array([
        [2.0, 2.0, 2.0, 3.0, 2.0, 2.0],  # real segment
    ], dtype=np.float64)

    dist = minimum_segment_distance(data0, data1)
    assert dist.shape == (1, 1)
    expected = sqrt(3)
    assert dist[0, 0] == pytest.approx(expected, abs=1e-12)


def test_corner_touch_1d():
    """
    Two collinear segments along x-axis:
    Segment A: [0,0,0 -> 1,0,0]
    Segment B: [1,0,0 -> 2,0,0]

    They share a corner at x=1. The distance is 0 at that shared endpoint.
    """
    data0 = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # A
    ], dtype=np.float64)
    data1 = np.array([
        [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],  # B
    ], dtype=np.float64)

    dist = minimum_segment_distance(data0, data1)
    assert dist.shape == (1, 1)
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_partial_overlap_1d():
    """
    Two collinear segments along x-axis that partially overlap:
    A: [0,0,0 -> 2,0,0]
    B: [1,0,0 -> 3,0,0]

    Overlap region is [1,2] along x.
    The distance is 0, because within that overlap region, points coincide.
    """
    data0 = np.array([
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    ], dtype=np.float64)
    data1 = np.array([
        [1.0, 0.0, 0.0, 3.0, 0.0, 0.0],
    ], dtype=np.float64)

    dist = minimum_segment_distance(data0, data1)
    assert dist.shape == (1, 1)
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_parallel_same_plane_offset():
    """
    Parallel segments in the same plane, offset by 5 units along y-axis:
    A: [0,0,0 -> 1,0,0]
    B: [0,5,0 -> 1,5,0]

    Distance is always 5 (the offset).
    """
    data0 = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float64)
    data1 = np.array([
        [0.0, 5.0, 0.0, 1.0, 5.0, 0.0],
    ], dtype=np.float64)

    dist = minimum_segment_distance(data0, data1)
    assert dist.shape == (1, 1)
    assert dist[0, 0] == pytest.approx(5.0, abs=1e-12)


def test_parallel_different_planes():
    """
    Parallel segments in distinct planes:
    Segment A in plane z=0, Segment B in plane z=2
    Both along x-axis, from x=0..2.

    So the distance is the constant difference in z: 2
    """
    data0 = np.array([
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    ], dtype=np.float64)
    data1 = np.array([
        [0.0, 0.0, 2.0, 2.0, 0.0, 2.0],
    ], dtype=np.float64)

    dist = minimum_segment_distance(data0, data1)
    assert dist.shape == (1, 1)
    assert dist[0, 0] == pytest.approx(2.0, abs=1e-12)


def test_reversed_segments():
    """
    Segments reversed but should yield the same distance as normal orientation.
    A: from (1,1,1) to (0,0,0)
    B: from (2,2,2) to (3,3,3)

    Both diagonals in 3D space. The minimal distance is from (1,1,1) to (2,2,2),
    which is sqrt(3). The orientation reversal should not matter.
    """
    data0 = np.array([
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)
    data1 = np.array([
        [3.0, 3.0, 3.0, 2.0, 2.0, 2.0],
    ], dtype=np.float64)

    dist = minimum_segment_distance(data0, data1)
    assert dist.shape == (1, 1)
    expected = sqrt((2 - 1)**2 + (2 - 1)**2 + (2 - 1)**2)  # sqrt(3)
    assert dist[0, 0] == pytest.approx(expected, abs=1e-12)


def test_line_crossing_at_endpoint():
    """
    A crosses B exactly at B's endpoint:
    A: from (0,0,0) to (2,2,0)
    B: from (2,2,0) to (2,2,2)

    The segments meet at (2,2,0). Minimum distance is 0.
    """
    data0 = np.array([
        [0.0, 0.0, 0.0, 2.0, 2.0, 0.0],
    ], dtype=np.float64)
    data1 = np.array([
        [2.0, 2.0, 0.0, 2.0, 2.0, 2.0],
    ], dtype=np.float64)

    dist = minimum_segment_distance(data0, data1)
    assert dist.shape == (1, 1)
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_multiple_pairs():
    """
    A test with multiple segments on each side, to verify shape correctness and
    distances between each pair.

    We'll carefully choose them so that we can reason about expected distances:
    - data0 has 2 segments: S0, S1
    - data1 has 2 segments: T0, T1

    We'll compute the 2x2 matrix of distances and check each entry.
    """
    data0 = np.array([
        [0,0,0, 0,0,1],    # S0: vertical from (0,0,0) to (0,0,1)
        [0,0,2, 1,0,2],    # S1: horizontal from (0,0,2) to (1,0,2)
    ], dtype=np.float64)

    data1 = np.array([
        [0,1,0, 0,2,0],    # T0: vertical from (0,1,0) to (0,2,0)
        [1,0,2, 2,0,2],    # T1: horizontal from (1,0,2) to (2,0,2)
    ], dtype=np.float64)

    dist = minimum_segment_distance(data0, data1)
    assert dist.shape == (2, 2)

    # dist[0, 0] => S0 vs T0
    #   S0: z from 0 to 1, y=0, x=0
    #   T0: y from 1 to 2, x=0, z=0
    # The closest approach is at S0 endpoint (0,0,0) to T0 endpoint (0,1,0).
    # Distance = 1
    assert dist[0, 0] == pytest.approx(1.0, abs=1e-12)

    # dist[0, 1] => S0 vs T1
    #   S0: (0,0,0)->(0,0,1); T1: (1,0,2)->(2,0,2)
    #   S0 is along z in the plane x=0, y=0
    #   T1 is along x in plane y=0, z=2
    # The closest approach might be from (0,0,1) to (1,0,2).
    # That distance is sqrt((1-0)^2 + (0-0)^2 + (2-1)^2) = sqrt(1 + 1) = sqrt(2)
    # Let's confirm that’s indeed the min.
    assert dist[0, 1] == pytest.approx(sqrt(2), abs=1e-12)

    # dist[1, 0] => S1 vs T0
    #   S1: (0,0,2)->(1,0,2)
    #   T0: (0,1,0)->(0,2,0)
    #   S1 is along x in plane y=0, z=2
    #   T0 is along y in plane x=0, z=0
    # The segments are quite separated in z. The closest approach might be
    # from S1 endpoint (0,0,2) to T0 endpoint (0,1,0)?
    # That’s distance sqrt((0-0)^2 + (1-0)^2 + (0-2)^2) = sqrt(0 +1 +4) = sqrt(5).
    # It's indeed minimal because the other combos are bigger.
    assert dist[1, 0] == pytest.approx(sqrt(5), abs=1e-12)

    # dist[1, 1] => S1 vs T1
    #   S1: (0,0,2)->(1,0,2), T1: (1,0,2)->(2,0,2)
    # They share a corner at (1,0,2). Distance = 0.
    assert dist[1, 1] == pytest.approx(0.0, abs=1e-12)


def test_many_random_small():
    """
    Random test with a small number of segments to ensure
    the function doesn't crash or produce nonsensical results (NaN or negative).

    This won't be an exact correctness check, but we at least confirm that
    distances are non-negative and we can check for outliers.
    """
    rng = np.random.default_rng(42)
    data0 = rng.random((5, 6))
    data1 = rng.random((6, 6))

    dist = minimum_segment_distance(data0, data1)
    assert dist.shape == (5, 6)
    assert np.all(dist >= 0.0)
    # We don't know exact distances, but let's ensure no NaN or inf.
    assert np.all(np.isfinite(dist))


@pytest.mark.parametrize(
    "n0, n1",
    [
        (10, 10),
        (100, 100),
        (500, 500),
    ]
)
def test_many_random_performance(n0, n1):
    """
    Larger random test to check performance and confirm no crashes/NaNs.
    """
    rng = np.random.default_rng(123)
    data0 = rng.random((n0, 6)).astype(np.float64)
    data1 = rng.random((n1, 6)).astype(np.float64)

    dist = minimum_segment_distance(data0, data1)
    assert dist.shape == (n0, n1)
    assert np.all(dist >= 0.0)
    assert np.all(np.isfinite(dist)), "Found NaN/inf values in distance result."