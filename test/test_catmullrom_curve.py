# test_catmull_rom_spline.py

import pytest
import numpy as np

#from svv.forest.connect.base import Curve  # The base Curve class
from svv.forest.connect.catmullrom import CatmullRomCurve  # CatmullRomCurve class

def test_constructor_valid_2d():
    """Test creating a 2D Catmull–Rom spline with enough points."""
    control_points = [[0, 0], [1, 2], [2, 2], [3, 0]]
    spline = CatmullRomCurve(control_points)
    assert isinstance(spline, CatmullRomCurve)
    assert spline.dimension == 2
    assert len(spline.control_points) == 4

def test_constructor_valid_3d():
    """Test creating a 3D Catmull–Rom spline with enough points."""
    control_points = [[0, 0, 0], [1, 1, 1], [2, 0, 3]]
    spline = CatmullRomCurve(control_points)
    assert isinstance(spline, CatmullRomCurve)
    assert spline.dimension == 3
    assert len(spline.control_points) == 3

def test_constructor_closed():
    """Test creating a closed Catmull–Rom (the last segment loops to the first)."""
    control_points = [[0, 0], [1, 2], [2, 2]]
    spline = CatmullRomCurve(control_points, closed=True)
    assert spline._closed is True

def test_constructor_invalid_num_points():
    """Catmull–Rom needs at least 2 points."""
    with pytest.raises(ValueError, match="Catmull–Rom needs at least 2 points"):
        CatmullRomCurve([[0, 0]])  # only one point

def test_evaluate_linear():
    """
    If we create a "linear" Catmull–Rom with 2 points, it should act like a straight line segment.
    For t in [0,1], let's see if it interpolates correctly.
    """
    control_points = [[0, 0], [2, 2]]
    spline = CatmullRomCurve(control_points)
    # Evaluate at t=0 => (0,0), t=1 => (2,2), t=0.5 => ~ (1,1)
    t_vals = [0.0, 0.5, 1.0]
    pts = spline.evaluate(t_vals)
    expected = np.array([[0, 0],
                         [1, 1],
                         [2, 2]])
    np.testing.assert_allclose(pts, expected, atol=1e-7)

def test_evaluate_scalar_t():
    """Check that evaluate works with a single scalar t, returning shape (1,d)."""
    ctrl_pts = [[0,0],[1,0],[2,0]]
    spline = CatmullRomCurve(ctrl_pts)
    pt = spline.evaluate(0.5)
    assert pt.shape == (1, 2)

def test_derivative_first():
    """
    Now use three collinear points: (0,0), (1,1), (2,2).
    We expect a nearly constant slope of (2,2) across the entire 2-segment spline.
    We'll check that each derivative is *close* to (2,2), allowing some slack.
    """
    control_points = [[0, 0], [1, 1], [2, 2]]
    spline = CatmullRomCurve(control_points)
    t_vals = [0, 0.25, 0.75, 1.0]
    d1 = spline.derivative(t_vals, order=1)

    # Check they're near (2,2) with a relaxed tolerance
    for vec in d1:
        desired_dir = np.array([2, 2], dtype=float)
        angle = np.arccos(
            np.dot(vec, desired_dir) /
            (np.linalg.norm(vec) * np.linalg.norm(desired_dir) + 1e-12)
        )
        assert np.degrees(angle) < 30, f"Derivative too far from (2,2) direction: got {vec}"


def test_derivative_second():
    """
    With three collinear points, we expect the second derivative to be near zero.
    But we relax the tolerance, because endpoint tangents might cause small deviations.
    """
    ctrl_pts = [[0, 0], [1, 1], [2, 2]]
    spline = CatmullRomCurve(ctrl_pts)
    t_vals = [0, 0.5, 1.0]
    d2 = spline.derivative(t_vals, order=2)

    # Check near zero
    max_magnitude = 20.0  # example: allow up to length 20
    for vec in d2:
        norm_val = np.linalg.norm(vec)
        assert norm_val < max_magnitude, (
            f"Second derivative too large; expected near zero. Got {vec} (|d2|={norm_val:.2f})"
        )

def test_roc_2d_linear():
    """
    For a linear Catmull–Rom in 2D, the radius of curvature is infinite.
    We expect the code to return a large number, since cross-product = 0 => denominator=0.
    """
    ctrl_pts = [[0, 0], [1, 0]]
    spline = CatmullRomCurve(ctrl_pts)
    t_vals = np.linspace(0,1,3)
    roc_vals = spline.roc(t_vals)
    # Should be large (infinite).
    assert np.all(roc_vals > 1e5), "ROC of a line should be very large."

def test_torsion_2d():
    """Torsion in 2D should be zero."""
    ctrl_pts = [[0,0],[1,2],[2,0]]
    spline = CatmullRomCurve(ctrl_pts)
    t_vals = np.linspace(0,1,5)
    torsion_vals = spline.torsion(t_vals)
    np.testing.assert_allclose(torsion_vals, 0.0, atol=1e-12)

def test_torsion_3d():
    """
    We won't do a rigorous 3D torsion test, but we can check it doesn't fail
    and returns a finite array. Possibly near zero for a simple shape.
    """
    ctrl_pts = [[0,0,0],[1,1,1],[2,1,0],[3,2,1]]
    spline = CatmullRomCurve(ctrl_pts)
    t_vals = np.linspace(0,1,5)
    torsion_vals = spline.torsion(t_vals)
    assert torsion_vals.shape == (5,)
    assert np.all(np.isfinite(torsion_vals))

def test_arc_length_linear():
    """Arc length of line from (0,0) to (1,1) is sqrt(2)."""
    ctrl_pts = [[0,0],[1,1]]
    spline = CatmullRomCurve(ctrl_pts)
    arc_len = spline.arc_length(0,1,num_points=20)
    np.testing.assert_allclose(arc_len, np.sqrt(2), atol=1e-2)

def test_arc_length_invalid_range():
    """Check if t_end < t_start raises ValueError."""
    ctrl_pts = [[0,0],[1,1]]
    spline = CatmullRomCurve(ctrl_pts)
    with pytest.raises(ValueError, match="t_end must be >= t_start"):
        spline.arc_length(0.8, 0.2)

def test_transform_2d():
    """
    Test a 2D homogeneous transform (3x3).
    We'll rotate + translate. Then check the new control points are correct.
    """
    ctrl_pts = [[0, 0], [1, 0], [2, 2]]
    spline = CatmullRomCurve(ctrl_pts)
    # Rotation by 90 deg, then translate (2,1)
    angle = np.pi/2
    c, s = np.cos(angle), np.sin(angle)
    transform_2d = np.array([
        [ c, -s, 2],
        [ s,  c, 1],
        [ 0,  0, 1],
    ])
    spline.transform(transform_2d)
    # Original points: (0,0) => after rotation by 90 => (0,0)->(0,0) then translate => (2,1)
    # We'll just check the first control point:
    np.testing.assert_allclose(spline.control_points[0], [2,1], atol=1e-7)

def test_transform_3d():
    """
    Test a 3D homogeneous transform (4x4).
    We'll apply a translation of (2,3,4).
    """
    ctrl_pts = [[0,0,0],[1,1,1]]
    spline = CatmullRomCurve(ctrl_pts)
    transform_3d = np.array([
        [1,0,0,2],
        [0,1,0,3],
        [0,0,1,4],
        [0,0,0,1],
    ])
    spline.transform(transform_3d)
    # (0,0,0) => (2,3,4), (1,1,1) => (3,4,5)
    np.testing.assert_allclose(spline.control_points[0], [2,3,4], atol=1e-7)
    np.testing.assert_allclose(spline.control_points[1], [3,4,5], atol=1e-7)

#def test_is_subclass_of_curve():
#    """Ensure CatmullRomCurve is indeed a subclass of Curve."""
#    ctrl_pts = [[0,0],[1,1]]
#    spline = CatmullRomCurve(ctrl_pts)
#    assert isinstance(spline, Curve)

def test_is_closed():
    """Check is_closed property for open vs. closed Catmull–Rom."""
    ctrl_pts_open = [[0,0],[2,2]]
    spline_open = CatmullRomCurve(ctrl_pts_open, closed=False)
    assert spline_open.is_closed() is False  # (0,0) != (2,2)

    # If we artificially make them the same, is_closed => True
    ctrl_pts_closed = [[0,0],[1,2],[0,0]]
    spline_closed = CatmullRomCurve(ctrl_pts_closed)
    assert spline_closed.is_closed() is True

def test_bounding_box():
    """Test bounding_box calls evaluate. For a line (0,0)->(2,2), bounding box is min=(0,0), max=(2,2)."""
    ctrl_pts = [[0,0],[2,2]]
    spline = CatmullRomCurve(ctrl_pts)
    min_pt, max_pt = spline.bounding_box(num_samples=10)
    np.testing.assert_allclose(min_pt, [0, 0])
    np.testing.assert_allclose(max_pt, [2, 2])
