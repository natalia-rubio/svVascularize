# test_nurbs_curve.py

import pytest
import numpy as np

#from svv.forest.connect.base import Curve # the base curve class
from svv.forest.connect.nurbs import NURBSCurve # The nurbs curve class


def test_constructor_valid_linear_2d():
    """
    Test constructing a 2D linear NURBS curve with 2 control points (straight line).
    For degree=1, we need knot_vector of length = N + degree + 1 = 2 + 1 + 1 = 4.
    """
    control_points = [[0, 0], [2, 2]]
    weights = [1, 1]  # each control point has weight=1
    knot_vector = [0, 0, 1, 1]  # typical 'clamped' form for linear
    degree = 1

    curve = NURBSCurve(control_points, weights, knot_vector, degree)
    assert isinstance(curve, NURBSCurve)
    assert curve.dimension == 2
    assert len(curve.control_points) == 2
    assert len(curve.weights) == 2
    assert len(curve.knot_vector) == 4


def test_constructor_invalid_weight_length():
    """
    control_points and weights must have same length.
    """
    control_points = [[0, 0], [1, 1]]
    weights = [1]  # mismatched length
    knot_vector = [0, 0, 1, 1]
    degree = 1

    with pytest.raises(ValueError, match="must have the same length"):
        NURBSCurve(control_points, weights, knot_vector, degree)


def test_constructor_invalid_knot_length():
    """
    For N=3 control points, degree=2 => required knot length = N+degree+1 = 6,
    so giving anything else should raise an error.
    """
    control_points = [[0, 0], [1, 1], [2, 2]]
    weights = [1, 1, 1]
    # This is the wrong length (should be 3+2+1=6). Let's use 5 instead:
    knot_vector = [0, 0, 0, 1, 1]
    degree = 2

    with pytest.raises(ValueError, match="knot_vector length must be"):
        NURBSCurve(control_points, weights, knot_vector, degree)


def test_constructor_non_decreasing_knot():
    """
    The knot vector must be non-decreasing.
    """
    control_points = [[0, 0], [1, 1], [2, 2]]
    weights = [1, 1, 1]
    # This has a decreasing portion: [0, 0, 1, 0, 1, 2]
    knot_vector = [0, 0, 1, 0, 1, 2]
    degree = 2

    with pytest.raises(ValueError, match="must be non-decreasing"):
        NURBSCurve(control_points, weights, knot_vector, degree)


def test_linear_nurbs_evaluate():
    """
    For a linear NURBS curve (degree=1) with 2 points, weights=1 => it's a straight line.
    We check that evaluate(t) is a linear interpolation between (0,0) and (2,2).
    """
    control_points = [[0, 0], [2, 2]]
    weights = [1, 1]
    knot_vector = [0, 0, 1, 1]
    degree = 1

    curve = NURBSCurve(control_points, weights, knot_vector, degree)

    # Evaluate at t=0 => (0,0), t=1 => (2,2), t=0.5 => (1,1)
    t_values = [0.0, 0.5, 1.0]
    pts = curve.evaluate(t_values)
    expected = np.array([[0, 0],
                         [1, 1],
                         [2, 2]])
    np.testing.assert_allclose(pts, expected, atol=1e-8)


def test_evaluate_scalar_t():
    """
    Check that passing a single scalar t also works and returns shape (1, dimension).
    """
    control_points = [[0, 0], [1, 0]]
    weights = [1, 1]
    knot_vector = [0, 0, 1, 1]
    degree = 1

    curve = NURBSCurve(control_points, weights, knot_vector, degree)
    point = curve.evaluate(0.5)
    # Should return shape (1,2), midpoint between (0,0) and (1,0) => (0.5, 0).
    assert point.shape == (1, 2)
    np.testing.assert_allclose(point, [[0.5, 0.0]], atol=1e-8)


def test_dimension_property():
    """
    Confirm dimension matches the second axis of control_points.
    """
    control_points_2d = [[0, 0], [1, 1], [2, 2]]
    weights_2d = [1, 2, 1]
    knot_vector_2d = [0, 0, 0, 1, 1, 1]
    curve_2d = NURBSCurve(control_points_2d, weights_2d, knot_vector_2d, degree=2)
    assert curve_2d.dimension == 2

    control_points_3d = [[0, 0, 0], [1, 1, 1], [2, 0, 2]]
    weights_3d = [1, 1, 1]
    knot_vector_3d = [0, 0, 0, 1, 1, 1]
    curve_3d = NURBSCurve(control_points_3d, weights_3d, knot_vector_3d, degree=2)
    assert curve_3d.dimension == 3


#def test_is_subclass_of_curve():
#    """
#    Check that NURBSCurve is indeed a subclass of Curve.
#    """
#    control_points = [[0, 0], [2, 2]]
#    weights = [1, 1]
#    knot_vector = [0, 0, 1, 1]
#    degree = 1
#    curve = NURBSCurve(control_points, weights, knot_vector, degree)
#    assert isinstance(curve, Curve)


def test_not_implemented_methods():
    """
    derivative, roc, torsion, arc_length, transform should raise NotImplementedError
    in the current version of NURBSCurve.
    """
    control_points = [[0, 0], [2, 2]]
    weights = [1, 1]
    knot_vector = [0, 0, 1, 1]
    degree = 1
    curve = NURBSCurve(control_points, weights, knot_vector, degree)

    t_vals = [0.0, 0.5, 1.0]
    with pytest.raises(NotImplementedError):
        curve.derivative(t_vals)
    with pytest.raises(NotImplementedError):
        curve.roc(t_vals)
    with pytest.raises(NotImplementedError):
        curve.torsion(t_vals)
    with pytest.raises(NotImplementedError):
        curve.arc_length(0.0, 1.0)
    with pytest.raises(NotImplementedError):
        curve.transform(np.eye(3))


def test_bounding_box_linear():
    """
    bounding_box calls evaluate internally.
    For the line from (0,0) to (2,2), the bounding box should be:
      min => (0,0), max => (2,2).
    """
    control_points = [[0, 0], [2, 2]]
    weights = [1, 1]
    knot_vector = [0, 0, 1, 1]
    degree = 1
    curve = NURBSCurve(control_points, weights, knot_vector, degree)

    min_pt, max_pt = curve.bounding_box(num_samples=10)
    np.testing.assert_allclose(min_pt, [0, 0])
    np.testing.assert_allclose(max_pt, [2, 2])


def test_is_closed():
    """
    By default, is_closed() checks start_point vs end_point at t=0 vs t=1.
    A line from (0,0) to (2,2) is not closed => False.
    """
    control_points = [[0, 0], [2, 2]]
    weights = [1, 1]
    knot_vector = [0, 0, 1, 1]
    degree = 1
    curve = NURBSCurve(control_points, weights, knot_vector, degree)
    assert curve.is_closed() is False

    # If we forcibly made them the same, is_closed() would be True
    control_points2 = [[0, 0], [0, 0]]  # start == end
    weights2 = [1, 1]
    knot_vector2 = [0, 0, 1, 1]
    curve2 = NURBSCurve(control_points2, weights2, knot_vector2, degree=1)
    assert curve2.is_closed() is True
