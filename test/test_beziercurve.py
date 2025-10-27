import pytest
import numpy as np

#from svv.forest.connect.base import Curve
from svv.forest.connect.bezier import BezierCurve


def test_constructor_valid_2d():
    """Test constructing a valid 2D BezierCurve."""
    control_points = [[0, 0], [1, 1], [2, 0]]  # Quadratic in 2D
    curve = BezierCurve(control_points)
    assert isinstance(curve, BezierCurve)
    assert curve.dimension == 2
    assert curve.control_points.shape == (3, 2)


def test_constructor_valid_3d():
    """Test constructing a valid 3D BezierCurve."""
    control_points = [[0, 0, 0], [1, 1, 1], [2, 1, 0], [3, 3, 3]]  # Cubic in 3D
    curve = BezierCurve(control_points)
    assert isinstance(curve, BezierCurve)
    assert curve.dimension == 3
    assert curve.control_points.shape == (4, 3)


def test_constructor_invalid_dimension():
    """Test that constructing with an invalid dimension (not 2D array) raises ValueError."""
    with pytest.raises(ValueError):
        BezierCurve([0, 1, 2])  # This is a 1D list, not (n+1, d)


def test_evaluate_linear_2d():
    """
    Test a simple linear (degree=1) 2D Bezier curve:
    control_points = [(0,0), (1,0)]
    The curve is a straight line from (0,0) to (1,0).
    """
    control_points = [[0, 0], [1, 0]]  # Linear in 2D
    curve = BezierCurve(control_points)

    # t=0 -> (0,0), t=1 -> (1,0), t=0.5 -> (0.5, 0)
    t_values = [0, 0.5, 1]
    expected = np.array([[0, 0],
                         [0.5, 0],
                         [1, 0]])
    result = curve.evaluate(t_values)
    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_evaluate_quadratic_2d():
    """
    Test a quadratic 2D Bezier curve:
    control_points = [(0,0), (1,1), (2,0)]
    Known property: at t=0.5, point is average of [0.5*(P0+P1), 0.5*(P1+P2)].
    """
    control_points = [[0, 0], [1, 1], [2, 0]]
    curve = BezierCurve(control_points)

    # Evaluate at specific t-values
    t_values = np.array([0, 0.5, 1])
    # t=0  => (0,0)
    # t=1  => (2,0)
    # t=0.5 => By De Casteljau:
    #    mid-level points: (0.5, 0.5) and (1.5, 0.5)
    #    final point: 0.5*(0.5,0.5) + 0.5*(1.5,0.5) = (1.0, 0.5)
    expected = np.array([[0, 0],
                         [1, 0.5],
                         [2, 0]])
    result = curve.evaluate(t_values)
    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_derivative_linear_2d():
    """
    Test that the derivative of a linear 2D Bezier curve is constant.
    control_points = [(0,0), (2,2)] => slope is (2,2).
    The derivative is (2,2) for all t. The second derivative is (0,0).
    """
    control_points = [[0, 0], [2, 2]]
    curve = BezierCurve(control_points)

    # First derivative should be constant
    t_values = [0, 0.3, 0.7, 1.0]
    d1 = curve.derivative(t_values, order=1)
    np.testing.assert_allclose(d1, [[2, 2]] * len(t_values), atol=1e-8)

    # Second derivative should be zero
    d2 = curve.derivative(t_values, order=2)
    np.testing.assert_allclose(d2, [[0, 0]] * len(t_values), atol=1e-8)

    # Beyond second derivative is still zero
    d3 = curve.derivative(t_values, order=3)
    np.testing.assert_allclose(d3, [[0, 0]] * len(t_values), atol=1e-8)


def test_derivative_quadratic_2d():
    """
    Test the first derivative for a quadratic Bezier curve:
    control_points = [(0,0), (2,2), (4,0)]
    Degree=2 => first derivative is linear, second derivative is constant.
    """
    control_points = [[0, 0], [2, 2], [4, 0]]
    curve = BezierCurve(control_points)

    # Evaluate derivative at t=0, t=1:
    # derivative control points => n*(P1 - P0), n*(P2 - P1)
    # => 2*[(2,2)-(0,0)], 2*[(4,0)-(2,2)] => (4,4), (4,-4)
    # De Casteljau for derivative at t=0 => (4,4), t=1 => (4,-4)
    # second derivative => linear difference => (4,-4)-(4,4) = (0,-8)
    # multiplied by (n-1) = (2-1)=1 => (0, -8)
    t_values = [0, 1]
    d1 = curve.derivative(t_values, order=1)  # Should be (4,4) at t=0, (4,-4) at t=1
    expected_d1 = np.array([[4, 4],
                            [4, -4]])
    np.testing.assert_allclose(d1, expected_d1, atol=1e-8)

    d2 = curve.derivative(t_values, order=2)  # Should be constant => (0, -8)
    expected_d2 = np.array([[0, -8],
                            [0, -8]])
    np.testing.assert_allclose(d2, expected_d2, atol=1e-8)


def test_roc_linear_2d():
    """
    For a linear Bezier curve, the radius of curvature is theoretically infinite.
    We expect the code to return a very large number (since cross-product = 0).
    """
    control_points = [[0, 0], [1, 0]]
    curve = BezierCurve(control_points)
    # For a line, v x a = 0 => denominator = 0 => roc => ~ large
    t_values = np.linspace(0, 1, 5)
    roc_vals = curve.roc(t_values)

    # All values should be large (the code uses an epsilon in the denominator)
    # Typically we'd get v_mags^3 / epsilon for a nonzero velocity.
    assert np.all(roc_vals > 1e6), "Radius of curvature for a straight line should be very large."


def test_torsion_2d():
    """
    Torsion of a curve in a 2D plane should be zero (or effectively zero).
    """
    control_points = [[0, 0], [1, 1], [2, 0]]
    curve = BezierCurve(control_points)
    t_values = np.linspace(0, 1, 5)
    torsion_vals = curve.torsion(t_values)
    # Should be all zero (or near-zero), as this is still a curve in 2D plane
    np.testing.assert_allclose(torsion_vals, np.zeros_like(torsion_vals), atol=1e-10)


def test_arc_length_linear_2d():
    """
    Arc length of a linear Bezier curve from (0,0) to (1,1) is sqrt(2).
    """
    control_points = [[0, 0], [1, 1]]
    curve = BezierCurve(control_points)
    arc_len = curve.arc_length(0, 1, num_points=10)
    np.testing.assert_allclose(arc_len, np.sqrt(2), atol=1e-3)


def test_arc_length_invalid_domain():
    """
    Test that passing invalid t_start or t_end raises ValueError.
    """
    control_points = [[0, 0], [1, 1]]
    curve = BezierCurve(control_points)
    with pytest.raises(ValueError):
        curve.arc_length(t_start=0.5, t_end=0.4)  # t_start >= t_end
    with pytest.raises(ValueError):
        curve.arc_length(t_start=-0.1, t_end=0.5)  # negative range
    with pytest.raises(ValueError):
        curve.arc_length(t_start=0.5, t_end=1.1)   # end > 1


def test_dimension_property():
    """Test the dimension property for a 2D and 3D Bezier curve."""
    curve2d = BezierCurve([[0, 0], [1, 1], [2, 0]])
    assert curve2d.dimension == 2

    curve3d = BezierCurve([[0, 0, 0], [1, 1, 1], [2, 0, 2]])
    assert curve3d.dimension == 3


def test_evaluate_scalar_t():
    """Test that evaluate works properly with a single (scalar) t value."""
    curve = BezierCurve([[0, 0], [1, 1], [2, 0]])
    point = curve.evaluate(0.5)  # should return shape (1, 2)
    assert point.shape == (1, 2)
    np.testing.assert_allclose(point[0], [1, 0.5], atol=1e-8)


def test_derivative_order_exceeds_degree():
    """
    Test that if derivative order exceeds polynomial degree, the result is zero.
    E.g., 3 control points => degree=2 => 3rd derivative and beyond should be zero.
    """
    curve = BezierCurve([[0, 0], [1, 1], [2, 0]])  # degree=2
    t_values = [0, 0.5, 1]
    d3 = curve.derivative(t_values, order=3)
    np.testing.assert_allclose(d3, 0.0, atol=1e-8)
    d4 = curve.derivative(t_values, order=4)
    np.testing.assert_allclose(d4, 0.0, atol=1e-8)
