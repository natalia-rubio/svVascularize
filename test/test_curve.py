import pytest
import numpy as np
from svv.forest.connect.bezier import BezierCurve
from svv.forest.connect.catmullrom import CatmullRomCurve
from svv.forest.connect.nurbs import NURBSCurve
from svv.forest.connect.curve import Curve


# Test Case 1: Default curve type (Bezier)
def test_default_curve_type():
    control_points = [(0, 0), (1, 2), (3, 4)]
    curve = Curve(control_points)

    # Check that the correct type is instantiated
    assert isinstance(curve.curve, BezierCurve)
    assert curve.curve.control_points.shape == (3, 2)  # 3 control points, 2D


# Test Case 2: CatmullRom curve type
def test_catmull_rom_curve():
    control_points = [(0, 0), (1, 2), (2, 3), (4, 4)]
    curve = Curve(control_points, curve_type="CatmullRom")

    # Check that the correct type is instantiated
    assert isinstance(curve.curve, CatmullRomCurve)
    assert curve.curve.control_points.shape == (4, 2)  # 4 control points, 2D


# Test Case 3: NURBS curve type with additional parameters
#def test_nurbs_curve():
#    control_points = [(0, 0), (1, 2), (3, 4)]
#    knots = [0, 0, 1, 1]
#    weights = [1, 1, 1]
#    curve = Curve(control_points, curve_type="NURBS", knots=knots, weights=weights)
#
#    # Check that the correct type is instantiated
#    assert isinstance(curve.curve, NURBSCurve)
#    assert curve.curve.control_points.shape == (3, 2)  # 3 control points, 2D
#    assert curve.curve.knots == knots
#    assert curve.curve.weights == weights


# Test Case 4: Unsupported curve type
def test_unsupported_curve_type():
    control_points = [(0, 0), (1, 2), (3, 4)]

    with pytest.raises(ValueError, match="Unsupported curve type"):
        Curve(control_points, curve_type="Unsupported")


# Test Case 5: Evaluate method
def test_evaluate_method():
    control_points = [(0, 0), (1, 2), (3, 4)]
    curve = Curve(control_points, curve_type="Bezier")
    t_values = [0.0, 0.5, 1.0]

    # Evaluate the curve at the specified parametric values
    evaluated_points = curve.evaluate(t_values)

    # Check the returned shape (it should be len(t_values) x d)
    assert evaluated_points.shape == (3, 2)  # 3 t_values, 2D points


# Test Case 6: Derivative method
def test_derivative_method():
    control_points = [(0, 0), (1, 2), (3, 4)]
    curve = Curve(control_points, curve_type="Bezier")
    t_values = [0.0, 0.5, 1.0]

    # Compute the first derivative of the curve
    derivative_points = curve.derivative(t_values, order=1)

    # Check the returned shape (it should be len(t_values) x d)
    assert derivative_points.shape == (3, 2)  # 3 t_values, 2D points


# Test Case 7: Arc length method
def test_arc_length_method():
    control_points = [(0, 0), (1, 2), (3, 4)]
    curve = Curve(control_points, curve_type="Bezier")

    # Compute the arc length between t_start=0 and t_end=1
    arc_length = curve.arc_length(t_start=0, t_end=1)

    # Check if the arc length is a positive value
    assert arc_length > 0


# Test Case 8: Radius of Curvature (ROC) method
def test_roc_method():
    control_points = [(0, 0), (1, 2), (3, 4)]
    curve = Curve(control_points, curve_type="Bezier")
    t_values = [0.0, 0.5, 1.0]

    # Compute the radius of curvature at the specified t_values
    roc_values = curve.roc(t_values)

    # Check if the ROC values are of the expected shape
    assert roc_values.shape == (3,)


# Test Case 9: Torsion method
def test_torsion_method():
    control_points = [(0, 0), (1, 2), (3, 4)]
    curve = Curve(control_points, curve_type="Bezier")
    t_values = [0.0, 0.5, 1.0]

    # Compute the torsion at the specified t_values
    torsion_values = curve.torsion(t_values)

    # Check if torsion is calculated, should be zero for 2D curves
    assert torsion_values.shape == (3,)
    assert np.all(torsion_values == 0)


# Test Case 10: Dimension property
def test_dimension_property():
    control_points = [(0, 0), (1, 2), (3, 4)]
    curve = Curve(control_points, curve_type="Bezier")

    # Check if dimension property returns the correct value
    assert curve.dimension == 2  # 2D curve


# Test Case 11: String representation
def test_str_method():
    control_points = [(0, 0), (1, 2), (3, 4)]
    curve = Curve(control_points, curve_type="Bezier")

    # Check if the string representation contains the correct curve type and control points count
    assert str(curve) == "Curve type: BezierCurve, Control Points: 3"

