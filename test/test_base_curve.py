# test_base_curve.py

import pytest
import numpy as np

from svv.forest.connect.base import BaseCurve

def test_cannot_instantiate_base_curve():
    """
    Attempting to instantiate an ABC (BaseCurve) directly should fail
    because it has abstract methods and properties.
    """
    with pytest.raises(TypeError):
        _ = BaseCurve()

# ----------------------------------------------------------------------------
# Mock subclass that provides minimal implementations of abstract methods.
# ----------------------------------------------------------------------------
class MockCurve(BaseCurve):
    """
    A minimal subclass of BaseCurve that just implements the abstract methods
    enough to allow instantiation and testing of base-class functionality.
    """

    def evaluate(self, t_values):
        # Return a straight line, e.g. y = x, in 2D
        t_values = np.atleast_1d(t_values)
        return np.column_stack((t_values, t_values))

    def derivative(self, t_values, order=1):
        # Derivative of y=x line is (1,1), second derivative is (0,0), etc.
        t_values = np.atleast_1d(t_values)
        if order == 1:
            # shape = (len(t_values), dimension)
            return np.tile([1.0, 1.0], (len(t_values), 1))
        else:
            # higher derivatives are zero
            return np.zeros((len(t_values), 2))

    def roc(self, t_values):
        # Radius of curvature for a line is infinite. Return large number.
        t_values = np.atleast_1d(t_values)
        return np.full(t_values.shape, 1e9)

    def torsion(self, t_values):
        # In 2D line, torsion = 0
        t_values = np.atleast_1d(t_values)
        return np.zeros_like(t_values, dtype=float)

    def arc_length(self, t_start=0.0, t_end=1.0, num_points=100):
        # For line y=x from t_start to t_end, the length is sqrt(2)*(t_end - t_start)
        return np.sqrt(2) * (t_end - t_start)

    @property
    def dimension(self):
        return 2  # This mock curve is in 2D

# ----------------------------------------------------------------------------
# Test the mock subclass (which indirectly tests some base-class behaviors).
# ----------------------------------------------------------------------------

def test_mock_curve_instantiation():
    """
    Instantiating MockCurve should succeed because
    it implements all abstract methods.
    """
    curve = MockCurve()
    assert isinstance(curve, BaseCurve)
    assert curve.dimension == 2

def test_mock_curve_evaluate():
    """
    Check that the mock curve returns the expected (x, x) line.
    """
    curve = MockCurve()
    t_values = [0, 0.5, 1.0]
    pts = curve.evaluate(t_values)
    expected = np.array([[0.0, 0.0],
                         [0.5, 0.5],
                         [1.0, 1.0]])
    np.testing.assert_allclose(pts, expected, atol=1e-8)

def test_mock_curve_derivative():
    """
    For a line y=x, the first derivative is (1,1),
    second derivative is (0,0), etc.
    """
    curve = MockCurve()
    t_values = [0, 0.25, 0.75, 1]
    d1 = curve.derivative(t_values, order=1)
    np.testing.assert_allclose(d1, [[1,1]]*len(t_values), atol=1e-8)
    d2 = curve.derivative(t_values, order=2)
    np.testing.assert_allclose(d2, [[0,0]]*len(t_values), atol=1e-8)

def test_mock_curve_roc():
    """
    For the line mock, roc() returns a large constant (representing infinity).
    """
    curve = MockCurve()
    t_vals = np.linspace(0, 1, 5)
    roc_vals = curve.roc(t_vals)
    assert np.all(roc_vals > 1e6), "ROC for a line should be large (infinite)."

def test_mock_curve_torsion():
    """
    Torsion in 2D line should be 0.
    """
    curve = MockCurve()
    t_vals = np.linspace(0, 1, 5)
    torsion_vals = curve.torsion(t_vals)
    np.testing.assert_allclose(torsion_vals, 0.0)

def test_mock_curve_arc_length():
    """
    Arc length for line y=x from 0 to 1 is sqrt(2).
    """
    curve = MockCurve()
    length = curve.arc_length(t_start=0, t_end=1, num_points=50)
    np.testing.assert_allclose(length, np.sqrt(2), atol=1e-6)

def test_mock_curve_bounding_box():
    """
    bounding_box is a base-class method that uses evaluate internally.
    The line from t=0 to t=1 goes from (0,0) to (1,1).
    """
    curve = MockCurve()
    min_pt, max_pt = curve.bounding_box(num_samples=5)
    np.testing.assert_allclose(min_pt, [0, 0])
    np.testing.assert_allclose(max_pt, [1, 1])

def test_mock_curve_is_closed():
    """
    is_closed checks whether the start point and end point are within tolerance.
    The line from (0,0) to (1,1) is not closed.
    """
    curve = MockCurve()
    assert curve.is_closed() == False

def test_transform_not_implemented():
    """
    The base transform method raises NotImplementedError
    unless a subclass overrides it.
    """
    curve = MockCurve()
    with pytest.raises(NotImplementedError):
        curve.transform(np.eye(3))
