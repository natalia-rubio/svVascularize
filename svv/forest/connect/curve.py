import numpy as np

from svv.forest.connect.bezier import BezierCurve
from svv.forest.connect.catmullrom import CatmullRomCurve
from svv.forest.connect.nurbs import NURBSCurve


class Curve:
    def __init__(self, control_points, curve_type="Bezier", **kwargs):
        """
        Initializes the Curve object based on the specified control points and curve type.

        Parameters
        ----------
        control_points : array-like
            Control points for the curve.
        curve_type : str, optional
            The type of curve to instantiate ('Bezier', 'CatmullRom', 'NURBS'). Default is 'Bezier'.
        **kwargs : Additional arguments to be passed to the corresponding curve constructor.
        """
        self.curve_type = curve_type

        if curve_type == "Bezier":
            self.curve = BezierCurve(control_points)
        elif curve_type == "CatmullRom":
            self.curve = CatmullRomCurve(control_points, **kwargs)
        elif curve_type == "NURBS":
            self.curve = NURBSCurve(control_points, **kwargs)
        else:
            raise ValueError("Unsupported curve type")

    def evaluate(self, t_values):
        return self.curve.evaluate(t_values)

    def derivative(self, t_values, order=1):
        return self.curve.derivative(t_values, order)

    def arc_length(self, t_start=0, t_end=1, num_points=100):
        return self.curve.arc_length(t_start, t_end, num_points)

    def roc(self, t_values):
        return self.curve.roc(t_values)

    def torsion(self, t_values):
        return self.curve.torsion(t_values)

    @property
    def dimension(self):
        return self.curve.dimension

    def __str__(self):
        return f"Curve type: {self.curve.__class__.__name__}, Control Points: {self.curve.control_points.shape[0]}"
