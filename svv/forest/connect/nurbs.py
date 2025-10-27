from abc import ABC

import numpy as np

from svv.forest.connect.base import BaseCurve


class NURBSCurve(BaseCurve, ABC):
    """
    A concrete implementation of a Non-Uniform Rational B-Spline (NURBS) curve.

    This class inherits from the abstract base Curve. It satisfies the required
    interface by implementing (or stubbing) all abstract methods.

    Parameters
    ----------
    control_points : (N, d) array-like
        The N control points, each of dimension d (d=2 or d=3 typically).
    weights : (N,) array-like or None, optional
        Weights associated with each control point. Must match length of control_points.
        Defaults to all 1's (uniform).
    knot_vector : (M,) array-like or None, optional
        The non-decreasing knot vector. Typically, M = N + degree + 1 for a clamped spline.
        If not provided, a default clamped knot vector is used.
    degree : int or None, optional
        The polynomial degree of the B-spline basis (e.g., 2 for quadratic, 3 for cubic).
        Defaults to `min(len(control_points) - 1, 3)`.

    Notes
    -----
    - This minimal class focuses on the evaluate() method, which takes parameter t in [0,1]
      and maps it to the relevant knot span [knot_vector[degree], knot_vector[-(degree+1)]].
    - The derivative, roc, torsion, and arc_length methods currently raise NotImplementedError.
      You may implement them with analytic or numeric approaches if needed.
    """

    def __init__(self, control_points, weights=None, knot_vector=None, degree=None):
        # Convert control_points to NumPy array
        self.control_points = np.array(control_points, dtype=float)

        # 1) Default weights: all 1s if not given
        if weights is None:
            self.weights = np.ones(len(self.control_points))
        else:
            self.weights = np.array(weights, dtype=float)

        # 2) Default degree: cubic or the maximum permissible (len - 1) if smaller
        if degree is None:
            self.degree = min(len(self.control_points) - 1, 3)
        else:
            self.degree = degree

        # 3) Default knot vector: clamped with uniform spacing in the interior
        #
        #    A "clamped" knot vector has multiplicities of (degree+1) at the
        #    beginning and end. For instance, if degree=3 and we have 5 control
        #    points, the vector might look like:
        #    [0, 0, 0, 0, 0.5, 1, 1, 1, 1]
        if knot_vector is None:
            num_ctrl_pts = len(self.control_points)
            n_knots = num_ctrl_pts + self.degree + 1

            # interior knots are uniformly spaced. For example, if degree=3 and
            # we have 5 points, we will have 9 knots total. The first 4 are 0,
            # the last 4 are 1, and there's 1 interior knot which is 0.5.
            # More generally, we have (n_knots - 2*(degree+1)) interior knots.
            n_interior = n_knots - 2 * (self.degree + 1)
            # If there are no interior knots, n_interior will be 0 and
            # np.linspace(0,1,0) is an empty array.

            # build the clamped knot vector
            self.knot_vector = np.concatenate([
                np.zeros(self.degree),                  # left clamped
                np.linspace(0, 1, n_interior + 2)[1:-1], # interior
                np.ones(self.degree)                    # right clamped
            ]) if n_interior >= 0 else None

            # Fallback if we somehow don't have enough points for the above approach:
            if self.knot_vector is None or len(self.knot_vector) != n_knots:
                # simpler fallback: just replicate the standard approach
                self.knot_vector = np.concatenate((
                    np.zeros(self.degree + 1),
                    np.linspace(0, 1, n_knots - 2 * (self.degree + 1)),
                    np.ones(self.degree + 1)
                ))
        else:
            self.knot_vector = np.array(knot_vector, dtype=float)

        # Basic validations
        if len(self.control_points) != len(self.weights):
            raise ValueError("control_points and weights must have the same length.")

        num_ctrl_pts = len(self.control_points)
        if len(self.knot_vector) != num_ctrl_pts + self.degree + 1:
            raise ValueError(
                "knot_vector length must be (num_control_points + degree + 1)."
            )

        if not np.all(self.knot_vector[:-1] <= self.knot_vector[1:]):
            raise ValueError("knot_vector must be non-decreasing.")

        # Precompute the valid param domain from the knot vector
        self._t_min = self.knot_vector[self.degree]
        self._t_max = self.knot_vector[-(self.degree + 1)]

    @property
    def dimension(self):
        """Return the dimension (2 or 3) of the control points."""
        return self.control_points.shape[1]

    def evaluate(self, t_values):
        """
        Evaluate the NURBS curve at the specified parameter values in [0,1].

        Parameters
        ----------
        t_values : float or array-like
            Parametric value(s) in [0, 1].

        Returns
        -------
        np.ndarray
            Evaluated points on the NURBS curve. Shape (len(t_values), dimension).

        Notes
        -----
        - We linearly map each t from [0,1] to the physical domain
          [self._t_min, self._t_max] before evaluating the NURBS basis.
        - The NURBS curve is given by:

            C(t) = (Σ ( N_{i,p}(t) * w_i * P_i )) / (Σ ( N_{i,p}(t) * w_i )),

          where N_{i,p}(t) is the B-spline basis function of degree p using
          the Cox–de Boor recursion, and w_i are the weights.
        """
        t_values = np.atleast_1d(t_values).astype(float)

        # Map [0,1] -> [t_min, t_max]
        physical_ts = self._t_min + t_values * (self._t_max - self._t_min)

        # We'll evaluate at each t in physical_ts
        points_out = np.zeros((len(physical_ts), self.dimension))

        for idx, t_physical in enumerate(physical_ts):
            # Compute numerator and denominator of the NURBS equation
            numerator = np.zeros(self.dimension, dtype=float)
            denominator = 0.0
            for i in range(len(self.control_points)):
                Ni = self._basis_function(i, self.degree, t_physical, self.knot_vector)
                w_i = self.weights[i]
                numerator += Ni * w_i * self.control_points[i]
                denominator += Ni * w_i

            # Safeguard if denominator is near zero
            if abs(denominator) < 1e-14:
                # Typically occurs if t is out of range or at extremes.
                # We'll clamp to first or last control point in that case.
                if t_physical <= self._t_min:
                    points_out[idx] = self.control_points[0]
                else:
                    points_out[idx] = self.control_points[-1]
            else:
                points_out[idx] = numerator / denominator

        return points_out

    def derivative(self, t_values, order=1):
        """
        Compute the nth derivative of the NURBS curve.
        Currently unimplemented.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("NURBS derivative not implemented yet.")

    def roc(self, t_values):
        """
        Radius of curvature not implemented for NURBS yet.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("NURBS radius of curvature not implemented yet.")

    def torsion(self, t_values):
        """
        Torsion not implemented for NURBS yet.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("NURBS torsion not implemented yet.")

    def arc_length(self, t_start=0.0, t_end=1.0, num_points=100):
        """
        Approximate the arc length. Currently unimplemented.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("NURBS arc length not implemented yet.")

    def transform(self, matrix):
        """
        Apply a transformation matrix to the control points.
        For now, raise NotImplementedError or implement if desired.
        """
        raise NotImplementedError("NURBS transform not implemented yet.")

    # ------------------------------------------------------------
    # Cox–de Boor recursion for the B-spline basis N_{i,p}(t).
    # ------------------------------------------------------------
    def _basis_function(self, i, p, t, knots):
        """
        Cox–de Boor recursive definition of B-spline basis function N_{i,p}(t).

        Parameters
        ----------
        i : int
            Index of the basis function.
        p : int
            Degree of the basis function.
        t : float
            Parameter to evaluate at.
        knots : array-like
            Knot vector.

        Returns
        -------
        float
            The value of the B-spline basis function N_{i,p}(t).
        """
        # Base case p=0
        if p == 0:
            return 1.0 if (knots[i] <= t < knots[i+1]) else 0.0

        denom1 = knots[i+p] - knots[i]
        denom2 = knots[i+p+1] - knots[i+1]

        term1 = 0.0
        term2 = 0.0

        if denom1 != 0:
            term1 = ((t - knots[i]) / denom1) * self._basis_function(i, p - 1, t, knots)
        if denom2 != 0:
            term2 = ((knots[i+p+1] - t) / denom2) * self._basis_function(i+1, p - 1, t, knots)

        return term1 + term2