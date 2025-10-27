import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class BaseCurve(ABC):
    """
    A base abstract class for curves in 2D or 3D space.

    Provides a consistent interface for evaluating, differentiating,
    computing curvature/torsion, arc length, etc. Also includes
    optional methods like dimension, bounding box, transformations,
    and plotting.
    """

    @abstractmethod
    def evaluate(self, t_values):
        """
        Evaluate the curve at given parametric values.
        Parameters
        ----------
        t_values : array-like
            Parametric values, typically in [0, 1].
        Returns
        -------
        np.ndarray
            Points on the curve at each parametric value.
        """
        pass

    @abstractmethod
    def derivative(self, t_values, order=1):
        """
        Compute the nth derivative of the curve.
        Parameters
        ----------
        t_values : array-like
            Parametric values in [0, 1].
        order : int, optional
            The order of the derivative (1 for first derivative, etc.).
        Returns
        -------
        np.ndarray
            Derivative values at each parametric value.
        """
        pass

    @abstractmethod
    def roc(self, t_values):
        """
        Compute the radius of curvature at specified parametric values.
        """
        pass

    @abstractmethod
    def torsion(self, t_values):
        """
        Compute the torsion at specified parametric values.
        """
        pass

    @abstractmethod
    def arc_length(self, t_start=0.0, t_end=1.0, num_points=100):
        """
        Approximate the arc length of the curve from t_start to t_end.
        """
        pass

    # -- Additional Optional Methods / Properties --

    @property
    @abstractmethod
    def dimension(self):
        """
        Return the dimension of the curve (2 for 2D, 3 for 3D).
        """
        pass

    def bounding_box(self, num_samples=100):
        """
        Return an axis-aligned bounding box of the curve over its parameter range.
        This implementation is a naive sample-based approach; override for exact if needed.
        """
        # By default, assume domain is [0,1], override if your curve has different domain
        t_values = np.linspace(0, 1, num_samples)
        points = self.evaluate(t_values)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        return min_coords, max_coords

    def is_closed(self, tol=1e-7):
        """
        Check if the curve is closed within some tolerance (start point == end point).
        """
        start_point = self.evaluate([0])[0]
        end_point = self.evaluate([1])[0]
        return np.allclose(start_point, end_point, atol=tol)

    def transform(self, matrix):
        """
        Apply a transformation matrix to the curve's underlying data.
        By default, raise NotImplementedError; each subclass must define its own transformation logic.
        """
        raise NotImplementedError(
            "Subclass must override transform method to handle transformations."
        )

    def plot(self, num_points=100, t_start=0.0, t_end=1.0, ax=None, **plot_kwargs):
        """
        Quick 2D plot of the curve using matplotlib.

        Parameters
        ----------
        num_points : int
            Number of sample points.
        t_start : float
            Start parameter (default 0.0).
        t_end : float
            End parameter (default 1.0).
        ax : matplotlib.axes.Axes
            If provided, draw on the given axes. Otherwise, create a new figure.
        plot_kwargs : dict
            Additional keyword arguments passed to `matplotlib.pyplot.plot`.
        """
        if self.dimension < 2:
            raise ValueError("Cannot plot a curve with dimension < 2.")

        # Evaluate points
        t_values = np.linspace(t_start, t_end, num_points)
        points = self.evaluate(t_values)  # shape: (num_points, dimension)

        if ax is None:
            fig, ax = plt.subplots()

        # If dimension=2, we can do a direct 2D plot
        if self.dimension == 2:
            ax.plot(points[:, 0], points[:, 1], **plot_kwargs)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_title("2D Curve")
        elif self.dimension == 3:
            # For a 3D curve, you'd need a 3D axis
            from mpl_toolkits.mplot3d import Axes3D  # or use matplotlib >= 3.2 with fig.add_subplot(projection='3d')
            if not hasattr(ax, 'plot3D'):
                # auto-create a 3D axis if user didn’t provide one
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], **plot_kwargs)
            ax.set_title("3D Curve")
        else:
            raise NotImplementedError("Plotting only implemented for 2D or 3D curves.")

        return ax
