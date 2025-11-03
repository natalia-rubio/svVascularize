import numpy
import pyvista


def read(data, **kwargs):
    """
    Read and process mesh data. Supports PyVista objects and file paths to mesh files.

    Parameters:
        data: pyvista.PolyData or str
            Input mesh data as a PyVista object or a file path to a supported mesh file.
        **kwargs:
            feature_angle: float
                Angle used to determine sharp edges for normal computation.


    Returns:
        points: numpy.ndarray
            Array of point coordinates from the mesh.
        normals: numpy.ndarray
            Array of computed point normals from the mesh.
        n: int
            Number of points in the mesh.
        d: int
            Dimension of the points (typically 3).
    """
    feature_angle = kwargs.get("feature_angle", 30.0)

    if isinstance(data, str):
        data = pyvista.read(data)

    if not isinstance(data, pyvista.PolyData):
        raise TypeError("Input data must be a PyVista PolyData object or a valid file path.")

    data = data.compute_normals(split_vertices=True, feature_angle=feature_angle)

    points = data.points.astype(numpy.float64)
    normals = data.point_normals.astype(numpy.float64)
    n = points.shape[0]
    d = points.shape[1]
    return points, normals, n, d