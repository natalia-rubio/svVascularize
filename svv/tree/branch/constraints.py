import numpy
from svv.tree.utils.c_angle import get_angles


def angle_constraint(points, point_1, point_2, constraint, greater):
    """
    This function computes the angles formed among an array of points
    and two other points (point_1 and point_2). The points that form
    angles satisfying the desired constraint are returned.

    Parameters
    ----------
    points : numpy.ndarray
        An array of points.
    point_1 : numpy.ndarray
        Point forming the first vector of the angles.
    point_2 : numpy.ndarray
        Point forming the second vector of the angles.
    constraint : float
        The constraint on the angles.
    greater : bool
        If True, the angles must be greater than the constraint.
        If False, the angles must be less than the constraint.

    Returns
    -------
    points : numpy.ndarray
        Points satisfying the constraint.
    """
    v1 = point_1 - points
    v2 = point_2 - points
    v1 = v1 / numpy.linalg.norm(v1, axis=1).reshape(-1, 1)
    v2 = v2 / numpy.linalg.norm(v2, axis=1).reshape(-1, 1)
    angles = get_angles(v1, v2)
    if greater:
        points = points[angles > constraint, :]
    else:
        points = points[angles < constraint, :]
    return points


def relative_length_constraint(points, point_1, point_2, point_3, ratio):
    """
    This function computes the relative lengths of the lines formed
    by an array of points and three other points (point_1, point_2,
    and point_3). The points that form relative lengths satisfying
    the desired constraint are returned.

    Parameters
    ----------
    points : numpy.ndarray
        An array of points.
    point_1 : numpy.ndarray
        Point forming the first line segment of the relative lengths.
    point_2 : numpy.ndarray
        Point forming the second line segment of the relative lengths.
    point_3 : numpy.ndarray
        Point forming the third line segment of the relative lengths.
    ratio : float
        The positive constraint on the relative lengths. (Must be
        greater than 0)

    Returns
    -------
    points : numpy.ndarray
        Points satisfying the constraint.
    """
    minimum_distance = min([numpy.linalg.norm(point_1 - point_2),
                            numpy.linalg.norm(point_1 - point_3),
                            numpy.linalg.norm(point_2 - point_3)])
    minimum_distance = minimum_distance * ratio
    points = points[numpy.linalg.norm(point_1 - points, axis=1) > minimum_distance, :]
    points = points[numpy.linalg.norm(point_2 - points, axis=1) > minimum_distance, :]
    points = points[numpy.linalg.norm(point_3 - points, axis=1) > minimum_distance, :]
    return points


def absolute_length_constraint(points, point_1, point_2, point_3, length):
    """
    This function computes the absolute lengths of the lines formed
    by an array of points and three other points (point_1, point_2,
    and point_3). The points that form absolute lengths satisfying
    the desired constraint are returned.

    Parameters
    ----------
    points : numpy.ndarray
        An array of points.
    point_1 : numpy.ndarray
        Point forming the first line segment of the absolute lengths.
    point_2 : numpy.ndarray
        Point forming the second line segment of the absolute lengths.
    point_3 : numpy.ndarray
        Point forming the third line segment of the absolute lengths.
    length : float
        The positive constraint on the absolute lengths. (Must be
        greater than 0)

    Returns
    -------
    points : numpy.ndarray
        Points satisfying the constraint.
    """
    minimum_distance = min([numpy.linalg.norm(point_1 - point_2),
                            numpy.linalg.norm(point_1 - point_3),
                            numpy.linalg.norm(point_2 - point_3)])
    if minimum_distance < length:
        points = numpy.array([[]])
    else:
        points = points[numpy.linalg.norm(point_1 - points, axis=1) > length, :]
        points = points[numpy.linalg.norm(point_2 - points, axis=1) > length, :]
        points = points[numpy.linalg.norm(point_3 - points, axis=1) > length, :]
    return points


def boundary_constraint(points, domain):
    """
    This function computes the points that lie within the domain.
    These points are returned.

    Parameters
    ----------
    points : numpy.ndarray
        An array of points.
    domain : svtoolkit.Domain
        The domain object that defines the space in which
        the points must lie.

    Returns
    -------
    points : numpy.ndarray
        Points satisfying the constraint.
    """
    points = points[domain(points) < 0, :]
    return points
