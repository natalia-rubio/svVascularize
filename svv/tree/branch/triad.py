import numpy as np


def local_linear_points(proximal, distal, sampling):
    """
    This function takes the proximal and distal points
    of a vessel and returns a grid of points sampling
    the space between the two points.

    Parameters
    ----------
    proximal : ndarray
        The proximal point of the vessel.
    distal : ndarray
        The distal point of the vessel.
    sampling : int
        The number of points to sample in each direction.

    Returns
    -------
    ndarray
        The array of points sampled between the two points.
    """
    proximal = proximal.reshape(1, -1)
    distal = distal.reshape(1, -1)
    t = np.linspace(0, 1, num=sampling)
    points = proximal*(1-t) + distal*t
    return points


def local_triad_points(proximal, distal, terminal, sampling):
    """
    This function takes the triad of the vessel and a
    new darcy point and returns a grid of points
    sampling the space within the triangular region
    enclosed by the proximal, distal and darcy point.

    Parameters
    ----------
    proximal : ndarray
        The proximal point of the vessel forming the triad.
    distal : ndarray
        The distal point of the vessel forming the triad.
    terminal : ndarray
        The new terminal darcy point forming the triad.
    sampling : int
        The number of points to sample in each direction.

    Returns
    -------
    ndarray
        The array of points sampled within the triangular region.
    """
    proximal = proximal.reshape(1, -1)
    distal = distal.reshape(1, -1)
    terminal = terminal.reshape(1, -1)
    line = np.linspace(0, 1, num=sampling)
    s, t = np.meshgrid(line, line)
    s = s.flatten().reshape(-1, 1)
    t = t.flatten().reshape(-1, 1)
    points = proximal*(1-t)*s + distal*(t*s) + terminal*(1-s)
    return points
