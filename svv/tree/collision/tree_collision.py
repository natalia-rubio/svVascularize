import numpy
from copy import deepcopy

from svv.tree.utils.c_close import sphere_proximity
from svv.tree.utils.c_obb import obb_any

def tree_proximity(data, vessel, **kwargs):
    """
    Check if a vessel has intersecting bounding spheres with vessels
    in the current tree.

    Parameters
    ----------
    data : numpy.ndarray
        An array of vessel data.
    vessel : numpy.ndarray
        An array of vessel data.
    **kwargs : dict
        Additional keyword arguments.
        Keyword arguments include:
            clearance : float
                The clearance between vessels. This adds additional buffer between
                vessels to check for possible collisions.
    Returns
    -------
    proximity : bool
        A boolean that indicates whether the vessel is in proximity to
        other vessels in the tree.
    """
    clearance = kwargs.get('clearance', 0.0)
    tmp = deepcopy(vessel)
    tmp[0, 21] += clearance
    return sphere_proximity(data, tmp)


def tree_collision(data, vessel, **kwargs):
    """
    Check if a vessel has a collision with any other vessels in
    the current tree.

    Parameters
    ----------
    data : numpy.ndarray
        An array of vessel data.
    vessel : numpy.ndarray
        An array of vessel data.
    **kwargs : dict
        Additional keyword arguments.
        Keyword arguments include:
            clearance : float
                The clearance between vessels. This adds additional buffer between
                vessels to check for possible collisions.
    Returns
    -------
    has_collision : bool
        A boolean that indicates whether the vessel is in collision with
        any other vessels in the tree.
    """
    clearance = kwargs.get('clearance', 0.0)
    tmp = deepcopy(vessel)
    tmp[:, 21] += clearance
    has_collision = False
    for i in range(tmp.shape[0]):
        proximity = sphere_proximity(data[:, :], tmp[i, :])
        has_collision = obb_any(data[proximity, :], tmp[i, :].reshape(1, -1))
        if has_collision:
            break
    return has_collision


def tree_self_collision(data, vessel, **kwargs):
    """
    Check if a vessel has a collision with any other vessels in
    the current tree excluding vessels that the vessel is
    directly connected to.

    Parameters
    ----------
    data : numpy.ndarray
        An array of vessel data.
    vessel : numpy.ndarray
        An array of vessel data.
    **kwargs : dict
        Additional keyword arguments.
        Keyword arguments include:
            clearance : float
                The clearance between vessels. This adds additional buffer between
                vessels to check for possible collisions.
    Returns
    -------
    has_collision : bool
        A boolean that indicates whether the vessel is in collision with
        any other vessels in the tree.

    [TODO] This function is not complete. It needs to be tested and
           verified that it works as expected.
    """
    clearance = kwargs.get('clearance', 0.0)
    tmp = deepcopy(vessel)
    tmp[0, 21] += clearance
    proximity = sphere_proximity(data, tmp.flatten())
    if not numpy.isnan(tmp[0, 17]):
        parent = int(tmp[0, 17])
        proximity[int(tmp[0, parent])] = False
        if not numpy.isnan(data[parent, 17]):
            grandparent = int(data[parent, 17])
            if int(data[grandparent, 15]) == parent:
                sister = int(data[grandparent, 16])
                proximity[sister] = False
            else:
                sister = int(data[grandparent, 15])
                proximity[sister] = False
    if not numpy.isnan(tmp[0, 15]):
        proximity[int(tmp[0, 15])] = False
    if not numpy.isnan(tmp[0, 16]):
        proximity[int(tmp[0, 16])] = False
    has_collision = obb_any(data[proximity, :], tmp)
    return has_collision
