# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython
from libc.math cimport fabs

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef dot(double[:] a, double[:] b):
    cdef Py_ssize_t i = a.shape[0]
    cdef double res = 0.0
    for ii in range(i):
        res += a[ii] * b[ii]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef dot_inplace(double[:] a, double[:] b, double res):
    cdef Py_ssize_t i = a.shape[0]
    res = 0.0
    for ii in range(i):
        res += a[ii] * b[ii]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef cross(double[:] a, double[:] b, double[:] res):
    res[0] = a[1] * b[2] - a[2] * b[1]
    res[1] = a[2] * b[0] - a[0] * b[2]
    res[2] = a[0] * b[1] - a[1] * b[0]

#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)
#cdef int separating_axis(double[:] position, double[:] plane,
#                     double[:] u1, double[:] v1, double[:] w1,
#                     double[:] u2, double[:] v2, double[:] w2,
#                     double u1s, double v1s, double w1s,
#                     double u2s, double v2s, double w2s):
#    if fabs(dot(position, plane)) > (fabs(u1s*dot(u1, plane)) +
#                                        fabs(v1s*dot(v1, plane)) +
#                                        fabs(w1s*dot(w1, plane)) +
#                                        fabs(u2s*dot(u2, plane)) +
#                                        fabs(v2s*dot(v2, plane)) +
#                                        fabs(w2s*dot(w2, plane))):
#        return 1
#    else:
#        return 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int separating_axis(double[3] position, double[3] plane,
                         double[3] u1, double[3] v1, double[3] w1,
                         double[3] u2, double[3] v2, double[3] w2,
                         double u1s, double v1s, double w1s,
                         double u2s, double v2s, double w2s) nogil:
    cdef double lhs, rhs
    cdef double pos_dot_plane
    cdef double u1_dot_plane, v1_dot_plane, w1_dot_plane
    cdef double u2_dot_plane, v2_dot_plane, w2_dot_plane

    # Compute dot products directly
    pos_dot_plane = position[0]*plane[0] + position[1]*plane[1] + position[2]*plane[2]
    u1_dot_plane = u1[0]*plane[0] + u1[1]*plane[1] + u1[2]*plane[2]
    v1_dot_plane = v1[0]*plane[0] + v1[1]*plane[1] + v1[2]*plane[2]
    w1_dot_plane = w1[0]*plane[0] + w1[1]*plane[1] + w1[2]*plane[2]
    u2_dot_plane = u2[0]*plane[0] + u2[1]*plane[1] + u2[2]*plane[2]
    v2_dot_plane = v2[0]*plane[0] + v2[1]*plane[1] + v2[2]*plane[2]
    w2_dot_plane = w2[0]*plane[0] + w2[1]*plane[1] + w2[2]*plane[2]

    # Compute the left-hand side (lhs) and right-hand side (rhs) of the inequality
    lhs = fabs(pos_dot_plane)
    rhs = (fabs(u1s * u1_dot_plane) +
           fabs(v1s * v1_dot_plane) +
           fabs(w1s * w1_dot_plane) +
           fabs(u2s * u2_dot_plane) +
           fabs(v2s * v2_dot_plane) +
           fabs(w2s * w2_dot_plane))

    # Return the result based on the comparison
    if lhs > rhs:
        return 1
    else:
        return 0

"""
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def obb_any(double[:,:] data, double[:,:] vessels):
    
    This Oriented Bounding Box (OBB) collision detection algorithm is based on the Separating Axis Theorem (SAT).
    It is used to determine if any of the OBBs in the data array are colliding with any of the OBBs in the vessels array.
    This algorithm returns True at the first instance of a collision and returns False if, and only if, no collision
    exists among the OBBs in the data and vessel arrays.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array of shape (n, j) where n is the number of OBBs in the data array. The columns of the data
        array represent the 3D position, orientation, and half-widths of the OBBs in the data array.
    vessels : numpy.ndarray
        A 2D numpy array of shape (m, j) where m is the number of OBBs in the vessels array. The columns of the
        vessels array represent the 3D position, orientation, and half-widths of the OBBs in the vessels array.

    Returns
    -------
    has_collision : bool
        A boolean value indicating if any of the OBBs in the data array are colliding with any of the OBBs in the
        vessels array.
    
    cdef Py_ssize_t j = data.shape[0]
    cdef Py_ssize_t i = vessels.shape[0]
    cdef int k = 3
    cdef int kk
    #cdef cnp.ndarray[cnp.float64_t, ndim=1] center_1 = np.zeros(k, dtype=np.float64)
    #cdef cnp.ndarray[cnp.float64_t, ndim=1] center_2 = np.zeros(k, dtype=np.float64)
    #cdef cnp.ndarray[cnp.float64_t, ndim=1] position = np.zeros(k, dtype=np.float64)
    #cdef cnp.ndarray[cnp.float64_t, ndim=1] plane = np.zeros(k, dtype=np.float64)
    #cdef double[:] center_1_mv = center_1
    #cdef double[:] center_2_mv = center_2
    #cdef double[:] position_mv = position
    #cdef double[:] plane_mv = plane
    cdef double[3] center_1_mv = [0.0, 0.0, 0.0]
    cdef double[3] center_2_mv = [0.0, 0.0, 0.0]
    cdef double[3] position_mv = [0.0, 0.0, 0.0]
    cdef double[3] plane_mv = [0.0, 0.0, 0.0]
    for ii in range(i):
        for kk in range(k):
            center_1_mv[kk] = (vessels[ii, kk] + vessels[ii, kk + k]) / 2.0
        for jj in range(j):
            for kk in range(k):
                center_2_mv[kk] = (data[jj, kk] + data[jj, kk + k]) / 2.0
                position_mv[kk] = center_2_mv[kk] - center_1_mv[kk]
            if separating_axis(position_mv, vessels[ii,6:9], vessels[ii, 6:9], vessels[ii,9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            if separating_axis(position_mv, vessels[ii, 9:12], vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # W1 axis
            if separating_axis(position_mv, vessels[ii, 12:15], vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # U2 axis
            if separating_axis(position_mv, data[jj, 6:9], vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # V2 axis
            if separating_axis(position_mv, data[jj, 9:12], vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # W2 axis
            if separating_axis(position_mv, data[jj, 12:15], vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # Mixed plane Separation Tests
            # U1 X U2
            cross(vessels[ii, 6:9], data[jj, 6:9], plane_mv)
            if separating_axis(position_mv, plane_mv, vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # U1 X V2
            cross(vessels[ii, 6:9], data[jj, 9:12], plane_mv)
            if separating_axis(position_mv, plane_mv, vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # U1 X W2
            cross(vessels[ii, 6:9], data[jj, 12:15], plane_mv)
            if separating_axis(position_mv, plane_mv, vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # V1 X U2
            cross(vessels[ii, 9:12], data[jj, 6:9], plane_mv)
            if separating_axis(position_mv, plane_mv, vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # V1 X V2
            cross(vessels[ii, 9:12], data[jj, 9:12], plane_mv)
            if separating_axis(position_mv, plane_mv, vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # V1 X W2
            cross(vessels[ii, 9:12], data[jj, 12:15], plane_mv)
            if separating_axis(position_mv, plane_mv, vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # W1 X U2
            cross(vessels[ii, 12:15], data[jj, 6:9], plane_mv)
            if separating_axis(position_mv, plane_mv, vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # W1 X V2
            cross(vessels[ii, 12:15], data[jj, 9:12], plane_mv)
            if separating_axis(position_mv, plane_mv, vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            # W1 X W2
            cross(vessels[ii, 12:15], data[jj, 12:15], plane_mv)
            if separating_axis(position_mv, plane_mv, vessels[ii, 6:9], vessels[ii, 9:12], vessels[ii, 12:15],
                               data[jj, 6:9], data[jj, 9:12], data[jj, 12:15], vessels[ii, 21], vessels[ii, 21],
                               vessels[ii, 20] / 2, data[jj, 21], data[jj, 21], data[jj, 20] / 2):
                continue
            else:
                return True
    return False
"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def obb_any(double[:,:] data, double[:,:] vessels):
    """
    Oriented Bounding Box (OBB) collision detection using the Separating Axis Theorem (SAT).
    Returns True if any collision is detected between the OBBs in 'data' and 'vessels'.
    """
    cdef Py_ssize_t num_data = data.shape[0]
    cdef Py_ssize_t num_vessels = vessels.shape[0]
    cdef Py_ssize_t i_data, i_vessel
    cdef double[3] center1, center2, position
    cdef double[3] u1, v1, w1, u2, v2, w2
    cdef double u1s, v1s, w1s, u2s, v2s, w2s
    cdef double[3] cross_prod
    cdef int collision

    with nogil:
        for i_vessel in range(num_vessels):
            # Extract vessel OBB center
            center1[0] = (vessels[i_vessel, 0] + vessels[i_vessel, 3]) * 0.5
            center1[1] = (vessels[i_vessel, 1] + vessels[i_vessel, 4]) * 0.5
            center1[2] = (vessels[i_vessel, 2] + vessels[i_vessel, 5]) * 0.5

            # Extract vessel OBB axes
            u1[0] = vessels[i_vessel, 6]
            u1[1] = vessels[i_vessel, 7]
            u1[2] = vessels[i_vessel, 8]

            v1[0] = vessels[i_vessel, 9]
            v1[1] = vessels[i_vessel,10]
            v1[2] = vessels[i_vessel,11]

            w1[0] = vessels[i_vessel,12]
            w1[1] = vessels[i_vessel,13]
            w1[2] = vessels[i_vessel,14]

            # Vessel OBB half-sizes
            u1s = vessels[i_vessel, 21]
            v1s = vessels[i_vessel, 21]
            w1s = vessels[i_vessel, 20] * 0.5

            for i_data in range(num_data):
                # Extract data OBB center
                center2[0] = (data[i_data, 0] + data[i_data, 3]) * 0.5
                center2[1] = (data[i_data, 1] + data[i_data, 4]) * 0.5
                center2[2] = (data[i_data, 2] + data[i_data, 5]) * 0.5

                # Compute vector between centers
                position[0] = center2[0] - center1[0]
                position[1] = center2[1] - center1[1]
                position[2] = center2[2] - center1[2]

                # Extract data OBB axes
                u2[0] = data[i_data, 6]
                u2[1] = data[i_data, 7]
                u2[2] = data[i_data, 8]

                v2[0] = data[i_data, 9]
                v2[1] = data[i_data,10]
                v2[2] = data[i_data,11]

                w2[0] = data[i_data,12]
                w2[1] = data[i_data,13]
                w2[2] = data[i_data,14]

                # Data OBB half-sizes
                u2s = data[i_data, 21]
                v2s = data[i_data, 21]
                w2s = data[i_data, 20] * 0.5

                # Begin testing axes for separation
                # Test vessel axes
                if separating_axis(position, u1, u1, v1, w1, u2, v2, w2,
                                   u1s, v1s, w1s, u2s, v2s, w2s):
                    continue  # Separating axis found

                if separating_axis(position, v1, u1, v1, w1, u2, v2, w2,
                                   u1s, v1s, w1s, u2s, v2s, w2s):
                    continue

                if separating_axis(position, w1, u1, v1, w1, u2, v2, w2,
                                   u1s, v1s, w1s, u2s, v2s, w2s):
                    continue

                # Test data axes
                if separating_axis(position, u2, u1, v1, w1, u2, v2, w2,
                                   u1s, v1s, w1s, u2s, v2s, w2s):
                    continue

                if separating_axis(position, v2, u1, v1, w1, u2, v2, w2,
                                   u1s, v1s, w1s, u2s, v2s, w2s):
                    continue

                if separating_axis(position, w2, u1, v1, w1, u2, v2, w2,
                                   u1s, v1s, w1s, u2s, v2s, w2s):
                    continue

                # Test cross products of axes
                # U1 x U2
                cross_prod[0] = u1[1] * u2[2] - u1[2] * u2[1]
                cross_prod[1] = u1[2] * u2[0] - u1[0] * u2[2]
                cross_prod[2] = u1[0] * u2[1] - u1[1] * u2[0]

                if not (cross_prod[0] == 0 and cross_prod[1] == 0 and cross_prod[2] == 0):
                    if separating_axis(position, cross_prod, u1, v1, w1, u2, v2, w2,
                                       u1s, v1s, w1s, u2s, v2s, w2s):
                        continue

                # U1 x V2
                cross_prod[0] = u1[1] * v2[2] - u1[2] * v2[1]
                cross_prod[1] = u1[2] * v2[0] - u1[0] * v2[2]
                cross_prod[2] = u1[0] * v2[1] - u1[1] * v2[0]

                if not (cross_prod[0] == 0 and cross_prod[1] == 0 and cross_prod[2] == 0):
                    if separating_axis(position, cross_prod, u1, v1, w1, u2, v2, w2,
                                       u1s, v1s, w1s, u2s, v2s, w2s):
                        continue

                # U1 x W2
                cross_prod[0] = u1[1] * w2[2] - u1[2] * w2[1]
                cross_prod[1] = u1[2] * w2[0] - u1[0] * w2[2]
                cross_prod[2] = u1[0] * w2[1] - u1[1] * w2[0]

                if not (cross_prod[0] == 0 and cross_prod[1] == 0 and cross_prod[2] == 0):
                    if separating_axis(position, cross_prod, u1, v1, w1, u2, v2, w2,
                                       u1s, v1s, w1s, u2s, v2s, w2s):
                        continue

                # V1 x U2
                cross_prod[0] = v1[1] * u2[2] - v1[2] * u2[1]
                cross_prod[1] = v1[2] * u2[0] - v1[0] * u2[2]
                cross_prod[2] = v1[0] * u2[1] - v1[1] * u2[0]

                if not (cross_prod[0] == 0 and cross_prod[1] == 0 and cross_prod[2] == 0):
                    if separating_axis(position, cross_prod, u1, v1, w1, u2, v2, w2,
                                       u1s, v1s, w1s, u2s, v2s, w2s):
                        continue

                # V1 x V2
                cross_prod[0] = v1[1] * v2[2] - v1[2] * v2[1]
                cross_prod[1] = v1[2] * v2[0] - v1[0] * v2[2]
                cross_prod[2] = v1[0] * v2[1] - v1[1] * v2[0]

                if not (cross_prod[0] == 0 and cross_prod[1] == 0 and cross_prod[2] == 0):
                    if separating_axis(position, cross_prod, u1, v1, w1, u2, v2, w2,
                                       u1s, v1s, w1s, u2s, v2s, w2s):
                        continue

                # V1 x W2
                cross_prod[0] = v1[1] * w2[2] - v1[2] * w2[1]
                cross_prod[1] = v1[2] * w2[0] - v1[0] * w2[2]
                cross_prod[2] = v1[0] * w2[1] - v1[1] * w2[0]

                if not (cross_prod[0] == 0 and cross_prod[1] == 0 and cross_prod[2] == 0):
                    if separating_axis(position, cross_prod, u1, v1, w1, u2, v2, w2,
                                       u1s, v1s, w1s, u2s, v2s, w2s):
                        continue

                # W1 x U2
                cross_prod[0] = w1[1] * u2[2] - w1[2] * u2[1]
                cross_prod[1] = w1[2] * u2[0] - w1[0] * u2[2]
                cross_prod[2] = w1[0] * u2[1] - w1[1] * u2[0]

                if not (cross_prod[0] == 0 and cross_prod[1] == 0 and cross_prod[2] == 0):
                    if separating_axis(position, cross_prod, u1, v1, w1, u2, v2, w2,
                                       u1s, v1s, w1s, u2s, v2s, w2s):
                        continue

                # W1 x V2
                cross_prod[0] = w1[1] * v2[2] - w1[2] * v2[1]
                cross_prod[1] = w1[2] * v2[0] - w1[0] * v2[2]
                cross_prod[2] = w1[0] * v2[1] - w1[1] * v2[0]

                if not (cross_prod[0] == 0 and cross_prod[1] == 0 and cross_prod[2] == 0):
                    if separating_axis(position, cross_prod, u1, v1, w1, u2, v2, w2,
                                       u1s, v1s, w1s, u2s, v2s, w2s):
                        continue

                # W1 x W2
                cross_prod[0] = w1[1] * w2[2] - w1[2] * w2[1]
                cross_prod[1] = w1[2] * w2[0] - w1[0] * w2[2]
                cross_prod[2] = w1[0] * w2[1] - w1[1] * w2[0]

                if not (cross_prod[0] == 0 and cross_prod[1] == 0 and cross_prod[2] == 0):
                    if separating_axis(position, cross_prod, u1, v1, w1, u2, v2, w2,
                                       u1s, v1s, w1s, u2s, v2s, w2s):
                        continue
                with gil:
                    # No separating axis found; collision detected
                    return True
    # No collisions detected
    return False
