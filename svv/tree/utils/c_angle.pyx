# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython
from libc.math cimport acos, pi
from libc.math cimport sqrt
from libcpp.vector cimport vector

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef dot(double[:] vector1, double[:] vector2):
    """
    Returns the dot product of two vectors.
    """
    cdef int i
    cdef double result = 0
    for i in range(vector1.shape[0]):
        result += vector1[i] * vector2[i]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def get_angles(double[:,:] vector1, double[:,:] vector2):
    """
    Returns the angle in degrees between the
    first and second vector.
    """
    cdef Py_ssize_t i = vector1.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] angles = np.zeros(i, dtype=np.float64)
    cdef angles_mv = angles
    cdef dotval, norm_u, norm_v, x, denom, ratio

    for ii in range(i):
        # dot(u,v)
        dotval = dot(vector1[ii,:], vector2[ii,:])

        # norm(u)
        norm_u = 0
        norm_v = 0
        # compute norms manually
        for x in vector1[ii,:]:
            norm_u += x*x
        for x in vector2[ii,:]:
            norm_v += x*x
        norm_u = sqrt(norm_u)
        norm_v = sqrt(norm_v)

        # protect against zero norm or out-of-bounds
        denom = norm_u * norm_v
        if denom < 1e-15:  # or some small epsilon to avoid division by zero
            angles_mv[ii] = 0.0  # or handle it gracefully
        else:
            # clamp the ratio to [-1, 1] to avoid numerical issues
            ratio = dotval / denom
            if ratio > 1.0:
                ratio = 1.0
            elif ratio < -1.0:
                ratio = -1.0

            angles_mv[ii] = acos(ratio) * (180.0 / pi)

    return angles