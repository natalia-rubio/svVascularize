# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython
from libc.math cimport sqrt, pow
from libcpp.vector cimport vector
from libcpp cimport bool

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef cross(vector[double] a, double[:] b):
    cdef vector[double] c = [0.0, 0.0, 0.0]
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def close_exact_points(double[:,:] data, double[:,:] points):
    cdef Py_ssize_t i = data.shape[0]
    cdef Py_ssize_t j = points.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] line_direction = np.zeros((i, 3), dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] closest_indices = np.zeros(j, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] parametric_values = np.zeros(j, dtype=np.float64)
    cdef vector[double] tmp = [0.0, 0.0, 0.0]
    cdef double[:,:] line_direction_mv = line_direction
    cdef double ss
    cdef double tt
    cdef double hh
    cdef double cd
    cdef double distance
    cdef vector[double] cc
    cdef Py_ssize_t closest_index
    cdef double t, t_closest
    for ii in range(i):
        distance = sqrt(pow(data[ii,3] - data[ii,0],2.0) + pow(data[ii,4] - data[ii,1],2.0) + pow(data[ii,5] - data[ii,2],2.0))
        line_direction_mv[ii,0] = (data[ii,3] - data[ii,0]) / distance
        line_direction_mv[ii,1] = (data[ii,4] - data[ii,1]) / distance
        line_direction_mv[ii,2] = (data[ii,5] - data[ii,2]) / distance
    for jj in range(j):
        min_distance = float('inf')
        for ii in range(i):
            ss = (line_direction_mv[ii,0]*(data[ii,0] - points[jj, 0])+
                  line_direction_mv[ii,1]*(data[ii,1] - points[jj, 1])+
                  line_direction_mv[ii,2]*(data[ii,2] - points[jj, 2]))
            tt = (line_direction_mv[ii,0]*(points[jj, 0] - data[ii,3])+
                  line_direction_mv[ii,1]*(points[jj, 1] - data[ii,4])+
                  line_direction_mv[ii,2]*(points[jj, 2] - data[ii,5]))
            hh = max(0.0, ss, tt)
            tmp[0] = points[jj, 0] - data[ii,0]
            tmp[1] = points[jj, 1] - data[ii,1]
            tmp[2] = points[jj, 2] - data[ii,2]
            cc = cross(tmp, line_direction_mv[ii,:])
            cd = sqrt(pow(cc[0],2.0) + pow(cc[1],2.0) + pow(cc[2],2.0))
            distance = sqrt(pow(hh,2.0) + pow(cd,2.0))
            if distance < min_distance:
                min_distance = distance
                closest_index = ii
                if (ss + tt) != 0:
                    t = ss / (ss + tt)
                else:
                    t = 0
                t_closest = t
        closest_indices[jj] = closest_index
        parametric_values[jj] = t_closest
    return closest_indices, parametric_values