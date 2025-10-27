# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython
from libc.math cimport sqrt, pow
from libcpp.vector cimport vector

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef norm(vector[double] difference):
    cdef double result = 0.0
    for ii in range(difference.size()):
        result += pow(difference[ii], 2)
    return sqrt(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef norm_inplace(double[:] difference, double[:] result):
    result[0] = 0.0
    for ii in range(3):
        result[0] += pow(difference[ii], 2)
    result[0] = sqrt(result[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef diff(double[:] point_0, double[:] point_1):
    cdef vector[double] result = [0.0, 0.0, 0.0]
    result[0] = point_0[0] - point_1[0]
    result[1] = point_0[1] - point_1[1]
    result[2] = point_0[2] - point_1[2]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef diff_inplace(double[:] point_0, double[:] point_1, double[:] result):
    result[0] = point_0[0] - point_1[0]
    result[1] = point_0[1] - point_1[1]
    result[2] = point_0[2] - point_1[2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def basis(double[:, :] point_0, double[:, :] point_1):
    cdef Py_ssize_t i = point_0.shape[0]
    cdef Py_ssize_t j = point_0.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] u_basis = np.zeros((i, j), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] v_basis = np.zeros((i, j), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] w_basis = np.zeros((i, j), dtype=np.float64)
    cdef double[:,:] u_mv = u_basis
    cdef double[:,:] v_mv = v_basis
    cdef double[:,:] w_mv = w_basis
    cdef vector[double] direction
    cdef double magnitude
    for ii in range(i):
        tmp = diff(point_1[ii,:], point_0[ii,:])
        magnitude = norm(tmp)
        for jj in range(j):
            w_mv[ii, jj] = tmp[jj] / magnitude
        if w_mv[ii, 2] == -1.0:
            u_mv[ii, 0] = -1.0
            v_mv[ii, 1] = -1.0
        else:
            u_mv[ii, 0] = 1.0 - pow(w_mv[ii, 0], 2.0) / (1.0 + w_mv[ii, 2])
            u_mv[ii, 1] = (-w_mv[ii, 0] * w_mv[ii, 1]) / (1.0 + w_mv[ii, 2])
            u_mv[ii, 2] = -w_mv[ii, 0]
            v_mv[ii, 0] = (-w_mv[ii, 0] * w_mv[ii, 1]) / (1.0 + w_mv[ii, 2])
            v_mv[ii, 1] = 1.0 - pow(w_mv[ii, 1], 2.0) / (1.0 + w_mv[ii, 2])
            v_mv[ii, 2] = -w_mv[ii, 1]
    return u_basis, v_basis, w_basis

@cython.cpow(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def basis_inplace(double[:, :] point_0, double[:, :] point_1, double[:, :] u_mv, double[:,:] v_mv, double[:, :] w_mv):
    cdef Py_ssize_t i = point_0.shape[0]
    cdef Py_ssize_t j = point_0.shape[1]
    #cdef cnp.ndarray[cnp.float64_t, ndim=2] u_basis = np.zeros((i, j), dtype=np.float64)
    #cdef cnp.ndarray[cnp.float64_t, ndim=2] v_basis = np.zeros((i, j), dtype=np.float64)
    #cdef cnp.ndarray[cnp.float64_t, ndim=2] w_basis = np.zeros((i, j), dtype=np.float64)
    #cdef double[:,:] u_mv = u_basis
    #cdef double[:,:] v_mv = v_basis
    #cdef double[:,:] w_mv = w_basis
    cdef double[3] direction = [0.0, 0.0, 0.0]
    #cdef vector[double] tmp = [1.0, 1.0, 1.0]
    cdef double magnitude = 1.0
    for ii in range(i):
        #diff_inplace(point_1[ii,:], point_0[ii,:], u_mv[ii,:])
        #norm_inplace(u_mv[ii, :], v_mv[ii, :])
        direction[0] = point_1[ii, 0] - point_0[ii, 0]
        direction[1] = point_1[ii, 1] - point_0[ii, 1]
        direction[2] = point_1[ii, 2] - point_0[ii, 2]
        magnitude = sqrt(pow(direction[0],2.0) + pow(direction[1],2.0) + pow(direction[2],2.0))
        #magnitude = v_mv[ii, 0]
        for jj in range(j):
            w_mv[ii, jj] = direction[jj] / magnitude
        if w_mv[ii, 2] == -1.0:
            u_mv[ii, 0] = -1.0
            v_mv[ii, 1] = -1.0
        else:
            u_mv[ii, 0] = 1.0 - pow(w_mv[ii, 0], 2.0) / (1.0 + w_mv[ii, 2])
            u_mv[ii, 1] = (-w_mv[ii, 0] * w_mv[ii, 1]) / (1.0 + w_mv[ii, 2])
            u_mv[ii, 2] = -w_mv[ii, 0]
            v_mv[ii, 0] = (-w_mv[ii, 0] * w_mv[ii, 1]) / (1.0 + w_mv[ii, 2])
            v_mv[ii, 1] = 1.0 - pow(w_mv[ii, 1], 2.0) / (1.0 + w_mv[ii, 2])
            v_mv[ii, 2] = -w_mv[ii, 1]