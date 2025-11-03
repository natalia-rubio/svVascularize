# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def pick_from_tetrahedron(double[:,:,:] simplicies, double[:,:,:] rdx):
    cdef Py_ssize_t i = simplicies.shape[0]
    cdef cnp.ndarray[double, ndim=2] points = np.zeros((i, 3), dtype=np.float64)
    cdef double tmp
    for ii in range(i):
        if rdx[ii, 1, 0] +  rdx[ii, 2, 0] > 1:
            rdx[ii, 1, 0] = 1.0 - rdx[ii, 1, 0]
            rdx[ii, 2, 0] = 1.0 - rdx[ii, 2, 0]
        if rdx[ii, 2, 0] + rdx[ii, 3, 0] > 1:
            tmp = rdx[ii, 3, 0]
            rdx[ii, 3, 0] = 1.0 - rdx[ii, 1, 0] - rdx[ii, 2, 0]
            rdx[ii, 2, 0] = 1.0 - tmp
        elif rdx[ii, 1, 0] + rdx[ii, 2, 0] + rdx[ii, 3, 0] > 1:
            tmp = rdx[ii, 3, 0]
            rdx[ii, 3, 0] = rdx[ii, 1, 0] + rdx[ii, 2, 0] + rdx[ii, 3, 0] - 1.0
            rdx[ii, 1, 0] = 1.0 - rdx[ii, 2, 0] - tmp
        rdx[ii, 0, 0] = 1.0 - rdx[ii, 1, 0] - rdx[ii, 2, 0] - rdx[ii, 3, 0]
    points = np.sum(np.multiply(rdx, simplicies), axis=1)
    return points


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def pick_from_triangle(double[:,:,:] simplices, double[:,:,:] rdx):
    cdef Py_ssize_t i = simplices.shape[0]
    cdef cnp.ndarray[double, ndim=2] points = np.zeros((i, 3), dtype=np.float64)
    for ii in range(i):
        if rdx[ii, 1, 0] +  rdx[ii, 2, 0] > 1:
            rdx[ii, 1, 0] = 1.0 - rdx[ii, 1, 0]
            rdx[ii, 2, 0] = 1.0 - rdx[ii, 2, 0]
        rdx[ii, 0, 0] = 1.0 - rdx[ii, 1, 0] - rdx[ii, 2, 0]
    points = np.sum(np.multiply(rdx, simplices), axis=1)
    return points

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def pick_from_line(double[:,:,:] simplices, double[:,:,:] rdx):
    cdef Py_ssize_t i = simplices.shape[0]
    cdef cnp.ndarray[double, ndim=2] points = np.zeros((i, 3), dtype=np.float64)
    for ii in range(i):
        rdx[ii, 0, 0] = 1.0 - rdx[ii, 1, 0]
    points = np.sum(np.multiply(rdx, simplices), axis=1)
    return points