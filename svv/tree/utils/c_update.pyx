# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython
#from cython.parallel import prange, parallel
from libc.math cimport pow, pi
from libcpp.vector cimport vector

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def update_resistance(double[:,:] data, long long[:,:] idx, double gamma, double nu):
    cdef double lr, lbif, rbif
    cdef Py_ssize_t size = data.shape[0]
    cdef vector[long long] vessels = range(size)
    cdef vector[long long] tmp_vessels
    cdef long long i
    max_depth = max(data[:, 26])
    while vessels.size() > 0:
        tmp_vessels.clear()
        for ii in range(vessels.size()):
            if data[vessels[ii], 26] != max_depth:
                tmp_vessels.push_back(vessels[ii])
            else:
                i = vessels[ii]
                if np.isnan(data[i, 15:17]).all():
                    data[i, 25] = (8 * nu / pi) * data[i, 20]
                    data[i, 27] = 0.0
                elif np.isnan(data[i, 15]):
                    right = idx[i, 1]
                    data[i, 25] = (8 * nu / pi) * data[i, 20] + data[right, 25]
                    data[i, 23] = 0.0
                    data[i, 24] = 1.0
                    data[i, 27] = data[right, 20] + data[right, 27]
                elif np.isnan(data[i, 16]):
                    left = idx[i, 0]
                    data[i, 25] = (8 * nu / pi) * data[i, 20] + data[left, 25]
                    data[i, 23] = 1.0
                    data[i, 24] = 0.0
                    data[i, 27] = data[left, 20] + data[left, 27]
                else:
                    left = idx[i, 0]
                    right = idx[i, 1]
                    lr = pow(((data[left, 22]*data[left, 25])/(data[right, 22]*data[right, 25])), 0.25)
                    lbif = pow((1.0 + pow(lr, -gamma)), -1.0/gamma)
                    rbif = pow((1.0 + pow(lr, gamma)), -1.0/gamma)
                    data[i, 25] = (8 * nu / pi) * data[i, 20] + pow(((pow(lbif, 4.0)/data[left, 25]) +
                                                                     (pow(rbif, 4.0)/data[right, 25])), -1.0)
                    data[i, 23] = lbif
                    data[i, 24] = rbif
                    data[i, 27] = pow(lbif, 2.0) * (data[left, 20] + data[left, 27]) + pow(rbif, 2.0) * (data[right, 20] + data[right, 27])
        vessels = tmp_vessels


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def update_radii(double[:,:] data, long long[:,:] idx, double root_pressure, double terminal_pressure):
    cdef vector[long long] left_vessels
    cdef vector[long long] right_vessels
    cdef vector[long long] tmp_left_vessels
    cdef vector[long long] tmp_right_vessels
    cdef long long i
    data[0, 21] = pow(((data[0, 25] * data[0, 22])/(root_pressure - terminal_pressure)), 0.25)
    if ~np.isnan(data[0, 15]):
        left_vessels.push_back(idx[0, 0])
    if ~np.isnan(data[0, 16]):
        right_vessels.push_back(idx[0, 1])
    while left_vessels.size() > 0 or right_vessels.size() > 0:
        tmp_left_vessels.clear()
        tmp_right_vessels.clear()
        for i in left_vessels:
            data[i, 21] = data[idx[i, 3], 23]*data[idx[i, 3], 21]
            if ~np.isnan(data[i, 15]):
                tmp_left_vessels.push_back(idx[i, 0])
            if ~np.isnan(data[i, 16]):
                tmp_right_vessels.push_back(idx[i, 1])
        for i in right_vessels:
            data[i, 21] = data[idx[i, 3], 24] * data[idx[i, 3], 21]
            if ~np.isnan(data[i, 15]):
                tmp_left_vessels.push_back(idx[i, 0])
            if ~np.isnan(data[i, 16]):
                tmp_right_vessels.push_back(idx[i, 1])
        left_vessels = tmp_left_vessels
        right_vessels = tmp_right_vessels

