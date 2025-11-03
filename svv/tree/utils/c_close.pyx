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
@cython.cdivision(True)
def sphere_proximity(double[:,:] data, double[:] vessels):
    cdef Py_ssize_t i = data.shape[0]
    cdef vector[bool] proximity
    cdef vector[double] center_0 = [0.0, 0.0, 0.0]
    cdef vector[double] center_1 = [0.0, 0.0, 0.0]
    cdef vector[double] diff = [0.0, 0.0, 0.0]
    cdef double radius_0 = 0.0
    cdef double radius_1 = 0.0
    cdef double distance = 0.0
    for ii in range(i):
        center_0[0] = (data[ii,0] + data[ii,3]) / 2.0
        center_0[1] = (data[ii,1] + data[ii,4]) / 2.0
        center_0[2] = (data[ii,2] + data[ii,5]) / 2.0
        center_1[0] = (vessels[0] + vessels[3]) / 2.0
        center_1[1] = (vessels[1] + vessels[4]) / 2.0
        center_1[2] = (vessels[2] + vessels[5]) / 2.0
        radius_0 = sqrt(pow(data[ii, 21], 2.0) + pow(data[ii, 20] / 2.0, 2.0))
        radius_1 = sqrt(pow(vessels[21], 2.0) + pow(vessels[20] / 2.0, 2.0))
        diff[0] = center_0[0] - center_1[0]
        diff[1] = center_0[1] - center_1[1]
        diff[2] = center_0[2] - center_1[2]
        distance = sqrt(pow(diff[0], 2.0) + pow(diff[1], 2.0) + pow(diff[2], 2.0))
        if distance < (radius_0 + radius_1):
            proximity.push_back(True)
        else:
            proximity.push_back(False)
    return proximity


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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def close(double[:,:] data, double[:] point, int n=20):
    cdef Py_ssize_t i = data.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] distances = np.zeros(i, dtype=np.float64)
    cdef double tmp[3]
    cdef double[:] distances_mv = distances
    if n > i:
        n = i
    for ii in range(i):
        tmp[0] = (data[ii,0] + data[ii,3])/2.0 - point[0]
        tmp[1] = (data[ii,1] + data[ii,4])/2.0 - point[1]
        tmp[2] = (data[ii,2] + data[ii,5])/2.0 - point[2]
        distances_mv[ii] = sqrt(pow(tmp[0],2) + pow(tmp[1],2) + pow(tmp[2],2))
    vessels = np.argsort(distances_mv)
    return vessels[:n], distances[:n]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def close_exact_point_sort(double[:,:] data, double[:] point):
    cdef Py_ssize_t i = data.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] line_direction = np.zeros((i,3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] line_distance = np.zeros(i, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] vessels = np.zeros(i, dtype=np.int64)
    cdef vector[double] tmp = [0.0, 0.0, 0.0]
    cdef double[:,:] line_direction_mv = line_direction
    cdef double[:] line_distance_mv = line_distance
    cdef long long[:] vessels_mv = vessels
    cdef double ss
    cdef double tt
    cdef double hh
    cdef double cd
    cdef vector[double] cc
    line_direction_mv[:,:] = data[:,12:15]
    for ii in range(i):
        ss = (line_direction_mv[ii,0]*(data[ii,0] - point[0])+
              line_direction_mv[ii,1]*(data[ii,1] - point[1])+
              line_direction_mv[ii,2]*(data[ii,2] - point[2]))
        tt = (line_direction_mv[ii,0]*(point[0] - data[ii,3])+
              line_direction_mv[ii,1]*(point[1] - data[ii,4])+
              line_direction_mv[ii,2]*(point[2] - data[ii,5]))
        hh = max(0.0, ss, tt)
        tmp[0] = point[0] - data[ii,0]
        tmp[1] = point[1] - data[ii,1]
        tmp[2] = point[2] - data[ii,2]
        cc = cross(tmp, line_direction_mv[ii,:])
        cd = sqrt(pow(cc[0],2) + pow(cc[1],2) + pow(cc[2],2))
        line_distance_mv[ii] = sqrt(pow(hh,2) + pow(cd,2))
    vessels = np.argsort(line_distance_mv)
    return vessels, line_distance

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def close_exact_point(double[:,:] data, double[:] point):
    cdef Py_ssize_t i = data.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] line_direction = np.zeros((i,3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] line_distance = np.zeros(i, dtype=np.float64)
    cdef vector[double] tmp = [0.0, 0.0, 0.0]
    cdef double[:,:] line_direction_mv = line_direction
    cdef double[:] line_distance_mv = line_distance
    cdef double ss
    cdef double tt
    cdef double hh
    cdef double cd
    cdef vector[double] cc
    line_direction_mv[:,:] = data[:,12:15]
    for ii in range(i):
        ss = (line_direction_mv[ii,0]*(data[ii,0] - point[0])+
              line_direction_mv[ii,1]*(data[ii,1] - point[1])+
              line_direction_mv[ii,2]*(data[ii,2] - point[2]))
        tt = (line_direction_mv[ii,0]*(point[0] - data[ii,3])+
              line_direction_mv[ii,1]*(point[1] - data[ii,4])+
              line_direction_mv[ii,2]*(point[2] - data[ii,5]))
        hh = max(0.0, ss, tt)
        tmp[0] = point[0] - data[ii,0]
        tmp[1] = point[1] - data[ii,1]
        tmp[2] = point[2] - data[ii,2]
        cc = cross(tmp, line_direction_mv[ii,:])
        cd = sqrt(pow(cc[0],2.0) + pow(cc[1],2.0) + pow(cc[2],2.0))
        line_distance_mv[ii] = sqrt(pow(hh,2.0) + pow(cd,2.0))
    return line_distance

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def close_exact_points(double[:,:] data, double[:,:] points):
    cdef Py_ssize_t i = data.shape[0]
    cdef Py_ssize_t j = points.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] line_direction = np.zeros((i,3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] line_distance = np.zeros((i, j), dtype=np.float64)
    cdef vector[double] tmp = [0.0, 0.0, 0.0]
    cdef double[:,:] line_direction_mv = line_direction
    cdef double[:,:] line_distance_mv = line_distance
    cdef double ss
    cdef double tt
    cdef double hh
    cdef double cd
    cdef vector[double] cc
    line_direction_mv[:,:] = data[:,12:15]
    for ii in range(i):
        for jj in range(j):
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
            line_distance_mv[ii,jj] = sqrt(pow(hh,2.0) + pow(cd,2.0)) - data[ii, 21]
    return line_distance

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def close_exact_points_sort(double[:,:] data, double[:,:] points):
    cdef Py_ssize_t i = data.shape[0]
    cdef Py_ssize_t j = points.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] line_direction = np.zeros((i,3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] line_distance = np.zeros((i, j), dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] vessels = np.zeros((i, j), dtype=np.int64)
    cdef vector[double] tmp = [0.0, 0.0, 0.0]
    cdef double[:,:] line_direction_mv = line_direction
    cdef double[:,:] line_distance_mv = line_distance
    cdef double ss
    cdef double tt
    cdef double hh
    cdef double cd
    cdef vector[double] cc
    line_direction_mv[:,:] = data[:,12:15]
    for ii in range(i):
        for jj in range(j):
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
            cd = sqrt(pow(cc[0],2) + pow(cc[1],2) + pow(cc[2],2))
            line_distance_mv[ii,jj] = sqrt(pow(hh,2) + pow(cd,2)) - data[ii, 21]
    vessels = np.argsort(line_distance_mv, axis=0)
    return vessels, line_distance