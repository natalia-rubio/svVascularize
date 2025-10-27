# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython
from libc.math cimport sqrt, pow
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libc.math cimport acos, M_PI

cnp.import_array()

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def norm(double[:,:] data):
    cdef Py_ssize_t i = data.shape[0]
    cdef Py_ssize_t j = data.shape[1]
    cdef Py_ssize_t ii
    cdef cnp.ndarray[cnp.float64_t, ndim=2] magnitudes = np.zeros((i,1), dtype=np.float64)
    cdef double[:,:] mv = magnitudes
    cdef double mag
    for ii in range(i):
        mag = 0
        for jj in range(j):
            mag += pow(data[ii,jj], 2)
        mv[ii, 0] = sqrt(mag)
    return magnitudes

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def argwhere_nonzeros(double[:] data):
    cdef Py_ssize_t i = data.shape[0]
    cdef Py_ssize_t ii
    cdef vector[int] nonzeros
    for ii in range(i):
        if data[ii] != 0:
            nonzeros.push_back(ii)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] results = np.array(nonzeros, dtype=np.int64)
    return results

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def argwhere_value_double(double[:] data, double value):
    cdef Py_ssize_t i = data.shape[0]
    cdef Py_ssize_t ii
    cdef vector[int] zeros
    for ii in range(i):
        if data[ii] == value:
            zeros.push_back(ii)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] results = np.array(zeros, dtype=np.int64)
    return results

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def argwhere_value_int(long long[:] data, long long value):
    cdef Py_ssize_t i = data.shape[0]
    cdef Py_ssize_t ii
    cdef vector[int] zeros
    for ii in range(i):
        if data[ii] == value:
            zeros.push_back(ii)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] results = np.array(zeros, dtype=np.int64)
    return results


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def any_value_double(double[:] data, double value):
    cdef Py_ssize_t i = data.shape[0]
    cdef Py_ssize_t ii
    for ii in range(i):
        if data[ii] == value:
            return True
    return False


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def any_value_int(int[:] data, int value):
    cdef Py_ssize_t i = data.shape[0]
    cdef Py_ssize_t ii
    for ii in range(i):
        if data[ii] == value:
            return True
    return False


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def c_dict(double[:,:] data):
    cdef Py_ssize_t i = data.shape[0]
    cdef Py_ssize_t j = data.shape[1]
    cdef Py_ssize_t ii, jj
    cdef unordered_map[int, vector[double]] dictionary
    cdef int key
    cdef vector[double] value
    for ii in range(i):
        key = ii
        for jj in range(j):
            value.push_back(data[ii,jj])
        dictionary[key] = value
        value.clear()
    return dictionary


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def duplicate_map(long long[:] unique_inverse, long long[:] unique_counts):
    cdef Py_ssize_t i = unique_counts.shape[0]
    cdef Py_ssize_t j = unique_inverse.shape[0]
    cdef Py_ssize_t ii, jj
    cdef unordered_map[int, vector[int]] duplicate_dict
    cdef unordered_set[int] duplicate_set
    cdef int key
    cdef vector[int] value
    for ii in range(i):
        if unique_counts[ii] <= 1:
            continue
        else:
            for jj in range(j):
                if unique_inverse[jj] == ii:
                    value.push_back(jj)
                    duplicate_set.insert(jj)
        duplicate_dict[ii] = value
        value.clear()
    return duplicate_dict, duplicate_set

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def _allocate_patch(long long[:] indices, double overlap, unordered_set[int] point_set, unordered_set[int] duplicates_set):
    """"
    Internal helper function for patch point allocation.
    This function takes a neighborhood points, the
    current remaining set of points, the set of
    duplicate points, and the overlap ratio
    and returns the set of points to be removed.
    """
    cdef Py_ssize_t i = indices.shape[0]
    cdef Py_ssize_t overlap_indicies = <Py_ssize_t>(i * overlap)
    for i in range(1, overlap_indicies):
        if point_set.count(indices[i]):
            if duplicates_set.size() > 0:
                if not duplicates_set.count(indices[i]):
                    point_set.erase(indices[i])
            else:
                point_set.erase(indices[i])
    return point_set

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def closest_point(long long index, unordered_set[int] included, double[:,:] points):
    cdef double distance
    cdef double min_distance = 0.0
    cdef int closest_index
    cdef Py_ssize_t i = points.shape[0]
    cdef Py_ssize_t j = points.shape[1]
    for ii in included:
        distance = 0.0
        for jj in range(j):
            distance += pow(points[index, jj] - points[ii, jj], 2)
        distance = sqrt(distance)
        if distance > min_distance and min_distance == 0.0:
            min_distance = distance
            closest_index = ii
        elif distance == min_distance:
            closest_index = ii
        elif distance > min_distance:
            continue
        else:
            min_distance = distance
            closest_index = ii
    return closest_index, min_distance

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def get_angle(int index_1, int index_2, double[:,:] normals):
    cdef double angle = 0.0
    cdef Py_ssize_t j = normals.shape[1]
    cdef Py_ssize_t jj
    for jj in range(j):
        angle = angle + normals[index_1, jj] * normals[index_2, jj]
    angle = acos(angle) * (180 / M_PI)
    return angle

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def _allocate_angle(long long idx, long long[:] indices, double[:,:] points, double[:,:] normals, double feature_angle):
    cdef Py_ssize_t i = points.shape[0]
    cdef Py_ssize_t j = points.shape[1]
    cdef unordered_set[int] include_indices
    cdef unordered_set[int] include_points
    cdef vector[int] allocated_indices
    for ii in range(i):
        if indices[ii] == idx:
            include_points.insert(ii)
            include_indices.insert(indices[ii])
            allocated_indices.push_back(indices[ii])
    cdef int closest_idx
    cdef double closest_distance
    for ii in range(1, i):
        closest_idx, closest_distance = closest_point(ii, include_points, points)
        if closest_distance == 0.0:
            continue
        angle = get_angle(ii, closest_idx, normals)
        if angle > feature_angle:
            continue
        else:
            include_points.insert(ii)
            include_indices.insert(indices[ii])
            allocated_indices.push_back(indices[ii])
    return allocated_indices