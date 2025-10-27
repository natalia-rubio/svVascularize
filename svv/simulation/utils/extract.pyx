# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

cnp.import_array()

@cython.boundscheck(False)
def build_vertex_map(int[:,:] faces, int n_vertices):
    cdef Py_ssize_t i = faces.shape[0]
    cdef Py_ssize_t j = faces.shape[1]
    cdef vector[vector[int]] vertex_map
    vertex_map.resize(n_vertices)
    for ii in range(i):
        for jj in range(j):
            vertex_map[faces[ii,jj]].push_back(ii)
    return vertex_map

