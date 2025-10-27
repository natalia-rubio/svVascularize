# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython
from libc.math cimport sqrt, pow
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def build_c_vessel_map(object vessel_map):
    cdef unordered_map[int, unordered_map[string, vector[int]]] c_map
    cdef vector[int] item
    cdef int key
    for key in vessel_map.keys():
        item = vessel_map[key]["downstream"]
        c_map[key][b"downstream"] = item
        item = vessel_map[key]["upstream"]
        c_map[key][b"upstream"] = item
    return c_map


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def update_alt(
    double[:] reduced_resistance,
    double[:] reduced_length,
    vector[int] main_idx,
    double[:] main_scale,
    vector[int] alt_idx,
    double[:] alt_scale,
    double[:, :] bifurcation_ratios,
    double[:] flows,
    double root_radius,
    double[:, :] data,
    double[:] tmp_28,
    object vessel_map  # still a Python dict of dicts/lists
):
    """
    1) We do a first pass to estimate how many total pushes we'll do
       so we can reserve() capacity in each std::vector.
    2) We cache Python dictionary lookups (vessel_map[idx]['downstream'])
       in a local variable to reduce overhead in the inner loop.
    """

    # Temporary vectors where we push back changes
    cdef vector[int] tmp_change_i
    cdef vector[int] tmp_change_j
    cdef vector[double] tmp_new_data
    cdef vector[double] tmp_old_data

    cdef int main_idx_size = len(main_idx)
    cdef int alt_idx_size = len(alt_idx)
    cdef vector[int] downstream_list
    # Local variables for loop indices & calculations
    cdef int i, j, idx, downstream_idx
    cdef int downstream_len
    cdef double old_val, new_val, scale_factor

    #------------------------------------------------
    # 1) First pass: estimate total # of pushes
    #------------------------------------------------
    # We'll do one push for the "root_radius"
    cdef int total_expected = 1

    # For each main_idx:
    #   flows(1) + reduced_resistance(1) + reduced_length(1) + radius_scaling(1) + 2 bifurcation = 6 changes
    #   actually 7 because flows is separate. Let's count carefully:
    #     - flows -> 1
    #     - reduced_resistance -> 1
    #     - reduced_length -> 1
    #     - radius scaling -> 1
    #     - bifurcation_ratios -> 2
    #   => 1 + 1 + 1 + 1 + 2 = 6
    # But we also push the "tmp_28" update to the vector?
    # Actually we do the "tmp_28" assignment but that doesn't get push_back-ed.
    # So the # of changes is indeed 1+1+1+1+2 = 6 for main.
    # Let's verify your code: you do 1 push for flows, 1 for res, 1 for length, 1 for scaling, 2 for bifact => 6 total.
    # We'll go with 6 per main_idx.
    cdef int pushes_per_main = 6
    total_expected += main_idx_size * pushes_per_main

    # For each alt_idx: we do 1 push for alt_idx’s own scale, plus one for *each downstream* vessel.
    cdef int sum_downstream = 0
    cdef int alt_down_len
    for i in range(alt_idx_size):
        # It's still a Python dict/list, so to estimate the length, we must do one dictionary lookup:
        alt_down_len = len(vessel_map[alt_idx[i]]['downstream'])
        sum_downstream += alt_down_len
    # So alt_idx_size pushes for the alt vessels themselves,
    # plus sum_downstream pushes for all downstream vessels
    total_expected += alt_idx_size + sum_downstream

    # Now reserve capacity so the vectors won't repeatedly reallocate
    tmp_change_i.reserve(total_expected)
    tmp_change_j.reserve(total_expected)
    tmp_new_data.reserve(total_expected)
    tmp_old_data.reserve(total_expected)

    #------------------------------------------------
    # 2) Populate the vectors (main & alt vessels)
    #------------------------------------------------

    # Push initial "root_radius" entry
    tmp_change_i.push_back(0)
    tmp_change_j.push_back(21)
    tmp_new_data.push_back(root_radius)
    tmp_old_data.push_back(data[0, 21])

    # ----------- MAIN VESSELS -----------
    for i in range(main_idx_size):
        idx = main_idx[i]

        # flows
        tmp_change_i.push_back(idx)
        tmp_change_j.push_back(22)
        tmp_new_data.push_back(flows[i])
        tmp_old_data.push_back(data[idx, 22])

        # reduced_resistance
        tmp_change_i.push_back(idx)
        tmp_change_j.push_back(25)
        tmp_new_data.push_back(reduced_resistance[i])
        tmp_old_data.push_back(data[idx, 25])

        # reduced_length
        tmp_change_i.push_back(idx)
        tmp_change_j.push_back(27)
        tmp_new_data.push_back(reduced_length[i])
        tmp_old_data.push_back(data[idx, 27])

        # radius scaling
        tmp_change_i.push_back(idx)
        tmp_change_j.push_back(28)
        tmp_new_data.push_back(main_scale[i])
        tmp_old_data.push_back(data[idx, 28])

        # bifurcation_ratios: two pushes
        tmp_change_i.push_back(idx)
        tmp_change_j.push_back(23)
        tmp_new_data.push_back(bifurcation_ratios[i, 0])
        tmp_old_data.push_back(data[idx, 23])

        tmp_change_i.push_back(idx)
        tmp_change_j.push_back(24)
        tmp_new_data.push_back(bifurcation_ratios[i, 1])
        tmp_old_data.push_back(data[idx, 24])

        # Update tmp_28 array
        tmp_28[idx] = main_scale[i]

    # ----------- ALTERNATE VESSELS -----------
    for i in range(alt_idx_size):
        idx = alt_idx[i]
        if not idx > -1.0:
            continue
        # Precompute the scale factor once
        scale_factor = alt_scale[i] / data[idx, 28]

        # Minimize repeated Python dict lookups by caching
        downstream_list = vessel_map[idx]['downstream']
        downstream_len = downstream_list.size()

        # Update all downstream vessels
        for j in range(downstream_len):
            downstream_idx = downstream_list[j]
            tmp_28[downstream_idx] *= scale_factor

            old_val = data[downstream_idx, 28]
            new_val = old_val * scale_factor

            tmp_change_i.push_back(downstream_idx)
            tmp_change_j.push_back(28)
            tmp_new_data.push_back(new_val)
            tmp_old_data.push_back(old_val)

        # Update the alt vessel itself
        tmp_28[idx] = alt_scale[i]

        tmp_change_i.push_back(idx)
        tmp_change_j.push_back(28)
        tmp_new_data.push_back(alt_scale[i])
        tmp_old_data.push_back(data[idx, 28])

    return tmp_change_i, tmp_change_j, tmp_new_data, tmp_old_data, tmp_28
