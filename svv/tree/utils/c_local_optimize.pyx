# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython
from libc.math cimport sqrt, pow, M_PI
from libcpp.vector cimport vector
from libc.math cimport HUGE_VAL
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string


cnp.import_array()

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double sqr(double x) nogil:
    return x * x


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double length_3d(double[:] p0, double[:] p1) nogil:
    cdef double dx = p0[0] - p1[0]
    cdef double dy = p0[1] - p1[1]
    cdef double dz = p0[2] - p1[2]
    return sqrt(dx*dx + dy*dy + dz*dz)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cdef inline double norm3(double dx, double dy, double dz) nogil:
    return sqrt(dx*dx + dy*dy + dz*dz)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef norm(vector[double] difference):
    cdef double result = 0.0
    for ii in range(difference.size()):
        result += pow(difference[ii], 2.0)
    return sqrt(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef inline void diff(double[:] point_0, double[:] point_1, double[3] result):
    result[0] = point_0[0] - point_1[0]
    result[1] = point_0[1] - point_1[1]
    result[2] = point_0[2] - point_1[2]

@cython.cpow(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef length(double[:] point_0, double[:] point_1, double result):
    result = sqrt(pow(point_0[0] - point_1[0], 2.0) + pow(point_0[1] - point_1[1], 2.0) + pow(point_0[2] - point_1[2], 2.0))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef basis(double[:] point_0, double[:] point_1):
    cdef vector[double] u_basis = [1.0, 1.0, 1.0]
    cdef vector[double] v_basis = [1.0, 1.0, 1.0]
    cdef vector[double] w_basis = [1.0, 1.0, 1.0]
    cdef double[3] tmp
    cdef double magnitude
    diff(point_1[:], point_0[:], tmp)
    magnitude = norm3(tmp[0], tmp[1], tmp[2])
    w_basis[0] = tmp[0] / magnitude
    w_basis[1] = tmp[1] / magnitude
    w_basis[2] = tmp[2] / magnitude
    if w_basis[2] == -1.0:
        u_basis[0] = -1.0
        v_basis[1] = -1.0
    else:
        u_basis[0] = 1.0 - pow(w_basis[0], 2.0) / (1.0 + w_basis[2])
        u_basis[1] = (-w_basis[0] * w_basis[1]) / (1.0 + w_basis[2])
        u_basis[2] = -w_basis[0]
        v_basis[0] = (-w_basis[0] * w_basis[1]) / (1.0 + w_basis[2])
        v_basis[1] = 1.0 - pow(w_basis[1], 2.0) / (1.0 + w_basis[2])
        v_basis[2] = -w_basis[1]
    return u_basis, v_basis, w_basis

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def basis3(double[:] point_0, double[:] point_1):
    """
    Given two 3D points, returns (u_basis, v_basis, w_basis),
    each a 3-element Python list describing orthonormal directions.
    """
    #cdef double dx = point_1[0] - point_0[0]
    #cdef double dy = point_1[1] - point_0[1]
    #cdef double dz = point_1[2] - point_0[2]
    cdef double dx
    cdef double dy
    cdef double dz
    cdef double[3] difference
    diff(point_0, point_1, difference)
    dx = difference[0]
    dy = difference[1]
    dz = difference[2]
    cdef double mag = norm3(difference[0], difference[1], difference[2])
    cdef double denom
    # Allocate local arrays for u, v, w
    cdef double u[3]
    cdef double v[3]
    cdef double w[3]

    # w = (dx, dy, dz) normalized
    w[0] = dx / mag
    w[1] = dy / mag
    w[2] = dz / mag

    if w[2] == -1.0:
        # The "special case" in your logic
        u[0] = -1.0
        u[1] =  0.0
        u[2] =  0.0
        v[0] =  0.0
        v[1] = -1.0
        v[2] =  0.0
    else:
        # Normal case
        # u_basis[0] = 1.0 - (w_basis[0]^2 / (1.0 + w_basis[2]))
        # etc...
        # We'll do it directly with multiplications:
        denom = 1.0 + w[2]

        u[0] = 1.0 - (w[0]*w[0] / denom)
        u[1] = -(w[0]*w[1]) / denom
        u[2] = -w[0]

        v[0] = -(w[0]*w[1]) / denom
        v[1] = 1.0 - (w[1]*w[1] / denom)
        v[2] = -w[1]

    # Convert to Python objects
    # (If performance is critical and you only use them in Cython,
    #  consider returning them as memoryviews or c arrays.)
    return ([u[0], u[1], u[2]],
            [v[0], v[1], v[2]],
            [w[0], w[1], w[2]])


@cython.cpow(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def bifurcation_cost(double[:] point, double[:,:] data, double[:] terminal, long[:,:] idxs,
                     int vessel, double gamma, double nu, double terminal_flow, double terminal_pressure,
                     double root_pressure, double radius_exponent, double length_exponent):
    cdef vector[double] reduced_resistances
    cdef vector[double] reduced_lengths
    cdef vector[double] bifurcations
    cdef vector[double] flows
    cdef vector[int] main_idx
    cdef vector[int] alt_idx
    cdef vector[double] new_scale # this replaces the data at column index 28 along the main_idx
    cdef vector[double] alt_scale # this replaces the data at column index 28 along the alt_idx
    cdef double upstream_length, downstream_length, terminal_length, r_terminal,\
                r_terminal_sister, f_terminal, f_terminal_sister, r0, l0, f_changed, f_stagnant
    cdef int previous, position, alt
    cdef double[3] upstream_diff, downstream_diff, terminal_diff
    diff(point, data[vessel, 0:3], upstream_diff)
    diff(point, data[vessel, 3:6], downstream_diff)
    diff(point, terminal, terminal_diff)
    upstream_length = norm3(upstream_diff[0], upstream_diff[1], upstream_diff[2])
    downstream_length = norm3(downstream_diff[0], downstream_diff[1], downstream_diff[2])
    terminal_length = norm3(terminal_diff[0], terminal_diff[1], terminal_diff[2])
    r_terminal = ((8.0 * nu) / M_PI) * terminal_length
    r_terminal_sister = ((8.0 * nu) / M_PI) * downstream_length + (data[vessel, 25] - ((8.0 * nu) / M_PI) * data[vessel, 20])
    f_terminal = pow(1+pow(((data[vessel, 22] * r_terminal_sister) / (terminal_flow * r_terminal)), gamma/4.0), -1.0/gamma)
    f_terminal_sister = pow(1+pow(((data[vessel, 22]*r_terminal_sister) / (terminal_flow * r_terminal)), -gamma/4.0), -1.0/gamma)
    r0 = ((8.0 * nu) / M_PI) * upstream_length + pow(((pow(f_terminal, 4.0))/r_terminal +
                                                     (pow(f_terminal_sister, 4.0))/r_terminal_sister), -1.0)
    l0 = pow(f_terminal, radius_exponent) * pow(terminal_length, length_exponent) + \
         pow(f_terminal_sister, radius_exponent) * (pow(downstream_length, length_exponent) + data[vessel, 27])
    reduced_resistances.push_back(r0)
    reduced_lengths.push_back(l0)
    flows.push_back(data[vessel, 22] + terminal_flow)
    main_idx.push_back(vessel)
    bifurcations.push_back(f_terminal_sister)
    bifurcations.push_back(f_terminal)
    if vessel == 0:
        l0 = pow(upstream_length, length_exponent) + l0
        r0 = pow((((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)), (1.0/4.0))
        alt_idx.push_back(-1)
        new_scale.push_back(1)
        alt_scale.push_back(1)
        return r0, l0, f_terminal, f_terminal_sister, 1.0, point, reduced_resistances,\
               reduced_lengths, bifurcations, flows, main_idx, alt_idx, new_scale, alt_scale,\
               r_terminal, r_terminal_sister
    previous = vessel
    vessel = idxs[vessel, 2]
    if idxs[vessel, 0] == previous:
        alt = idxs[vessel, 1]
        if alt == -1:
            alt = -2
        alt_idx.push_back(alt)
        position = 0
    else:
        alt = idxs[vessel, 0]
        if alt == -1:
            alt = -2
        alt_idx.push_back(alt)
        position = 1
    if alt == -2:
        f_changed = 1.0
        f_stagnant = 0.0
        f_terminal = f_terminal * f_changed
        f_terminal_sister = f_terminal_sister * f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
        l0 = pow(f_changed, radius_exponent) * (pow(upstream_length, length_exponent) + l0)
    else:
        f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)), (gamma/4.0))), (-1.0/gamma))
        f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)), (-gamma/4.0))), (-1.0/gamma))
        f_terminal = f_terminal*f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + pow(((pow(f_changed, 4.0)) /r0 + (pow(f_stagnant, 4.0)) /data[alt, 25]), -1.0)
        l0 = pow(f_changed, radius_exponent) *(pow(upstream_length, length_exponent) + l0) + \
             pow(f_stagnant, radius_exponent) * (pow(data[alt, 20], length_exponent) + data[alt, 27])
    reduced_resistances.push_back(r0)
    reduced_lengths.push_back(l0)
    flows.push_back(data[vessel, 22] + terminal_flow)
    main_idx.push_back(vessel)
    new_scale.push_back(f_changed)
    alt_scale.push_back(f_stagnant)
    if position == 0:
        bifurcations.push_back(f_changed)
        bifurcations.push_back(f_stagnant)
    else:
        bifurcations.push_back(f_stagnant)
        bifurcations.push_back(f_changed)
    if vessel == 0:
        l0 = pow(data[vessel, 20], length_exponent) + l0
        r0 = pow((((data[vessel, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
        return r0, l0, f_terminal,f_terminal_sister,f_parent,point,reduced_resistances,\
                reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,r_terminal,\
                r_terminal_sister
    previous = vessel
    vessel = idxs[vessel, 2]
    while vessel >= 0:
        if idxs[vessel, 0] == previous:
            alt = idxs[vessel, 1]
            if alt == -1:
                alt = -2
            alt_idx.push_back(alt)
            position = 0
        else:
            alt = idxs[vessel, 0]
            if alt == -1:
                alt = -2
            alt_idx.push_back(alt)
            position = 1
        if alt == -2:
            f_changed = 1.0
            f_stagnant = 0.0
            r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
            l0 = pow(f_changed, length_exponent) * (data[previous, 20] + l0)
            f_terminal = f_terminal * f_changed
            f_terminal_sister = f_terminal_sister * f_changed
            f_parent = f_parent * f_changed
        else:
            f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((terminal_flow+data[previous, 22])*r0)), (gamma/4.0))), (-1.0/gamma))
            f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt,25])/((terminal_flow+data[previous,22])*r0)), (-gamma/4.0))), (-1.0/gamma))
            f_terminal = f_terminal*f_changed
            f_terminal_sister = f_terminal_sister*f_changed
            f_parent = f_parent*f_changed
            r0 = ((8.0 * nu) / M_PI) * data[vessel,20] + pow(((pow(f_changed, 4.0) / r0) + (pow(f_stagnant, 4.0) / data[alt, 25])), -1.0)
            l0 = pow(f_changed, radius_exponent) * (pow(data[previous, 20], length_exponent) + l0) + \
                 pow(f_stagnant, radius_exponent) *(pow(data[alt, 20], length_exponent) + data[alt, 27])
        reduced_resistances.push_back(r0)
        reduced_lengths.push_back(l0)
        flows.push_back(data[vessel, 22] + terminal_flow)
        main_idx.push_back(vessel)
        if vessel >= 0:
            for j in range(new_scale.size()):
                new_scale[j] = new_scale[j]*f_changed
                alt_scale[j] = alt_scale[j]*f_stagnant
            new_scale.push_back(f_changed)
            alt_scale.push_back(f_stagnant)
        if position == 0:
            bifurcations.push_back(f_changed)
            bifurcations.push_back(f_stagnant)
        else:
            bifurcations.push_back(f_stagnant)
            bifurcations.push_back(f_changed)
        previous = vessel
        vessel = idxs[vessel, 2]
    l0 = pow(data[0, 20], length_exponent) + l0
    r0 = pow((((data[0, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
    return r0, l0, f_terminal, f_terminal_sister, f_parent, point, reduced_resistances, \
           reduced_lengths, bifurcations, flows, main_idx, alt_idx, new_scale, alt_scale, r_terminal, \
           r_terminal_sister

@cython.cpow(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def tree_cost(double[:] point, double[:,:] data, double[:] terminal, long[:,:] idxs, int vessel, double gamma, double nu,
              double terminal_flow, double terminal_pressure, double root_pressure, double radius_exponent,
              double length_exponent):
    cdef double upstream_length, downstream_length, terminal_length, r_terminal, r_terminal_sister, r_parent
    cdef double f_terminal, f_terminal_sister, r0, l0, f_changed, f_stagnant, f_parent, cost
    cdef int previous, position, alt
    cdef double[3] direction_upstream
    cdef double[3] direction_downstream
    cdef double[3] direction_terminal
    direction_upstream[0] = point[0] - data[vessel, 0]
    direction_upstream[1] = point[1] - data[vessel, 1]
    direction_upstream[2] = point[2] - data[vessel, 2]
    direction_downstream[0] = point[0] - data[vessel, 3]
    direction_downstream[1] = point[1] - data[vessel, 4]
    direction_downstream[2] = point[2] - data[vessel, 5]
    direction_terminal[0] = point[0] - terminal[0]
    direction_terminal[1] = point[1] - terminal[1]
    direction_terminal[2] = point[2] - terminal[2]
    upstream_length = sqrt(direction_upstream[0] ** 2.0 + direction_upstream[1] ** 2.0 + direction_upstream[2] ** 2.0)
    downstream_length = sqrt(direction_downstream[0] ** 2.0 + direction_downstream[1] ** 2.0 + direction_downstream[2] ** 2.0)
    terminal_length = sqrt(direction_terminal[0] ** 2.0 + direction_terminal[1] ** 2.0 + direction_terminal[2] ** 2.0)
    #upstream_length = norm(diff(point, data[vessel, 0:3]))
    #downstream_length = norm(diff(point, data[vessel, 3:6]))
    #terminal_length = norm(diff(point, terminal))
    #upstream_length = 0.0
    #downstream_length = 0.0
    #terminal_length = 0.0
    #length(point, data[vessel, 0:3], upstream_length)
    #length(point, data[vessel, 3:6], downstream_length)
    #length(point, terminal, terminal_length)
    if upstream_length == 0.0:
        return HUGE_VAL
    elif downstream_length == 0.0:
        return HUGE_VAL
    elif terminal_length == 0.0:
        return HUGE_VAL
    r_terminal = ((8.0 * nu) / M_PI) * terminal_length
    r_terminal_sister = ((8.0 * nu) / M_PI) * downstream_length + (data[vessel, 25] -
                                                                   ((8.0 * nu) / M_PI) * data[vessel, 20])
    f_terminal = pow(1+pow(((data[vessel, 22] * r_terminal_sister) / (terminal_flow * r_terminal)), gamma/4.0),
                     -1.0/gamma)
    f_terminal_sister = pow(1+pow(((data[vessel, 22]*r_terminal_sister) / (terminal_flow * r_terminal)), -gamma/4.0),
                            -1.0/gamma)
    f_parent = 1.0
    r0 = ((8.0 * nu) / M_PI) * upstream_length + pow(((pow(f_terminal, 4.0))/r_terminal +
                                                     (pow(f_terminal_sister, 4.0))/r_terminal_sister), -1.0)
    r_parent = r0
    l0 = pow(f_terminal, radius_exponent) * pow(terminal_length, length_exponent) + \
         pow(f_terminal_sister, radius_exponent) * (pow(downstream_length, length_exponent) + data[vessel, 27])
    if vessel == 0:
        l0 = pow(upstream_length, length_exponent) + l0
        r0 = pow((((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)), (1.0/4.0))
        cost = M_PI * pow(r0, radius_exponent) * l0
        return cost
    previous = vessel
    vessel = idxs[vessel, 2]
    if idxs[vessel, 0] == previous:
        alt = idxs[vessel, 1]
        if alt == -1:
            alt = -2
        position = 0
    else:
        alt = idxs[vessel, 0]
        if alt == -1:
            alt = -2
        position = 1
    if alt == -2:
        f_changed = 1.0
        f_stagnant = 0.0
        f_terminal = f_terminal * f_changed
        f_terminal_sister = f_terminal_sister * f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
        l0 = pow(f_changed, radius_exponent) * (pow(upstream_length, length_exponent) + l0)
    else:
        f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)),
                                 (gamma/4.0))), (-1.0/gamma))
        f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)),
                                  (-gamma/4.0))), (-1.0/gamma))
        f_terminal = f_terminal * f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + pow(((pow(f_changed, 4.0)) /r0 +
                                                           (pow(f_stagnant, 4.0)) /data[alt, 25]), -1.0)
        l0 = pow(f_changed, radius_exponent) *(pow(upstream_length, length_exponent) + l0) + \
             pow(f_stagnant, radius_exponent) * (pow(data[alt, 20], length_exponent) + data[alt, 27])
    if vessel == 0:
        l0 = pow(data[vessel, 20], length_exponent) + l0
        r0 = pow((((data[vessel, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
        cost = M_PI * pow(r0, radius_exponent) * l0
        return cost
    previous = vessel
    vessel = idxs[vessel, 2]
    while vessel >= 0:
        if idxs[vessel, 0] == previous:
            alt = idxs[vessel, 1]
            if alt == -1:
                alt = -2
            position = 0
        else:
            alt = idxs[vessel, 0]
            if alt == -1:
                alt = -2
            position = 1
        if alt == -2:
            f_changed = 1.0
            f_stagnant = 0.0
            r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
            l0 = pow(f_changed, length_exponent) * (data[previous, 20] + l0)
            f_terminal = f_terminal * f_changed
            f_terminal_sister = f_terminal_sister * f_changed
            f_parent = f_parent * f_changed
        else:
            f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((terminal_flow+data[previous, 22])*r0)),
                                     (gamma/4.0))), (-1.0/gamma))
            f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt,25])/((terminal_flow+data[previous,22])*r0)),
                                      (-gamma/4.0))), (-1.0/gamma))
            f_terminal = f_terminal*f_changed
            f_terminal_sister = f_terminal_sister*f_changed
            f_parent = f_parent*f_changed
            r0 = ((8.0 * nu) / M_PI) * data[vessel,20] + pow(((pow(f_changed, 4.0) / r0) +
                                                              (pow(f_stagnant, 4.0) / data[alt, 25])), -1.0)
            l0 = pow(f_changed, radius_exponent) * (pow(data[previous, 20], length_exponent) + l0) + \
                 pow(f_stagnant, radius_exponent) *(pow(data[alt, 20], length_exponent) + data[alt, 27])
        previous = vessel
        vessel = idxs[vessel, 2]
    l0 = pow(data[0, 20], length_exponent) + l0
    r0 = pow((((data[0, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
    cost = M_PI * pow(r0, radius_exponent) * l0
    return cost

@cython.cpow(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def create_new_vessels(double[:] point, double[:,:] data, double[:] terminal, double[:,:] terminal_vessel,
                       double[:,:] terminal_sister_vessel, double[:,:] terminal_parent_vessel, double max_node,
                       double max_vessel, long[:,:] idxs, int vessel, double gamma, double nu, double terminal_flow,
                       double terminal_pressure, double root_pressure, double radius_exponent, double length_exponent):
    cdef double upstream_length, downstream_length, terminal_length, r_terminal, r_terminal_sister, r_parent
    cdef double f_terminal, f_terminal_sister, r0, l0, f_changed, f_stagnant, f_parent, cost, rl_parent
    cdef double terminal_bifurcation_ratio, terminal_sister_bifurcation_ratio
    cdef int previous, position, alt, original_vessel
    original_vessel = vessel
    cdef double[3] upstream_diff, downstream_diff, terminal_diff
    diff(point, data[vessel, 0:3], upstream_diff)
    diff(point, data[vessel, 3:6], downstream_diff)
    diff(point, terminal, terminal_diff)
    upstream_length = norm3(upstream_diff[0], upstream_diff[1], upstream_diff[2])
    downstream_length = norm3(downstream_diff[0], downstream_diff[1], downstream_diff[2])
    terminal_length = norm3(terminal_diff[0], terminal_diff[1], terminal_diff[2])
    r_terminal = ((8.0 * nu) / M_PI) * terminal_length
    r_terminal_sister = ((8.0 * nu) / M_PI) * downstream_length + (data[vessel, 25] -
                                                                   ((8.0 * nu) / M_PI) * data[vessel, 20])
    f_terminal = pow(1+pow(((data[vessel, 22] * r_terminal_sister) / (terminal_flow * r_terminal)), gamma/4.0),
                     -1.0/gamma)
    terminal_bifurcation_ratio = f_terminal
    f_terminal_sister = pow(1+pow(((data[vessel, 22]*r_terminal_sister) / (terminal_flow * r_terminal)), -gamma/4.0),
                            -1.0/gamma)
    terminal_sister_bifurcation_ratio = f_terminal_sister
    f_parent = 1.0
    r0 = ((8.0 * nu) / M_PI) * upstream_length + pow(((pow(f_terminal, 4.0))/r_terminal +
                                                     (pow(f_terminal_sister, 4.0))/r_terminal_sister), -1.0)
    r_parent = r0
    l0 = pow(f_terminal, radius_exponent) * pow(terminal_length, length_exponent) + \
         pow(f_terminal_sister, radius_exponent) * (pow(downstream_length, length_exponent) + data[vessel, 27])
    rl_parent = l0
    if vessel == 0:
        l0 = pow(upstream_length, length_exponent) + l0
        r0 = pow((((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)), (1.0/4.0))
        terminal_vessel[0, 0:3] = point[:]
        terminal_vessel[0, 3:6] = terminal[:]
        u, v, w = basis3(terminal_vessel[0, 0:3], terminal_vessel[0, 3:6])
        terminal_vessel[0, 6] = u[0]
        terminal_vessel[0, 7] = u[1]
        terminal_vessel[0, 8] = u[2]
        terminal_vessel[0, 9] = v[0]
        terminal_vessel[0, 10] = v[1]
        terminal_vessel[0, 11] = v[2]
        terminal_vessel[0, 12] = w[0]
        terminal_vessel[0, 13] = w[1]
        terminal_vessel[0, 14] = w[2]
        terminal_vessel[0, 17] = original_vessel * 1.0
        terminal_vessel[0, 18] = max_node + 1.0
        terminal_vessel[0, 19] = max_node + 2.0
        terminal_vessel[0, 20] = terminal_length
        terminal_vessel[0, 21] = r0 * f_terminal
        terminal_vessel[0, 22] = terminal_flow
        terminal_vessel[0, 25] = r_terminal
        terminal_vessel[0, 26] = data[original_vessel, 26] + 1.0
        terminal_vessel[0, 27] = 0.0
        terminal_vessel[0, 28] = f_terminal
        terminal_sister_vessel[0, 0:3] = point[:]
        terminal_sister_vessel[0, 3:6] = data[original_vessel, 3:6]
        u, v, w = basis3(terminal_sister_vessel[0, 0:3], terminal_sister_vessel[0, 3:6])
        terminal_sister_vessel[0, 6] = u[0]
        terminal_sister_vessel[0, 7] = u[1]
        terminal_sister_vessel[0, 8] = u[2]
        terminal_sister_vessel[0, 9] = v[0]
        terminal_sister_vessel[0, 10] = v[1]
        terminal_sister_vessel[0, 11] = v[2]
        terminal_sister_vessel[0, 12] = w[0]
        terminal_sister_vessel[0, 13] = w[1]
        terminal_sister_vessel[0, 14] = w[2]
        terminal_sister_vessel[0, 15] = data[original_vessel, 15]
        terminal_sister_vessel[0, 16] = data[original_vessel, 16]
        terminal_sister_vessel[0, 17] = original_vessel * 1.0
        terminal_sister_vessel[0, 18] = max_node + 1.0
        terminal_sister_vessel[0, 19] = data[original_vessel, 19]
        terminal_sister_vessel[0, 20] = downstream_length
        terminal_sister_vessel[0, 21] = r0 * f_terminal_sister
        terminal_sister_vessel[0, 22] = data[original_vessel, 22]
        terminal_sister_vessel[0, 23] = data[original_vessel, 23]
        terminal_sister_vessel[0, 24] = data[original_vessel, 24]
        terminal_sister_vessel[0, 25] = r_terminal_sister
        terminal_sister_vessel[0, 26] = data[original_vessel, 26] + 1.0
        terminal_sister_vessel[0, 27] = data[original_vessel, 27]
        terminal_sister_vessel[0, 28] = f_terminal_sister
        terminal_parent_vessel[0, 0:3] = data[original_vessel, 0:3]
        terminal_parent_vessel[0, 3:6] = point[:]
        u, v, w = basis3(terminal_parent_vessel[0, 0:3], terminal_parent_vessel[0, 3:6])
        terminal_parent_vessel[0, 6] = u[0]
        terminal_parent_vessel[0, 7] = u[1]
        terminal_parent_vessel[0, 8] = u[2]
        terminal_parent_vessel[0, 9] = v[0]
        terminal_parent_vessel[0, 10] = v[1]
        terminal_parent_vessel[0, 11] = v[2]
        terminal_parent_vessel[0, 12] = w[0]
        terminal_parent_vessel[0, 13] = w[1]
        terminal_parent_vessel[0, 14] = w[2]
        terminal_parent_vessel[0, 15] = max_vessel
        terminal_parent_vessel[0, 16] = max_vessel + 1.0
        terminal_parent_vessel[0, 17] = data[original_vessel, 17]
        terminal_parent_vessel[0, 18] = data[original_vessel, 18]
        terminal_parent_vessel[0, 19] = max_node + 1.0
        terminal_parent_vessel[0, 20] = upstream_length
        terminal_parent_vessel[0, 21] = r0 * f_parent
        terminal_parent_vessel[0, 22] = data[original_vessel, 22] + terminal_flow
        terminal_parent_vessel[0, 23] = terminal_bifurcation_ratio
        terminal_parent_vessel[0, 24] = terminal_sister_bifurcation_ratio
        terminal_parent_vessel[0, 25] = r_parent
        terminal_parent_vessel[0, 26] = data[original_vessel, 26]
        terminal_parent_vessel[0, 27] = rl_parent
        terminal_parent_vessel[0, 28] = f_parent
        return
    previous = vessel
    vessel = idxs[vessel, 2]
    if idxs[vessel, 0] == previous:
        alt = idxs[vessel, 1]
        if alt == -1:
            alt = -2
        position = 0
    else:
        alt = idxs[vessel, 0]
        if alt == -1:
            alt = -2
        position = 1
    if alt == -2:
        f_changed = 1.0
        f_stagnant = 0.0
        f_terminal = f_terminal * f_changed
        f_terminal_sister = f_terminal_sister * f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
        l0 = pow(f_changed, radius_exponent) * (pow(upstream_length, length_exponent) + l0)
    else:
        f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)),
                                 (gamma/4.0))), (-1.0/gamma))
        f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)),
                                  (-gamma/4.0))), (-1.0/gamma))
        f_terminal = f_terminal * f_changed
        f_terminal_sister = f_terminal_sister * f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + pow(((pow(f_changed, 4.0)) /r0 +
                                                           (pow(f_stagnant, 4.0)) /data[alt, 25]), -1.0)
        l0 = pow(f_changed, radius_exponent) *(pow(upstream_length, length_exponent) + l0) + \
             pow(f_stagnant, radius_exponent) * (pow(data[alt, 20], length_exponent) + data[alt, 27])
    if vessel == 0:
        l0 = pow(data[vessel, 20], length_exponent) + l0
        r0 = pow((((data[vessel, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
        terminal_vessel[0, 0:3] = point[:]
        terminal_vessel[0, 3:6] = terminal[:]
        u, v, w = basis3(terminal_vessel[0, 0:3], terminal_vessel[0, 3:6])
        terminal_vessel[0, 6] = u[0]
        terminal_vessel[0, 7] = u[1]
        terminal_vessel[0, 8] = u[2]
        terminal_vessel[0, 9] = v[0]
        terminal_vessel[0, 10] = v[1]
        terminal_vessel[0, 11] = v[2]
        terminal_vessel[0, 12] = w[0]
        terminal_vessel[0, 13] = w[1]
        terminal_vessel[0, 14] = w[2]
        terminal_vessel[0, 17] = original_vessel * 1.0
        terminal_vessel[0, 18] = max_node + 1.0
        terminal_vessel[0, 19] = max_node + 2.0
        terminal_vessel[0, 20] = terminal_length
        terminal_vessel[0, 21] = r0 * f_terminal
        terminal_vessel[0, 22] = terminal_flow
        terminal_vessel[0, 25] = r_terminal
        terminal_vessel[0, 26] = data[original_vessel, 26] + 1.0
        terminal_vessel[0, 27] = 0.0
        terminal_vessel[0, 28] = f_terminal
        terminal_sister_vessel[0, 0:3] = point[:]
        terminal_sister_vessel[0, 3:6] = data[original_vessel, 3:6]
        u, v, w = basis3(terminal_sister_vessel[0, 0:3], terminal_sister_vessel[0, 3:6])
        terminal_sister_vessel[0, 6] = u[0]
        terminal_sister_vessel[0, 7] = u[1]
        terminal_sister_vessel[0, 8] = u[2]
        terminal_sister_vessel[0, 9] = v[0]
        terminal_sister_vessel[0, 10] = v[1]
        terminal_sister_vessel[0, 11] = v[2]
        terminal_sister_vessel[0, 12] = w[0]
        terminal_sister_vessel[0, 13] = w[1]
        terminal_sister_vessel[0, 14] = w[2]
        terminal_sister_vessel[0, 15] = data[original_vessel, 15]
        terminal_sister_vessel[0, 16] = data[original_vessel, 16]
        terminal_sister_vessel[0, 17] = original_vessel * 1.0
        terminal_sister_vessel[0, 18] = max_node + 1.0
        terminal_sister_vessel[0, 19] = data[original_vessel, 19]
        terminal_sister_vessel[0, 20] = downstream_length
        terminal_sister_vessel[0, 21] = r0 * f_terminal_sister
        terminal_sister_vessel[0, 22] = data[original_vessel, 22]
        terminal_sister_vessel[0, 23] = data[original_vessel, 23]
        terminal_sister_vessel[0, 24] = data[original_vessel, 24]
        terminal_sister_vessel[0, 25] = r_terminal_sister
        terminal_sister_vessel[0, 26] = data[original_vessel, 26] + 1.0
        terminal_sister_vessel[0, 27] = data[original_vessel, 27]
        terminal_sister_vessel[0, 28] = f_terminal_sister
        terminal_parent_vessel[0, 0:3] = data[original_vessel, 0:3]
        terminal_parent_vessel[0, 3:6] = point[:]
        u, v, w = basis3(terminal_parent_vessel[0, 0:3], terminal_parent_vessel[0, 3:6])
        terminal_parent_vessel[0, 6] = u[0]
        terminal_parent_vessel[0, 7] = u[1]
        terminal_parent_vessel[0, 8] = u[2]
        terminal_parent_vessel[0, 9] = v[0]
        terminal_parent_vessel[0, 10] = v[1]
        terminal_parent_vessel[0, 11] = v[2]
        terminal_parent_vessel[0, 12] = w[0]
        terminal_parent_vessel[0, 13] = w[1]
        terminal_parent_vessel[0, 14] = w[2]
        terminal_parent_vessel[0, 15] = max_vessel
        terminal_parent_vessel[0, 16] = max_vessel + 1.0
        terminal_parent_vessel[0, 17] = data[original_vessel, 17]
        terminal_parent_vessel[0, 18] = data[original_vessel, 18]
        terminal_parent_vessel[0, 19] = max_node + 1.0
        terminal_parent_vessel[0, 20] = upstream_length
        terminal_parent_vessel[0, 21] = r0 * f_parent
        terminal_parent_vessel[0, 22] = data[original_vessel, 22] + terminal_flow
        terminal_parent_vessel[0, 23] = terminal_bifurcation_ratio
        terminal_parent_vessel[0, 24] = terminal_sister_bifurcation_ratio
        terminal_parent_vessel[0, 25] = r_parent
        terminal_parent_vessel[0, 26] = data[original_vessel, 26]
        terminal_parent_vessel[0, 27] = rl_parent
        terminal_parent_vessel[0, 28] = f_parent
        return
    previous = vessel
    vessel = idxs[vessel, 2]
    while vessel >= 0:
        if idxs[vessel, 0] == previous:
            alt = idxs[vessel, 1]
            if alt == -1:
                alt = -2
            position = 0
        else:
            alt = idxs[vessel, 0]
            if alt == -1:
                alt = -2
            position = 1
        if alt == -2:
            f_changed = 1.0
            f_stagnant = 0.0
            r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
            l0 = pow(f_changed, length_exponent) * (data[previous, 20] + l0)
            f_terminal = f_terminal * f_changed
            f_terminal_sister = f_terminal_sister * f_changed
            f_parent = f_parent * f_changed
        else:
            f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((terminal_flow+data[previous, 22])*r0)),
                                     (gamma/4.0))), (-1.0/gamma))
            f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt,25])/((terminal_flow+data[previous,22])*r0)),
                                      (-gamma/4.0))), (-1.0/gamma))
            f_terminal = f_terminal * f_changed
            f_terminal_sister = f_terminal_sister * f_changed
            f_parent = f_parent * f_changed
            r0 = ((8.0 * nu) / M_PI) * data[vessel,20] + pow(((pow(f_changed, 4.0) / r0) +
                                                              (pow(f_stagnant, 4.0) / data[alt, 25])), -1.0)
            l0 = pow(f_changed, radius_exponent) * (pow(data[previous, 20], length_exponent) + l0) + \
                 pow(f_stagnant, radius_exponent) *(pow(data[alt, 20], length_exponent) + data[alt, 27])
        previous = vessel
        vessel = idxs[vessel, 2]
    l0 = pow(data[0, 20], length_exponent) + l0
    r0 = pow((((data[0, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
    terminal_vessel[0, 0:3] = point[:]
    terminal_vessel[0, 3:6] = terminal[:]
    u, v, w = basis3(terminal_vessel[0, 0:3], terminal_vessel[0, 3:6])
    terminal_vessel[0, 6] = u[0]
    terminal_vessel[0, 7] = u[1]
    terminal_vessel[0, 8] = u[2]
    terminal_vessel[0, 9] = v[0]
    terminal_vessel[0, 10] = v[1]
    terminal_vessel[0, 11] = v[2]
    terminal_vessel[0, 12] = w[0]
    terminal_vessel[0, 13] = w[1]
    terminal_vessel[0, 14] = w[2]
    terminal_vessel[0, 17] = original_vessel * 1.0
    terminal_vessel[0, 18] = max_node + 1.0
    terminal_vessel[0, 19] = max_node + 2.0
    terminal_vessel[0, 20] = terminal_length
    terminal_vessel[0, 21] = r0 * f_terminal
    terminal_vessel[0, 22] = terminal_flow
    terminal_vessel[0, 25] = r_terminal
    terminal_vessel[0, 26] = data[original_vessel, 26] + 1.0
    terminal_vessel[0, 27] = 0.0
    terminal_vessel[0, 28] = f_terminal
    terminal_sister_vessel[0, 0:3] = point[:]
    terminal_sister_vessel[0, 3:6] = data[original_vessel, 3:6]
    u, v, w = basis3(terminal_sister_vessel[0, 0:3], terminal_sister_vessel[0, 3:6])
    terminal_sister_vessel[0, 6] = u[0]
    terminal_sister_vessel[0, 7] = u[1]
    terminal_sister_vessel[0, 8] = u[2]
    terminal_sister_vessel[0, 9] = v[0]
    terminal_sister_vessel[0, 10] = v[1]
    terminal_sister_vessel[0, 11] = v[2]
    terminal_sister_vessel[0, 12] = w[0]
    terminal_sister_vessel[0, 13] = w[1]
    terminal_sister_vessel[0, 14] = w[2]
    terminal_sister_vessel[0, 15] = data[original_vessel, 15]
    terminal_sister_vessel[0, 16] = data[original_vessel, 16]
    terminal_sister_vessel[0, 17] = original_vessel * 1.0
    terminal_sister_vessel[0, 18] = max_node + 1.0
    terminal_sister_vessel[0, 19] = data[original_vessel, 19]
    terminal_sister_vessel[0, 20] = downstream_length
    terminal_sister_vessel[0, 21] = r0 * f_terminal_sister
    terminal_sister_vessel[0, 22] = data[original_vessel, 22]
    terminal_sister_vessel[0, 23] = data[original_vessel, 23]
    terminal_sister_vessel[0, 24] = data[original_vessel, 24]
    terminal_sister_vessel[0, 25] = r_terminal_sister
    terminal_sister_vessel[0, 26] = data[original_vessel, 26] + 1.0
    terminal_sister_vessel[0, 27] = data[original_vessel, 27]
    terminal_sister_vessel[0, 28] = f_terminal_sister
    terminal_parent_vessel[0, 0:3] = data[original_vessel, 0:3]
    terminal_parent_vessel[0, 3:6] = point[:]
    u, v, w = basis3(terminal_parent_vessel[0, 0:3], terminal_parent_vessel[0, 3:6])
    terminal_parent_vessel[0, 6] = u[0]
    terminal_parent_vessel[0, 7] = u[1]
    terminal_parent_vessel[0, 8] = u[2]
    terminal_parent_vessel[0, 9] = v[0]
    terminal_parent_vessel[0, 10] = v[1]
    terminal_parent_vessel[0, 11] = v[2]
    terminal_parent_vessel[0, 12] = w[0]
    terminal_parent_vessel[0, 13] = w[1]
    terminal_parent_vessel[0, 14] = w[2]
    terminal_parent_vessel[0, 15] = max_vessel
    terminal_parent_vessel[0, 16] = max_vessel + 1.0
    terminal_parent_vessel[0, 17] = data[original_vessel, 17]
    terminal_parent_vessel[0, 18] = data[original_vessel, 18]
    terminal_parent_vessel[0, 19] = max_node + 1.0
    terminal_parent_vessel[0, 20] = upstream_length
    terminal_parent_vessel[0, 21] = r0 * f_parent
    terminal_parent_vessel[0, 22] = data[original_vessel, 22] + terminal_flow
    terminal_parent_vessel[0, 23] = terminal_bifurcation_ratio
    terminal_parent_vessel[0, 24] = terminal_sister_bifurcation_ratio
    terminal_parent_vessel[0, 25] = r_parent
    terminal_parent_vessel[0, 26] = data[original_vessel, 26]
    terminal_parent_vessel[0, 27] = rl_parent
    terminal_parent_vessel[0, 28] = f_parent
    return

@cython.cpow(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def update_vessels(double[:] point, double[:,:] data, double[:] terminal, long[:,:] idxs,
                   int vessel, double gamma, double nu, double terminal_flow, double terminal_pressure,
                   double root_pressure, double radius_exponent, double length_exponent):
    cdef double upstream_length, downstream_length, terminal_length, r_terminal, r_terminal_sister, r_parent
    cdef double f_terminal, f_terminal_sister, r0, l0, f_changed, f_stagnant, f_parent, cost
    cdef vector[vector[double]] bifurcation_ratios
    cdef vector[double] tmp_bifurcation_ratios = [0.0, 0.0]
    cdef vector[int] main_idx
    cdef vector[int] alt_idx
    cdef vector[double] main_scale
    cdef vector[double] alt_scale
    cdef vector[double] reduced_resistance
    cdef vector[double] reduced_length
    cdef vector[double] flows
    cdef int previous, position, alt, alt_position
    cdef double[3] upstream_diff, downstream_diff, terminal_diff
    diff(point, data[vessel, 0:3], upstream_diff)
    diff(point, data[vessel, 3:6], downstream_diff)
    diff(point, terminal, terminal_diff)
    upstream_length = norm3(upstream_diff[0], upstream_diff[1], upstream_diff[2])
    downstream_length = norm3(downstream_diff[0], downstream_diff[1], downstream_diff[2])
    terminal_length = norm3(terminal_diff[0], terminal_diff[1], terminal_diff[2])
    r_terminal = ((8.0 * nu) / M_PI) * terminal_length
    r_terminal_sister = ((8.0 * nu) / M_PI) * downstream_length + (data[vessel, 25] -
                                                                   ((8.0 * nu) / M_PI) * data[vessel, 20])
    f_terminal = pow(1+pow(((data[vessel, 22] * r_terminal_sister) / (terminal_flow * r_terminal)), gamma/4.0),
                     -1.0/gamma)
    f_terminal_sister = pow(1+pow(((data[vessel, 22]*r_terminal_sister) / (terminal_flow * r_terminal)), -gamma/4.0),
                            -1.0/gamma)
    f_parent = 1.0
    r0 = ((8.0 * nu) / M_PI) * upstream_length + pow(((pow(f_terminal, 4.0))/r_terminal +
                                                     (pow(f_terminal_sister, 4.0))/r_terminal_sister), -1.0)
    r_parent = r0
    l0 = pow(f_terminal, radius_exponent) * pow(terminal_length, length_exponent) + \
         pow(f_terminal_sister, radius_exponent) * (pow(downstream_length, length_exponent) + data[vessel, 27])
    if vessel == 0:
        l0 = pow(upstream_length, length_exponent) + l0
        r0 = pow((((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)), (1.0/4.0))
        return reduced_resistance, reduced_length, main_idx, main_scale, alt_idx, alt_scale, bifurcation_ratios, flows, r0, l0
    previous = vessel
    vessel = idxs[vessel, 2]
    if idxs[vessel, 0] == previous:
        alt = idxs[vessel, 1]
        if alt == -1:
            alt = -2
        position = 0
        alt_position = 1
    else:
        alt = idxs[vessel, 0]
        if alt == -1:
            alt = -2
        position = 1
        alt_position = 0
    if alt == -2:
        f_changed = 1.0
        f_stagnant = 0.0
        f_terminal = f_terminal * f_changed
        f_terminal_sister = f_terminal_sister * f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
        l0 = pow(f_changed, radius_exponent) * (pow(upstream_length, length_exponent) + l0)
    else:
        f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)),
                                 (gamma/4.0))), (-1.0/gamma))
        f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)),
                                  (-gamma/4.0))), (-1.0/gamma))
        f_terminal = f_terminal * f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + pow(((pow(f_changed, 4.0)) /r0 +
                                                           (pow(f_stagnant, 4.0)) /data[alt, 25]), -1.0)
        l0 = pow(f_changed, radius_exponent) *(pow(upstream_length, length_exponent) + l0) + \
             pow(f_stagnant, radius_exponent) * (pow(data[alt, 20], length_exponent) + data[alt, 27])
    tmp_bifurcation_ratios[position] = f_changed
    tmp_bifurcation_ratios[alt_position] = f_stagnant
    if vessel == 0:
        main_idx.push_back(vessel)
        main_scale.push_back(1.0)
        alt_idx.push_back(alt)
        alt_scale.push_back(f_stagnant)
        reduced_resistance.push_back(r0)
        reduced_length.push_back(l0)
        tmp_bifurcation_ratios[position] = f_changed
        tmp_bifurcation_ratios[alt_position] = f_stagnant
        bifurcation_ratios.push_back(tmp_bifurcation_ratios)
        flows.push_back(data[vessel, 22] + terminal_flow)
        l0 = pow(data[vessel, 20], length_exponent) + l0
        r0 = pow((((data[vessel, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
        return reduced_resistance, reduced_length, main_idx, main_scale, alt_idx, alt_scale, bifurcation_ratios, flows, r0, l0
    else:
        main_idx.push_back(vessel)
        alt_idx.push_back(alt)
        main_scale.push_back(1.0)
        alt_scale.push_back(f_stagnant)
        reduced_resistance.push_back(r0)
        reduced_length.push_back(l0)
        tmp_bifurcation_ratios[position] = f_changed
        tmp_bifurcation_ratios[alt_position] = f_stagnant
        bifurcation_ratios.push_back(tmp_bifurcation_ratios)
        flows.push_back(data[vessel, 22] + terminal_flow)
    # All vessels past this point are above the triad
    previous = vessel
    vessel = idxs[vessel, 2]
    while vessel >= 0:
        if idxs[vessel, 0] == previous:
            alt = idxs[vessel, 1]
            if alt == -1:
                alt = -2
            position = 0
            alt_position = 1
        else:
            alt = idxs[vessel, 0]
            if alt == -1:
                alt = -2
            position = 1
            alt_position = 0
        if alt == -2:
            f_changed = 1.0
            f_stagnant = 0.0
            r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
            l0 = pow(f_changed, length_exponent) * (data[previous, 20] + l0)
            f_terminal = f_terminal * f_changed
            f_terminal_sister = f_terminal_sister * f_changed
            f_parent = f_parent * f_changed
        else:
            f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((terminal_flow+data[previous, 22])*r0)),
                                     (gamma/4.0))), (-1.0/gamma))
            f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt,25])/((terminal_flow+data[previous,22])*r0)),
                                      (-gamma/4.0))), (-1.0/gamma))
            f_terminal = f_terminal * f_changed
            f_terminal_sister = f_terminal_sister * f_changed
            f_parent = f_parent * f_changed
            r0 = ((8.0 * nu) / M_PI) * data[vessel,20] + pow(((pow(f_changed, 4.0) / r0) +
                                                              (pow(f_stagnant, 4.0) / data[alt, 25])), -1.0)
            l0 = pow(f_changed, radius_exponent) * (pow(data[previous, 20], length_exponent) + l0) + \
                 pow(f_stagnant, radius_exponent) *(pow(data[alt, 20], length_exponent) + data[alt, 27])
        for j in range(main_scale.size()):
            main_scale[j] = main_scale[j] * f_changed
            alt_scale[j] = alt_scale[j] * f_changed
        main_idx.push_back(vessel)
        alt_idx.push_back(alt)
        main_scale.push_back(1.0)
        alt_scale.push_back(f_stagnant)
        reduced_resistance.push_back(r0)
        reduced_length.push_back(l0)
        tmp_bifurcation_ratios[position] = f_changed
        tmp_bifurcation_ratios[alt_position] = f_stagnant
        bifurcation_ratios.push_back(tmp_bifurcation_ratios)
        flows.push_back(data[vessel, 22] + terminal_flow)
        previous = vessel
        vessel = idxs[vessel, 2]
    l0 = pow(data[0, 20], length_exponent) + l0
    r0 = pow((((data[0, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
    return reduced_resistance, reduced_length, main_idx, main_scale, alt_idx, alt_scale, bifurcation_ratios, flows, r0, l0

@cython.cpow(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def tree_cost_2(double[:] point, double[:,:] data, double[:] terminal, long[:,:] idxs, int vessel, double gamma, double nu,
              double terminal_flow, double terminal_pressure, double root_pressure, double radius_exponent,
              double length_exponent):
    cdef double upstream_length, downstream_length, terminal_length, r_terminal, r_terminal_sister, r_parent
    cdef double f_terminal, f_terminal_sister, r0, l0, f_changed, f_stagnant, f_parent, cost
    cdef int previous, position, alt
    cdef double[3] direction_upstream
    cdef double[3] direction_downstream
    cdef double[3] direction_terminal
    direction_upstream[0] = point[0] - data[vessel, 0]
    direction_upstream[1] = point[1] - data[vessel, 1]
    direction_upstream[2] = point[2] - data[vessel, 2]
    direction_downstream[0] = point[0] - data[vessel, 3]
    direction_downstream[1] = point[1] - data[vessel, 4]
    direction_downstream[2] = point[2] - data[vessel, 5]
    direction_terminal[0] = point[0] - terminal[0]
    direction_terminal[1] = point[1] - terminal[1]
    direction_terminal[2] = point[2] - terminal[2]
    upstream_length = sqrt(direction_upstream[0] ** 2.0 + direction_upstream[1] ** 2.0 + direction_upstream[2] ** 2.0)
    downstream_length = sqrt(direction_downstream[0] ** 2.0 + direction_downstream[1] ** 2.0 + direction_downstream[2] ** 2.0)
    terminal_length = sqrt(direction_terminal[0] ** 2.0 + direction_terminal[1] ** 2.0 + direction_terminal[2] ** 2.0)
    if upstream_length == 0.0:
        return HUGE_VAL #3.0*(M_PI*pow(data[0,21],2.0)*(data[0,20] + data[0,27]))
    elif downstream_length == 0.0:
        return HUGE_VAL #3.0*(M_PI*pow(data[0,21],2.0)*(data[0,20] + data[0,27]))
    elif terminal_length == 0.0:
        return HUGE_VAL #3.0*(M_PI*pow(data[0,21],2.0)*(data[0,20] + data[0,27]))
    r_terminal = ((8.0 * nu) / M_PI) * terminal_length
    r_terminal_sister = ((8.0 * nu) / M_PI) * downstream_length + (data[vessel, 25] - ((8.0 * nu) / M_PI) * data[vessel, 20])
    f_terminal = pow(1+pow(((data[vessel, 22] * r_terminal_sister) / (terminal_flow * r_terminal)), gamma/4.0),-1.0/gamma)
    f_terminal_sister = pow(1+pow(((data[vessel, 22]*r_terminal_sister) / (terminal_flow * r_terminal)), -gamma/4.0),-1.0/gamma)
    f_parent = 1.0
    r0 = ((8.0 * nu) / M_PI) * upstream_length + pow(((pow(f_terminal, 4.0))/r_terminal + (pow(f_terminal_sister, 4.0))/r_terminal_sister), -1.0)
    r_parent = r0
    l0 = pow(f_terminal, radius_exponent) * pow(terminal_length, length_exponent) + pow(f_terminal_sister, radius_exponent) * (pow(downstream_length, length_exponent) + data[vessel, 27])
    if vessel == 0:
        l0 = pow(upstream_length, length_exponent) + l0
        r0 = pow((((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)), (1.0/4.0))
        cost = M_PI * pow(r0, radius_exponent) * l0
        return cost
    previous = vessel
    vessel = idxs[vessel, 2]
    if idxs[vessel, 0] == previous:
        alt = idxs[vessel, 1]
        if alt == -1:
            alt = -2
        position = 0
    else:
        alt = idxs[vessel, 0]
        if alt == -1:
            alt = -2
        position = 1
    if alt == -2:
        f_changed = 1.0
        f_stagnant = 0.0
        f_terminal = f_terminal * f_changed
        f_terminal_sister = f_terminal_sister * f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
        l0 = pow(f_changed, radius_exponent) * (pow(upstream_length, length_exponent) + l0)
    else:
        f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)),(gamma/4.0))), (-1.0/gamma))
        f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)),(-gamma/4.0))), (-1.0/gamma))
        f_terminal = f_terminal * f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + pow(((pow(f_changed, 4.0)) /r0 + (pow(f_stagnant, 4.0)) /data[alt, 25]), -1.0)
        l0 = pow(f_changed, radius_exponent) *(pow(upstream_length, length_exponent) + l0) + pow(f_stagnant, radius_exponent) * (pow(data[alt, 20], length_exponent) + data[alt, 27])
    if vessel == 0:
        l0 = pow(data[vessel, 20], length_exponent) + l0
        r0 = pow((((data[vessel, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
        cost = M_PI * pow(r0, radius_exponent) * l0
        return cost
    previous = vessel
    vessel = idxs[vessel, 2]
    while vessel >= 0:
        if idxs[vessel, 0] == previous:
            alt = idxs[vessel, 1]
            if alt == -1:
                alt = -2
            position = 0
        else:
            alt = idxs[vessel, 0]
            if alt == -1:
                alt = -2
            position = 1
        if alt == -2:
            f_changed = 1.0
            f_stagnant = 0.0
            r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
            l0 = pow(f_changed, radius_exponent) * (data[previous, 20] + l0)
            f_terminal = f_terminal * f_changed
            f_terminal_sister = f_terminal_sister * f_changed
            f_parent = f_parent * f_changed
        else:
            f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((terminal_flow+data[previous, 22])*r0)),
                                     (gamma/4.0))), (-1.0/gamma))
            f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt,25])/((terminal_flow+data[previous,22])*r0)),
                                      (-gamma/4.0))), (-1.0/gamma))
            f_terminal = f_terminal*f_changed
            f_terminal_sister = f_terminal_sister*f_changed
            f_parent = f_parent*f_changed
            r0 = ((8.0 * nu) / M_PI) * data[vessel,20] + pow(((pow(f_changed, 4.0) / r0) +
                                                              (pow(f_stagnant, 4.0) / data[alt, 25])), -1.0)
            l0 = pow(f_changed, radius_exponent) * (pow(data[previous, 20], length_exponent) + l0) + \
                 pow(f_stagnant, radius_exponent) *(pow(data[alt, 20], length_exponent) + data[alt, 27])
        previous = vessel
        vessel = idxs[vessel, 2]
    l0 = pow(data[0, 20], length_exponent) + l0
    r0 = pow((((data[0, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
    cost = M_PI * pow(r0, radius_exponent) * l0
    return cost

@cython.cpow(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def update_vessels_2(double[:] point, double[:,:] data, double[:] terminal, long[:,:] idxs,
                   int vessel, double gamma, double nu, double terminal_flow, double terminal_pressure,
                   double root_pressure, double radius_exponent, double length_exponent):
    cdef double upstream_length, downstream_length, terminal_length, r_terminal, r_terminal_sister, r_parent
    cdef double f_terminal, f_terminal_sister, r0, l0, f_changed, f_stagnant, f_parent, cost
    cdef vector[vector[double]] bifurcation_ratios
    cdef vector[double] tmp_bifurcation_ratios = [0.0, 0.0]
    cdef vector[int] main_idx
    cdef vector[int] alt_idx
    cdef vector[double] main_scale
    cdef vector[double] alt_scale
    cdef vector[double] reduced_resistance
    cdef vector[double] reduced_length
    cdef vector[double] flows
    cdef int previous, position, alt, alt_position
    cdef double[3] upstream_diff, downstream_diff, terminal_diff
    diff(point, data[vessel, 0:3], upstream_diff)
    diff(point, data[vessel, 3:6], downstream_diff)
    diff(point, terminal, terminal_diff)
    upstream_length = norm3(upstream_diff[0], upstream_diff[1], upstream_diff[2])
    downstream_length = norm3(downstream_diff[0], downstream_diff[1], downstream_diff[2])
    terminal_length = norm3(terminal_diff[0], terminal_diff[1], terminal_diff[2])
    r_terminal = ((8.0 * nu) / M_PI) * terminal_length
    r_terminal_sister = ((8.0 * nu) / M_PI) * downstream_length + (data[vessel, 25] -
                                                                   ((8.0 * nu) / M_PI) * data[vessel, 20])
    f_terminal = pow(1+pow(((data[vessel, 22] * r_terminal_sister) / (terminal_flow * r_terminal)), gamma/4.0),
                     -1.0/gamma)
    f_terminal_sister = pow(1+pow(((data[vessel, 22]*r_terminal_sister) / (terminal_flow * r_terminal)), -gamma/4.0),
                            -1.0/gamma)
    f_parent = 1.0
    r0 = ((8.0 * nu) / M_PI) * upstream_length + pow(((pow(f_terminal, 4.0))/r_terminal +
                                                     (pow(f_terminal_sister, 4.0))/r_terminal_sister), -1.0)
    r_parent = r0
    l0 = pow(f_terminal, radius_exponent) * pow(terminal_length, length_exponent) + \
         pow(f_terminal_sister, radius_exponent) * (pow(downstream_length, length_exponent) + data[vessel, 27])
    if vessel == 0:
        l0 = pow(upstream_length, length_exponent) + l0
        r0 = pow((((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)), (1.0/4.0))
        return reduced_resistance, reduced_length, main_idx, main_scale, alt_idx, alt_scale, bifurcation_ratios, flows, r0, l0
    previous = vessel
    vessel = idxs[vessel, 2]
    if idxs[vessel, 0] == previous:
        alt = idxs[vessel, 1]
        if alt == -1:
            alt = -2
        position = 0
        alt_position = 1
    else:
        alt = idxs[vessel, 0]
        if alt == -1:
            alt = -2
        position = 1
        alt_position = 0
    if alt == -2:
        f_changed = 1.0
        f_stagnant = 0.0
        f_terminal = f_terminal * f_changed
        f_terminal_sister = f_terminal_sister * f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
        l0 = pow(f_changed, radius_exponent) * (pow(upstream_length, length_exponent) + l0)
    else:
        f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)),
                                 (gamma/4.0))), (-1.0/gamma))
        f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((data[previous, 22]+terminal_flow)*r0)),
                                  (-gamma/4.0))), (-1.0/gamma))
        f_terminal = f_terminal * f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_changed
        r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + pow(((pow(f_changed, 4.0)) /r0 +
                                                           (pow(f_stagnant, 4.0)) /data[alt, 25]), -1.0)
        l0 = pow(f_changed, radius_exponent) *(pow(upstream_length, length_exponent) + l0) + \
             pow(f_stagnant, radius_exponent) * (pow(data[alt, 20], length_exponent) + data[alt, 27])
    tmp_bifurcation_ratios[position] = f_changed
    tmp_bifurcation_ratios[alt_position] = f_stagnant
    if vessel == 0:
        main_idx.push_back(vessel)
        main_scale.push_back(1.0)
        alt_idx.push_back(alt)
        alt_scale.push_back(f_stagnant)
        reduced_resistance.push_back(r0)
        reduced_length.push_back(l0)
        tmp_bifurcation_ratios[position] = f_changed
        tmp_bifurcation_ratios[alt_position] = f_stagnant
        bifurcation_ratios.push_back(tmp_bifurcation_ratios)
        flows.push_back(data[vessel, 22] + terminal_flow)
        l0 = pow(data[vessel, 20], length_exponent) + l0
        r0 = pow((((data[vessel, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
        return reduced_resistance, reduced_length, main_idx, main_scale, alt_idx, alt_scale, bifurcation_ratios, flows, r0, l0
    else:
        main_idx.push_back(vessel)
        alt_idx.push_back(alt)
        main_scale.push_back(1.0)
        alt_scale.push_back(f_stagnant)
        reduced_resistance.push_back(r0)
        reduced_length.push_back(l0)
        tmp_bifurcation_ratios[position] = f_changed
        tmp_bifurcation_ratios[alt_position] = f_stagnant
        bifurcation_ratios.push_back(tmp_bifurcation_ratios)
        flows.push_back(data[vessel, 22] + terminal_flow)
    # All vessels past this point are above the triad
    previous = vessel
    vessel = idxs[vessel, 2]
    while vessel >= 0:
        if idxs[vessel, 0] == previous:
            alt = idxs[vessel, 1]
            if alt == -1:
                alt = -2
            position = 0
            alt_position = 1
        else:
            alt = idxs[vessel, 0]
            if alt == -1:
                alt = -2
            position = 1
            alt_position = 0
        if alt == -2:
            f_changed = 1.0
            f_stagnant = 0.0
            r0 = ((8.0 * nu) / M_PI) * data[vessel, 20] + r0
            l0 = pow(f_changed, radius_exponent) * (data[previous, 20] + l0)
            f_terminal = f_terminal * f_changed
            f_terminal_sister = f_terminal_sister * f_changed
            f_parent = f_parent * f_changed
        else:
            f_changed = pow((1.0+pow(((data[alt, 22]*data[alt, 25])/((terminal_flow+data[previous, 22])*r0)),
                                     (gamma/4.0))), (-1.0/gamma))
            f_stagnant = pow((1.0+pow(((data[alt, 22]*data[alt,25])/((terminal_flow+data[previous,22])*r0)),
                                      (-gamma/4.0))), (-1.0/gamma))
            f_terminal = f_terminal * f_changed
            f_terminal_sister = f_terminal_sister * f_changed
            f_parent = f_parent * f_changed
            r0 = ((8.0 * nu) / M_PI) * data[vessel,20] + pow(((pow(f_changed, 4.0) / r0) +
                                                              (pow(f_stagnant, 4.0) / data[alt, 25])), -1.0)
            l0 = pow(f_changed, radius_exponent) * (pow(data[previous, 20], length_exponent) + l0) + \
                 pow(f_stagnant, radius_exponent) *(pow(data[alt, 20], length_exponent) + data[alt, 27])
        for j in range(main_scale.size()):
            main_scale[j] = main_scale[j] * f_changed
            alt_scale[j] = alt_scale[j] * f_changed
        main_idx.push_back(vessel)
        alt_idx.push_back(alt)
        main_scale.push_back(1.0)
        alt_scale.push_back(f_stagnant)
        reduced_resistance.push_back(r0)
        reduced_length.push_back(l0)
        tmp_bifurcation_ratios[position] = f_changed
        tmp_bifurcation_ratios[alt_position] = f_stagnant
        bifurcation_ratios.push_back(tmp_bifurcation_ratios)
        flows.push_back(data[vessel, 22] + terminal_flow)
        previous = vessel
        vessel = idxs[vessel, 2]
    l0 = pow(data[0, 20], length_exponent) + l0
    r0 = pow((((data[0, 22]+terminal_flow)*r0)/(root_pressure-terminal_pressure)), (1.0/4.0))
    return reduced_resistance, reduced_length, main_idx, main_scale, alt_idx, alt_scale, bifurcation_ratios, flows, r0, l0