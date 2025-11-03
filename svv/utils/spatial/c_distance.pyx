# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython
from libc.math cimport sqrt
from libcpp.vector cimport vector

cnp.import_array()

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef dot(vector[double] a, vector[double] b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef norm(vector[double] a):
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void cross3d(const vector[double] a, const vector[double] b, vector[double] out):
    """
    out = a x b (3D cross product)
    """
    out[0] = a[1]*b[2] - a[2]*b[1]
    out[1] = a[2]*b[0] - a[0]*b[2]
    out[2] = a[0]*b[1] - a[1]*b[0]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double norm3d(const vector[double] a):
    """
    Returns Euclidean norm of vector a
    """
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double dot3d(const vector[double] a, const vector[double] b):
    """
    Returns dot product: a . b
    """
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double point_to_segment_distance(
    double px, double py, double pz,
    double x0, double y0, double z0,
    double x1, double y1, double z1
) except -1.0:
    """
    Distance from point (px,py,pz) to the segment [(x0,y0,z0) -> (x1,y1,z1)].
    Returns the minimal Euclidean distance.
    """
    # -- cdef declarations at top --
    cdef double vx, vy, vz
    cdef double wx, wy, wz
    cdef double seg_len_sq
    cdef double proj
    cdef double cx, cy, cz
    cdef double dx, dy, dz

    # Vector from segment start to end
    vx = x1 - x0
    vy = y1 - y0
    vz = z1 - z0

    # Vector from segment start to point
    wx = px - x0
    wy = py - y0
    wz = pz - z0

    seg_len_sq = vx*vx + vy*vy + vz*vz

    if seg_len_sq < 1e-14:
        # Degenerate => both ends are the same point
        dx = px - x0
        dy = py - y0
        dz = pz - z0
        return sqrt(dx*dx + dy*dy + dz*dz)

    # Projection of w onto v, as a fraction of |v|^2
    proj = (wx*vx + wy*vy + wz*vz) / seg_len_sq
    if proj < 0.0:
        proj = 0.0
    elif proj > 1.0:
        proj = 1.0

    # Closest point on the segment
    cx = x0 + proj*vx
    cy = y0 + proj*vy
    cz = z0 + proj*vz

    # Distance from p to this closest point
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    return sqrt(dx*dx + dy*dy + dz*dz)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def minimum_segment_distance(
    double[:,:] data_0,
    double[:,:] data_1
):
    """
    Returns an (i, j) array of minimal distances between each pair of
    line segments in data_0 vs. data_1.

    Each row in data_k is [x0,y0,z0, x1,y1,z1].
    """
    # -- cdef declarations at top --
    cdef Py_ssize_t i, j
    cdef Py_ssize_t ii, jj

    cdef double EPS
    cdef vector[double] ab = [0,0,0]
    cdef vector[double] cd = [0,0,0]
    cdef vector[double] cross_ab_cd = [0,0,0]
    cdef double ab_ab, cd_cd, ab_cd_val, denominator
    cdef double cross_len_ab_cd
    cdef double best_d

    cdef double cax, cay, caz
    cdef double dx, dy, dz
    cdef double px1, py1, pz1
    cdef double px2, py2, pz2
    cdef double Ax0, Ay0, Az0
    cdef double Cx0, Cy0, Cz0
    cdef double ca_ab, ca_cd
    cdef double t_, s_

    cdef cnp.ndarray[cnp.float64_t, ndim=2] line_distance
    cdef double[:,:] dist_mv

    # -- initialization --
    i = data_0.shape[0]
    j = data_1.shape[0]
    EPS = 1e-14

    line_distance = np.zeros((i, j), dtype=np.float64)
    dist_mv = line_distance

    for ii in range(i):
        for jj in range(j):

            # Directions
            ab[0] = data_0[ii,3] - data_0[ii,0]
            ab[1] = data_0[ii,4] - data_0[ii,1]
            ab[2] = data_0[ii,5] - data_0[ii,2]

            cd[0] = data_1[jj,3] - data_1[jj,0]
            cd[1] = data_1[jj,4] - data_1[jj,1]
            cd[2] = data_1[jj,5] - data_1[jj,2]

            ab_ab = dot3d(ab, ab)
            cd_cd = dot3d(cd, cd)
            ab_cd_val = dot3d(ab, cd)

            denominator = ab_ab * cd_cd - ab_cd_val * ab_cd_val

            # (1) Handle degenerate segments
            if ab_ab < EPS and cd_cd < EPS:
                # Both are essentially points
                dx = data_0[ii,0] - data_1[jj,0]
                dy = data_0[ii,1] - data_1[jj,1]
                dz = data_0[ii,2] - data_1[jj,2]
                dist_mv[ii,jj] = sqrt(dx*dx + dy*dy + dz*dz)
                continue
            elif ab_ab < EPS:
                # Segment A is a point => point-to-segment distance
                dist_mv[ii,jj] = point_to_segment_distance(
                    data_0[ii,0], data_0[ii,1], data_0[ii,2],
                    data_1[jj,0], data_1[jj,1], data_1[jj,2],
                    data_1[jj,3], data_1[jj,4], data_1[jj,5]
                )
                continue
            elif cd_cd < EPS:
                # Segment B is a point => point-to-segment distance
                dist_mv[ii,jj] = point_to_segment_distance(
                    data_1[jj,0], data_1[jj,1], data_1[jj,2],
                    data_0[ii,0], data_0[ii,1], data_0[ii,2],
                    data_0[ii,3], data_0[ii,4], data_0[ii,5]
                )
                continue

            # (2) Check for parallel lines
            cross3d(ab, cd, cross_ab_cd)
            cross_len_ab_cd = norm3d(cross_ab_cd)
            #if cross_len_ab_cd < EPS:
            if abs(denominator) < EPS:
                # Parallel lines
                # => distance is min of (endpoints of A to B) and (endpoints of B to A)
                best_d = 1e15

                # A0 -> B
                dx = point_to_segment_distance(
                    data_0[ii,0], data_0[ii,1], data_0[ii,2],
                    data_1[jj,0], data_1[jj,1], data_1[jj,2],
                    data_1[jj,3], data_1[jj,4], data_1[jj,5]
                )
                if dx < best_d:
                    best_d = dx

                # A1 -> B
                dx = point_to_segment_distance(
                    data_0[ii,3], data_0[ii,4], data_0[ii,5],
                    data_1[jj,0], data_1[jj,1], data_1[jj,2],
                    data_1[jj,3], data_1[jj,4], data_1[jj,5]
                )
                if dx < best_d:
                    best_d = dx

                # B0 -> A
                dx = point_to_segment_distance(
                    data_1[jj,0], data_1[jj,1], data_1[jj,2],
                    data_0[ii,0], data_0[ii,1], data_0[ii,2],
                    data_0[ii,3], data_0[ii,4], data_0[ii,5]
                )
                if dx < best_d:
                    best_d = dx

                # B1 -> A
                dx = point_to_segment_distance(
                    data_1[jj,3], data_1[jj,4], data_1[jj,5],
                    data_0[ii,0], data_0[ii,1], data_0[ii,2],
                    data_0[ii,3], data_0[ii,4], data_0[ii,5]
                )
                if dx < best_d:
                    best_d = dx

                dist_mv[ii,jj] = best_d

            else:
                # (3) Skew or intersecting lines => standard formula
                #   t_ = ( (ab.cd)*(ca.cd) - (ca.ab)*(cd.cd) ) / denominator
                #   s_ = ( (ab.ab)*(ca.cd) - (ab.cd)*(ca.ab) ) / denominator

                # ca = A0 - C0
                Ax0 = data_0[ii,0]
                Ay0 = data_0[ii,1]
                Az0 = data_0[ii,2]

                Cx0 = data_1[jj,0]
                Cy0 = data_1[jj,1]
                Cz0 = data_1[jj,2]

                cax = Ax0 - Cx0
                cay = Ay0 - Cy0
                caz = Az0 - Cz0

                ca_ab = cax*ab[0] + cay*ab[1] + caz*ab[2]
                ca_cd = cax*cd[0] + cay*cd[1] + caz*cd[2]

                t_ = (ab_cd_val*ca_cd - ca_ab*cd_cd)/denominator
                s_ = (ab_ab*ca_cd - ab_cd_val*ca_ab)/denominator

                if t_ < 0.0:
                    t_ = 0.0
                elif t_ > 1.0:
                    t_ = 1.0

                if s_ < 0.0:
                    s_ = 0.0
                elif s_ > 1.0:
                    s_ = 1.0

                # p1 = A0 + t_* ab
                px1 = Ax0 + t_*ab[0]
                py1 = Ay0 + t_*ab[1]
                pz1 = Az0 + t_*ab[2]

                # p2 = C0 + s_* cd
                px2 = Cx0 + s_*cd[0]
                py2 = Cy0 + s_*cd[1]
                pz2 = Cz0 + s_*cd[2]

                dx = px1 - px2
                dy = py1 - py2
                dz = pz1 - pz2

                dist_mv[ii,jj] = sqrt(dx*dx + dy*dy + dz*dz)

    return line_distance

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def minimum_self_segment_distance(double[:,:] data) -> float:
    """
    Computes the minimal distance among all NON-ADJACENT pairs of segments
    in `data`. Each row is [x0, y0, z0, x1, y1, z1].

    We skip adjacent pairs (i.e. we do NOT compare segment i with i or i+1).

    Returns:
        float: The minimal distance found among non-adjacent segment pairs.
               If data has < 2 segments (or effectively no non-adjacent pairs),
               returns a large sentinel value (1e20).
    """
    cdef Py_ssize_t n = data.shape[0]
    # If fewer than 3 segments, there's no "non-adjacent pair" to compare
    if n < 2:
        return 1e20

    cdef Py_ssize_t i, j

    # Temporary vectors and scalars for the geometry
    cdef vector[double] ab = [0,0,0]
    cdef vector[double] cd = [0,0,0]
    cdef vector[double] cross_ab_cd = [0,0,0]
    cdef double ab_ab, cd_cd, ab_cd_val, denominator
    cdef double cross_len_ab_cd
    cdef double Ax0, Ay0, Az0
    cdef double Bx0, By0, Bz0
    cdef double cax, cay, caz
    cdef double ca_ab, ca_cd
    cdef double t_, s_
    cdef double dx, dy, dz
    cdef double px1, py1, pz1
    cdef double px2, py2, pz2

    cdef double best_dist = 1e20  # track minimal distance
    cdef double local_best = 1e15
    cdef double EPS = 1e-14

    # Loop over all non-adjacent pairs (i < j-1)
    # e.g., skip (i, i) and (i, i+1) since those are the same or adjacent
    for i in range(n):
        for j in range(i+2, n):
            # Build direction for segment i
            ab[0] = data[i,3] - data[i,0]
            ab[1] = data[i,4] - data[i,1]
            ab[2] = data[i,5] - data[i,2]

            # Build direction for segment j
            cd[0] = data[j,3] - data[j,0]
            cd[1] = data[j,4] - data[j,1]
            cd[2] = data[j,5] - data[j,2]

            ab_ab = dot3d(ab, ab)
            cd_cd = dot3d(cd, cd)
            ab_cd_val = dot3d(ab, cd)

            denominator = ab_ab * cd_cd - ab_cd_val * ab_cd_val

            # (1) Handle degenerate segments
            if ab_ab < EPS and cd_cd < EPS:
                # Both points
                dx = data[i,0] - data[j,0]
                dy = data[i,1] - data[j,1]
                dz = data[i,2] - data[j,2]
                best_dist = min(best_dist, sqrt(dx*dx + dy*dy + dz*dz))
                continue
            elif ab_ab < EPS:
                # Segment i is a point
                dx = point_to_segment_distance(
                    data[i,0], data[i,1], data[i,2],
                    data[j,0], data[j,1], data[j,2],
                    data[j,3], data[j,4], data[j,5]
                )
                best_dist = min(best_dist, dx)
                continue
            elif cd_cd < EPS:
                # Segment j is a point
                dx = point_to_segment_distance(
                    data[j,0], data[j,1], data[j,2],
                    data[i,0], data[i,1], data[i,2],
                    data[i,3], data[i,4], data[i,5]
                )
                best_dist = min(best_dist, dx)
                continue

            # (2) Check parallel
            cross3d(ab, cd, cross_ab_cd)
            cross_len_ab_cd = norm3d(cross_ab_cd)
            if cross_len_ab_cd < EPS:
                # parallel => do endpoints vs. other segment approach
                local_best = 1e15

                # i0 -> seg j
                dx = point_to_segment_distance(
                    data[i,0], data[i,1], data[i,2],
                    data[j,0], data[j,1], data[j,2],
                    data[j,3], data[j,4], data[j,5]
                )
                local_best = min(local_best, dx)

                # i1 -> seg j
                dx = point_to_segment_distance(
                    data[i,3], data[i,4], data[i,5],
                    data[j,0], data[j,1], data[j,2],
                    data[j,3], data[j,4], data[j,5]
                )
                local_best = min(local_best, dx)

                # j0 -> seg i
                dx = point_to_segment_distance(
                    data[j,0], data[j,1], data[j,2],
                    data[i,0], data[i,1], data[i,2],
                    data[i,3], data[i,4], data[i,5]
                )
                local_best = min(local_best, dx)

                # j1 -> seg i
                dx = point_to_segment_distance(
                    data[j,3], data[j,4], data[j,5],
                    data[i,0], data[i,1], data[i,2],
                    data[i,3], data[i,4], data[i,5]
                )
                local_best = min(local_best, dx)

                best_dist = min(best_dist, local_best)

            else:
                # (3) Skew or intersecting
                #   ca = i0 - j0
                Ax0 = data[i,0]
                Ay0 = data[i,1]
                Az0 = data[i,2]

                Bx0 = data[j,0]
                By0 = data[j,1]
                Bz0 = data[j,2]

                cax = Ax0 - Bx0
                cay = Ay0 - By0
                caz = Az0 - Bz0

                ca_ab = cax*ab[0] + cay*ab[1] + caz*ab[2]
                ca_cd = cax*cd[0] + cay*cd[1] + caz*cd[2]

                t_ = (ab_cd_val*ca_cd - ca_ab*cd_cd) / denominator
                s_ = (ab_ab*ca_cd - ab_cd_val*ca_ab) / denominator

                if t_ < 0.0:
                    t_ = 0.0
                elif t_ > 1.0:
                    t_ = 1.0
                if s_ < 0.0:
                    s_ = 0.0
                elif s_ > 1.0:
                    s_ = 1.0

                px1 = Ax0 + t_*ab[0]
                py1 = Ay0 + t_*ab[1]
                pz1 = Az0 + t_*ab[2]

                px2 = Bx0 + s_*cd[0]
                py2 = By0 + s_*cd[1]
                pz2 = Bz0 + s_*cd[2]

                dx = px1 - px2
                dy = py1 - py2
                dz = pz1 - pz2
                best_dist = min(best_dist, sqrt(dx*dx + dy*dy + dz*dz))

    return best_dist