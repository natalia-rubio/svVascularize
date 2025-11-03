import numpy as np

def _point_to_segment_distance(px, py, pz, x0, y0, z0, x1, y1, z1) -> float:
    vx, vy, vz = x1 - x0, y1 - y0, z1 - z0
    wx, wy, wz = px - x0, py - y0, pz - z0
    seg_len_sq = vx*vx + vy*vy + vz*vz
    if seg_len_sq < 1e-14:
        dx, dy, dz = px - x0, py - y0, pz - z0
        return float(np.sqrt(dx*dx + dy*dy + dz*dz))
    proj = (wx*vx + wy*vy + wz*vz) / seg_len_sq
    proj = 0.0 if proj < 0.0 else (1.0 if proj > 1.0 else proj)
    cx, cy, cz = x0 + proj*vx, y0 + proj*vy, z0 + proj*vz
    dx, dy, dz = px - cx, py - cy, pz - cz
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))


def minimum_segment_distance(data_0: np.ndarray, data_1: np.ndarray) -> np.ndarray:
    i, j = data_0.shape[0], data_1.shape[0]
    out = np.zeros((i, j), dtype=float)
    for ii in range(i):
        a0 = data_0[ii, 0:3]
        a1 = data_0[ii, 3:6]
        ab = a1 - a0
        ab_ab = float(np.dot(ab, ab))
        for jj in range(j):
            c0 = data_1[jj, 0:3]
            c1 = data_1[jj, 3:6]
            cd = c1 - c0
            cd_cd = float(np.dot(cd, cd))
            # degenerate cases
            if ab_ab < 1e-14 and cd_cd < 1e-14:
                out[ii, jj] = float(np.linalg.norm(a0 - c0))
                continue
            if ab_ab < 1e-14:
                out[ii, jj] = _point_to_segment_distance(*a0, *c0, *c1)
                continue
            if cd_cd < 1e-14:
                out[ii, jj] = _point_to_segment_distance(*c0, *a0, *a1)
                continue
            ab_cd = float(np.dot(ab, cd))
            denom = ab_ab * cd_cd - ab_cd * ab_cd
            if abs(denom) < 1e-14:
                # parallel, check endpoints
                best = min(
                    _point_to_segment_distance(*a0, *c0, *c1),
                    _point_to_segment_distance(*a1, *c0, *c1),
                    _point_to_segment_distance(*c0, *a0, *a1),
                    _point_to_segment_distance(*c1, *a0, *a1),
                )
                out[ii, jj] = best
            else:
                ca = a0 - c0
                ca_ab = float(np.dot(ca, ab))
                ca_cd = float(np.dot(ca, cd))
                t_ = (ab_cd * ca_cd - ca_ab * cd_cd) / denom
                s_ = (ab_ab * ca_cd - ab_cd * ca_ab) / denom
                t_ = 0.0 if t_ < 0.0 else (1.0 if t_ > 1.0 else t_)
                s_ = 0.0 if s_ < 0.0 else (1.0 if s_ > 1.0 else s_)
                p1 = a0 + t_ * ab
                p2 = c0 + s_ * cd
                out[ii, jj] = float(np.linalg.norm(p1 - p2))
    return out


def minimum_self_segment_distance(data: np.ndarray) -> float:
    n = data.shape[0]
    if n < 2:
        return 1e20
    best = 1e20
    for i in range(n):
        a0 = data[i, 0:3]
        a1 = data[i, 3:6]
        for j in range(i+2, n):
            c0 = data[j, 0:3]
            c1 = data[j, 3:6]
            d = minimum_segment_distance(np.array([[*a0, *a1]]), np.array([[*c0, *c1]]) )[0, 0]
            if d < best:
                best = d
    return float(best)

