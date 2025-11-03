import numpy as np

def sphere_proximity(data: np.ndarray, vessels: np.ndarray):
    centers0 = (data[:, 0:3] + data[:, 3:6]) * 0.5
    centers1 = (vessels[0:3] + vessels[3:6]) * 0.5
    r0 = np.sqrt(data[:, 21] ** 2.0 + (data[:, 20] * 0.5) ** 2.0)
    r1 = float(np.sqrt(vessels[21] ** 2.0 + (vessels[20] * 0.5) ** 2.0))
    d = np.linalg.norm(centers0 - centers1, axis=1)
    return (d < (r0 + r1))


def close(data: np.ndarray, point: np.ndarray, n: int = 20):
    centers = (data[:, 0:3] + data[:, 3:6]) * 0.5
    diff = centers - point.reshape(1, 3)
    d = np.linalg.norm(diff, axis=1)
    order = np.argsort(d)
    n = min(n, data.shape[0])
    return order[:n], d[order[:n]]


def _line_directions(data: np.ndarray) -> np.ndarray:
    # Match Cython implementation: use precomputed direction in columns 12:15
    return data[:, 12:15]


def close_exact_point_sort(data: np.ndarray, point: np.ndarray):
    dirs = _line_directions(data)
    ss = np.einsum('ij,ij->i', dirs, (data[:, 0:3] - point.reshape(1, 3)))
    tt = np.einsum('ij,ij->i', dirs, (point.reshape(1, 3) - data[:, 3:6]))
    hh = np.maximum(0.0, np.maximum(ss, tt))
    tmp = (point.reshape(1, 3) - data[:, 0:3])
    cc = np.cross(tmp, dirs)
    cd = np.linalg.norm(cc, axis=1)
    line_distance = np.sqrt(hh * hh + cd * cd)
    vessels = np.argsort(line_distance)
    return vessels, line_distance


def close_exact_point(data: np.ndarray, point: np.ndarray):
    dirs = _line_directions(data)
    i = data.shape[0]
    line_distance = np.zeros(i, dtype=float)
    for ii in range(i):
        d = dirs[ii, :]
        ss = d[0] * (data[ii, 0] - point[0]) + d[1] * (data[ii, 1] - point[1]) + d[2] * (data[ii, 2] - point[2])
        tt = d[0] * (point[0] - data[ii, 3]) + d[1] * (point[1] - data[ii, 4]) + d[2] * (point[2] - data[ii, 5])
        hh = max(0.0, ss, tt)
        tmp0 = point[0] - data[ii, 0]
        tmp1 = point[1] - data[ii, 1]
        tmp2 = point[2] - data[ii, 2]
        cc = np.cross(np.array([tmp0, tmp1, tmp2]), d)
        cd = float(np.sqrt(cc[0] ** 2.0 + cc[1] ** 2.0 + cc[2] ** 2.0))
        line_distance[ii] = float(np.sqrt(hh ** 2.0 + cd ** 2.0))
    return line_distance


def close_exact_points(data: np.ndarray, points: np.ndarray):
    # Return line_distance (i, j) subtracting vessel radius, matching Cython
    dirs = _line_directions(data)
    i = data.shape[0]
    j = points.shape[0]
    line_distance = np.zeros((i, j), dtype=float)
    for ii in range(i):
        d = dirs[ii, :]
        ss = d[0] * (data[ii, 0] - points[:, 0]) + d[1] * (data[ii, 1] - points[:, 1]) + d[2] * (data[ii, 2] - points[:, 2])
        tt = d[0] * (points[:, 0] - data[ii, 3]) + d[1] * (points[:, 1] - data[ii, 4]) + d[2] * (points[:, 2] - data[ii, 5])
        hh = np.maximum(0.0, np.maximum(ss, tt))
        tmp = points - data[ii, 0:3]
        cc = np.cross(tmp, d)
        cd = np.linalg.norm(cc, axis=1)
        line_distance[ii, :] = np.sqrt(hh * hh + cd * cd) - data[ii, 21]
    return line_distance


def close_exact_points_sort(data: np.ndarray, points: np.ndarray):
    dirs = _line_directions(data)
    i = data.shape[0]
    j = points.shape[0]
    line_distance = np.zeros((i, j), dtype=float)
    for ii in range(i):
        d = dirs[ii, :]
        ss = d[0] * (data[ii, 0] - points[:, 0]) + d[1] * (data[ii, 1] - points[:, 1]) + d[2] * (data[ii, 2] - points[:, 2])
        tt = d[0] * (points[:, 0] - data[ii, 3]) + d[1] * (points[:, 1] - data[ii, 4]) + d[2] * (points[:, 2] - data[ii, 5])
        hh = np.maximum(0.0, np.maximum(ss, tt))
        tmp = points - data[ii, 0:3]
        cc = np.cross(tmp, d)
        cd = np.linalg.norm(cc, axis=1)
        line_distance[ii, :] = np.sqrt(hh * hh + cd * cd) - data[ii, 21]
    vessels = np.argsort(line_distance, axis=0)
    return vessels, line_distance
