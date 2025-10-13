import numpy as np

def close_exact_points(data: np.ndarray, points: np.ndarray):
    i = data.shape[0]
    j = points.shape[0]
    dirs = data[:, 3:6] - data[:, 0:3]
    lens = np.linalg.norm(dirs, axis=1).reshape(-1, 1)
    lens = np.where(lens == 0, 1.0, lens)
    dirs = dirs / lens
    closest_indices = np.zeros(j, dtype=np.int64)
    parametric_values = np.zeros(j, dtype=float)
    for jj in range(j):
        p = points[jj, :]
        ss = np.einsum('ij,ij->i', dirs, (data[:, 0:3] - p))
        tt = np.einsum('ij,ij->i', dirs, (p - data[:, 3:6]))
        hh = np.maximum(0.0, np.maximum(ss, tt))
        tmp = (p - data[:, 0:3])
        cc = np.cross(tmp, dirs)
        cd = np.linalg.norm(cc, axis=1)
        dist = np.sqrt(hh * hh + cd * cd)
        ii = int(np.argmin(dist))
        closest_indices[jj] = ii
        denom = ss[ii] + tt[ii]
        t = 0.0 if denom == 0 else ss[ii] / denom
        parametric_values[jj] = t
    return closest_indices, parametric_values

