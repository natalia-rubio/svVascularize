import numpy as np

def basis(point_0: np.ndarray, point_1: np.ndarray):
    p0 = np.asarray(point_0, dtype=float)
    p1 = np.asarray(point_1, dtype=float)
    direction = p1 - p0
    mag = np.linalg.norm(direction, axis=1).reshape(-1, 1)
    # Match Cython behavior: no zero-magnitude check
    # (will produce NaN/Inf for identical points, same as Cython)
    w = direction / mag
    u = np.zeros_like(w)
    v = np.zeros_like(w)
    mask = (w[:, 2] == -1.0)
    u[mask, 0] = -1.0
    v[mask, 1] = -1.0
    inv = 1.0 / (1.0 + w[~mask, 2:3])
    u[~mask, 0] = 1.0 - (w[~mask, 0] ** 2.0) * inv[:, 0]
    u[~mask, 1] = -(w[~mask, 0] * w[~mask, 1]) * inv[:, 0]
    u[~mask, 2] = -w[~mask, 0]
    v[~mask, 0] = -(w[~mask, 0] * w[~mask, 1]) * inv[:, 0]
    v[~mask, 1] = 1.0 - (w[~mask, 1] ** 2.0) * inv[:, 0]
    v[~mask, 2] = -w[~mask, 1]
    return u, v, w


def basis_inplace(point_0, point_1, u_mv, v_mv, w_mv):
    u, v, w = basis(point_0, point_1)
    u_mv[:, :] = u
    v_mv[:, :] = v
    w_mv[:, :] = w

