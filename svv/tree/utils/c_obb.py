import numpy as np

def _dot(a, b):
    return float(np.dot(a, b))


def _separating_axis(position, plane, u1, v1, w1, u2, v2, w2, u1s, v1s, w1s, u2s, v2s, w2s):
    lhs = abs(_dot(position, plane))
    rhs = (
        abs(u1s * _dot(u1, plane)) +
        abs(v1s * _dot(v1, plane)) +
        abs(w1s * _dot(w1, plane)) +
        abs(u2s * _dot(u2, plane)) +
        abs(v2s * _dot(v2, plane)) +
        abs(w2s * _dot(w2, plane))
    )
    return lhs > rhs


def obb_any(data: np.ndarray, vessels: np.ndarray) -> bool:
    num_data = data.shape[0]
    num_vessels = vessels.shape[0]
    for i_v in range(num_vessels):
        center1 = (vessels[i_v, 0:3] + vessels[i_v, 3:6]) * 0.5
        position = np.zeros(3)
        u1 = vessels[i_v, 6:9]
        v1 = vessels[i_v, 9:12]
        w1 = vessels[i_v, 12:15]
        u1s = float(vessels[i_v, 21])
        v1s = float(vessels[i_v, 21])
        w1s = float(vessels[i_v, 20] * 0.5)
        for i_d in range(num_data):
            center2 = (data[i_d, 0:3] + data[i_d, 3:6]) * 0.5
            position[:] = center2 - center1
            u2 = data[i_d, 6:9]
            v2 = data[i_d, 9:12]
            w2 = data[i_d, 12:15]
            u2s = float(data[i_d, 21])
            v2s = float(data[i_d, 21])
            w2s = float(data[i_d, 20] * 0.5)

            # Vessel axes
            if _separating_axis(position, u1, u1, v1, w1, u2, v2, w2, u1s, v1s, w1s, u2s, v2s, w2s):
                continue
            if _separating_axis(position, v1, u1, v1, w1, u2, v2, w2, u1s, v1s, w1s, u2s, v2s, w2s):
                continue
            if _separating_axis(position, w1, u1, v1, w1, u2, v2, w2, u1s, v1s, w1s, u2s, v2s, w2s):
                continue
            # Data axes
            if _separating_axis(position, u2, u1, v1, w1, u2, v2, w2, u1s, v1s, w1s, u2s, v2s, w2s):
                continue
            if _separating_axis(position, v2, u1, v1, w1, u2, v2, w2, u1s, v1s, w1s, u2s, v2s, w2s):
                continue
            if _separating_axis(position, w2, u1, v1, w1, u2, v2, w2, u1s, v1s, w1s, u2s, v2s, w2s):
                continue

            # Cross-product axes (skip zero vectors)
            separated = False
            for a in (u1, v1, w1):
                for b in (u2, v2, w2):
                    plane = np.cross(a, b)
                    if not np.any(plane):
                        continue
                    if _separating_axis(position, plane, u1, v1, w1, u2, v2, w2, u1s, v1s, w1s, u2s, v2s, w2s):
                        separated = True
                        break
                if separated:
                    break
            if separated:
                continue

            # No separating axis found -> collision
            return True
    return False
