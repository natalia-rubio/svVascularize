import numpy as np

def get_angles(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    v1 = np.asarray(vector1, dtype=float)
    v2 = np.asarray(vector2, dtype=float)
    dot = np.einsum('ij,ij->i', v1, v2)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2

    # Match Cython behavior: if denom < 1e-15, return angle = 0.0
    # Otherwise compute arccos of clamped ratio
    angles = np.zeros(len(v1), dtype=np.float64)
    valid = denom >= 1e-15

    if np.any(valid):
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = dot[valid] / denom[valid]
        ratio = np.clip(ratio, -1.0, 1.0)
        angles[valid] = np.degrees(np.arccos(ratio))

    return angles

