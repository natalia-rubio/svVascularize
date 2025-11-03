import numpy as np

def update_resistance(data: np.ndarray, idx: np.ndarray, gamma: float, nu: float):
    size = data.shape[0]
    vessels = list(range(size))
    max_depth = int(np.max(data[:, 26])) if size else 0
    while vessels:
        tmp = []
        for i in vessels:
            if data[i, 26] != max_depth:
                tmp.append(i)
                continue
            # leaves / single child / full bifurcation
            if np.isnan(data[i, 15:17]).all():
                data[i, 25] = (8 * nu / np.pi) * data[i, 20]
                data[i, 27] = 0.0
            elif np.isnan(data[i, 15]):
                right = int(idx[i, 1])
                data[i, 25] = (8 * nu / np.pi) * data[i, 20] + data[right, 25]
                data[i, 23] = 0.0
                data[i, 24] = 1.0
                data[i, 27] = data[right, 20] + data[right, 27]
            elif np.isnan(data[i, 16]):
                left = int(idx[i, 0])
                data[i, 25] = (8 * nu / np.pi) * data[i, 20] + data[left, 25]
                data[i, 23] = 1.0
                data[i, 24] = 0.0
                data[i, 27] = data[left, 20] + data[left, 27]
            else:
                left = int(idx[i, 0])
                right = int(idx[i, 1])
                lr = ((data[left, 22] * data[left, 25]) / (data[right, 22] * data[right, 25])) ** 0.25
                lbif = (1.0 + (lr ** -gamma)) ** (-1.0 / gamma)
                rbif = (1.0 + (lr ** gamma)) ** (-1.0 / gamma)
                data[i, 25] = (8 * nu / np.pi) * data[i, 20] + 1.0 / (
                    (lbif ** 4.0) / data[left, 25] + (rbif ** 4.0) / data[right, 25]
                )
                data[i, 23] = lbif
                data[i, 24] = rbif
                data[i, 27] = (lbif ** 2.0) * (data[left, 20] + data[left, 27]) + (rbif ** 2.0) * (
                    data[right, 20] + data[right, 27]
                )
        vessels = tmp


def update_radii(data: np.ndarray, idx: np.ndarray, root_pressure: float, terminal_pressure: float):
    if data.shape[0] == 0:
        return
    data[0, 21] = ((data[0, 25] * data[0, 22]) / (root_pressure - terminal_pressure)) ** 0.25
    left_vessels = [] if np.isnan(data[0, 15]) else [int(idx[0, 0])]
    right_vessels = [] if np.isnan(data[0, 16]) else [int(idx[0, 1])]
    while left_vessels or right_vessels:
        tmp_left = []
        tmp_right = []
        for i in left_vessels:
            data[i, 21] = data[int(idx[i, 3]), 23] * data[int(idx[i, 3]), 21]
            if not np.isnan(data[i, 15]):
                tmp_left.append(int(idx[i, 0]))
            if not np.isnan(data[i, 16]):
                tmp_right.append(int(idx[i, 1]))
        for i in right_vessels:
            data[i, 21] = data[int(idx[i, 3]), 24] * data[int(idx[i, 3]), 21]
            if not np.isnan(data[i, 15]):
                tmp_left.append(int(idx[i, 0]))
            if not np.isnan(data[i, 16]):
                tmp_right.append(int(idx[i, 1]))
        left_vessels = tmp_left
        right_vessels = tmp_right

