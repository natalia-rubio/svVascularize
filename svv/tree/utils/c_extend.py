from typing import Dict, List, Tuple
import numpy as np

def build_c_vessel_map(vessel_map):
    # Python fallback: just return the same mapping
    return vessel_map


def update_alt(
    reduced_resistance: np.ndarray,
    reduced_length: np.ndarray,
    main_idx: List[int],
    main_scale: np.ndarray,
    alt_idx: List[int],
    alt_scale: np.ndarray,
    bifurcation_ratios: np.ndarray,
    flows: np.ndarray,
    root_radius: float,
    data: np.ndarray,
    tmp_28: np.ndarray,
    vessel_map,
):
    # Prepare change logs
    change_i: List[int] = []
    change_j: List[int] = []
    new_data: List[float] = []
    old_data: List[float] = []

    # Root radius update
    change_i.append(0); change_j.append(21)
    new_data.append(float(root_radius)); old_data.append(float(data[0, 21]))

    # Main vessels updates
    for i, idx in enumerate(main_idx):
        # flows
        change_i.append(idx); change_j.append(22)
        new_data.append(float(flows[i])); old_data.append(float(data[idx, 22]))
        # reduced_resistance
        change_i.append(idx); change_j.append(25)
        new_data.append(float(reduced_resistance[i])); old_data.append(float(data[idx, 25]))
        # reduced_length
        change_i.append(idx); change_j.append(27)
        new_data.append(float(reduced_length[i])); old_data.append(float(data[idx, 27]))
        # radius scaling
        change_i.append(idx); change_j.append(28)
        new_data.append(float(main_scale[i])); old_data.append(float(data[idx, 28]))
        # bifurcation ratios two columns (23,24)
        change_i.append(idx); change_j.append(23)
        new_data.append(float(bifurcation_ratios[i, 0])); old_data.append(float(data[idx, 23]))
        change_i.append(idx); change_j.append(24)
        new_data.append(float(bifurcation_ratios[i, 1])); old_data.append(float(data[idx, 24]))
        tmp_28[idx] = float(main_scale[i])

    # Alternate tree updates: propagate scale downstream
    for i, idx in enumerate(alt_idx):
        if idx <= -1:
            continue
        scale_factor = float(alt_scale[i]) / float(data[idx, 28])
        downstream_list = vessel_map[idx]['downstream']
        for d_idx in downstream_list:
            tmp_28[d_idx] *= scale_factor
            change_i.append(int(d_idx)); change_j.append(28)
            old_val = float(data[d_idx, 28])
            new_val = float(old_val * scale_factor)
            new_data.append(new_val); old_data.append(old_val)
        tmp_28[idx] = float(alt_scale[i])
        change_i.append(int(idx)); change_j.append(28)
        new_data.append(float(alt_scale[i])); old_data.append(float(data[idx, 28]))

    # Return vectors similar to cython version
    return change_i, change_j, new_data, old_data, tmp_28

