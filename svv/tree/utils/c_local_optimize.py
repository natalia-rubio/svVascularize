import numpy as np


def _norm3(dx: float, dy: float, dz: float) -> float:
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def basis3(point_0, point_1):
    """
    Python fallback for basis3 in c_local_optimize.pyx.

    Given two 3D points, returns (u, v, w) as Python lists of length 3.
    Matches the orientation and special-case logic of the .pyx version:
    - w points along (point_0 - point_1) normalized
    - if w[2] == -1.0, set u=[-1,0,0], v=[0,-1,0]
    - else compute u, v via the same closed-form expressions
    """
    p0 = np.asarray(point_0, dtype=float).reshape(-1)
    p1 = np.asarray(point_1, dtype=float).reshape(-1)
    diff = p0 - p1
    mag = _norm3(diff[0], diff[1], diff[2])
    # Match Cython behavior if mag == 0.0 (will produce inf/NaN)
    w0 = diff[0] / mag
    w1 = diff[1] / mag
    w2 = diff[2] / mag

    if w2 == -1.0:
        u = [-1.0, 0.0, 0.0]
        v = [0.0, -1.0, 0.0]
    else:
        denom = 1.0 + w2
        u = [0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0]
        u[0] = 1.0 - (w0 * w0) / denom
        u[1] = -(w0 * w1) / denom
        u[2] = -w0
        v[0] = -(w0 * w1) / denom
        v[1] = 1.0 - (w1 * w1) / denom
        v[2] = -w1
    w = [w0, w1, w2]
    return u, v, w


def _initial_lengths(point, data, terminal, vessel):
    p = np.asarray(point, dtype=float).reshape(3)
    upstream_vec = p - data[vessel, 0:3]
    downstream_vec = p - data[vessel, 3:6]
    terminal_vec = p - np.asarray(terminal, dtype=float).reshape(3)
    up = _norm3(upstream_vec[0], upstream_vec[1], upstream_vec[2])
    down = _norm3(downstream_vec[0], downstream_vec[1], downstream_vec[2])
    term = _norm3(terminal_vec[0], terminal_vec[1], terminal_vec[2])
    return up, down, term


def bifurcation_cost(point, data, terminal, idxs, vessel, gamma, nu,
                     terminal_flow, terminal_pressure, root_pressure,
                     radius_exponent, length_exponent):
    """
    Python fallback matching c_local_optimize.pyx::bifurcation_cost.

    Returns tuple:
    (r0, l0, f_terminal, f_terminal_sister, f_parent, point,
     reduced_resistances, reduced_lengths, bifurcations, flows,
     main_idx, alt_idx, new_scale, alt_scale, r_terminal, r_terminal_sister)
    """
    data = np.asarray(data, dtype=float)
    idxs = np.asarray(idxs, dtype=int)
    up_len, down_len, term_len = _initial_lengths(point, data, terminal, vessel)

    if up_len == 0.0 or down_len == 0.0 or term_len == 0.0:
        # Follow .pyx behavior: huge values propagate; here keep outputs consistent
        # by using numpy.inf in place of HUGE_VAL paths. We still return vectors.
        r_terminal = (8.0 * nu / np.pi) * term_len
        r_terminal_sister = (8.0 * nu / np.pi) * down_len + (data[vessel, 25] - (8.0 * nu / np.pi) * data[vessel, 20])
        return (np.inf, np.inf, np.nan, np.nan, np.nan, np.asarray(point, dtype=float),
                [], [], [], [], [], [], [], [], r_terminal, r_terminal_sister)

    r_terminal = (8.0 * nu / np.pi) * term_len
    r_terminal_sister = (8.0 * nu / np.pi) * down_len + (data[vessel, 25] - (8.0 * nu / np.pi) * data[vessel, 20])

    ratio = (data[vessel, 22] * r_terminal_sister) / (terminal_flow * r_terminal)
    f_terminal = (1.0 + ratio ** (gamma / 4.0)) ** (-1.0 / gamma)
    f_terminal_sister = (1.0 + ratio ** (-gamma / 4.0)) ** (-1.0 / gamma)

    r0 = (8.0 * nu / np.pi) * up_len + 1.0 / ((f_terminal ** 4.0) / r_terminal + (f_terminal_sister ** 4.0) / r_terminal_sister)
    l0 = (f_terminal ** radius_exponent) * (term_len ** length_exponent) + \
         (f_terminal_sister ** radius_exponent) * ((down_len ** length_exponent) + data[vessel, 27])

    reduced_resistances = [float(r0)]
    reduced_lengths = [float(l0)]
    bifurcations = [float(f_terminal_sister), float(f_terminal)]
    flows = [float(data[vessel, 22] + terminal_flow)]
    main_idx = [int(vessel)]
    alt_idx = []
    new_scale = []
    alt_scale = []

    if vessel == 0:
        l0 = (up_len ** length_exponent) + l0
        r0 = (((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)) ** 0.25
        alt_idx.append(-1)
        new_scale.append(1.0)
        alt_scale.append(1.0)
        return (float(r0), float(l0), float(f_terminal), float(f_terminal_sister), 1.0,
                np.asarray(point, dtype=float), reduced_resistances, reduced_lengths, bifurcations,
                flows, main_idx, alt_idx, new_scale, alt_scale, float(r_terminal), float(r_terminal_sister))

    previous = int(vessel)
    vessel = int(idxs[vessel, 2])

    # Determine alternative branch at first parent
    if int(idxs[vessel, 0]) == previous:
        alt = int(idxs[vessel, 1])
        position = 0
    else:
        alt = int(idxs[vessel, 0])
        position = 1
    alt = -2 if alt == -1 else alt
    alt_idx.append(int(alt))

    if alt == -2:
        f_changed = 1.0
        f_stagnant = 0.0
        f_terminal *= f_changed
        f_terminal_sister *= f_changed
        f_parent = f_changed
        r0 = (8.0 * nu / np.pi) * data[vessel, 20] + r0
        l0 = (f_changed ** radius_exponent) * ((up_len ** length_exponent) + l0)
    else:
        ratio = (data[alt, 22] * data[alt, 25]) / ((data[previous, 22] + terminal_flow) * r0)
        f_changed = (1.0 + ratio ** (gamma / 4.0)) ** (-1.0 / gamma)
        f_stagnant = (1.0 + ratio ** (-gamma / 4.0)) ** (-1.0 / gamma)
        f_terminal *= f_changed
        f_terminal_sister *= f_changed
        f_parent = f_changed
        r0 = (8.0 * nu / np.pi) * data[vessel, 20] + 1.0 / (
            (f_changed ** 4.0) / r0 + (f_stagnant ** 4.0) / data[alt, 25]
        )
        l0 = (f_changed ** radius_exponent) * ((up_len ** length_exponent) + l0) + \
             (f_stagnant ** radius_exponent) * ((data[alt, 20] ** length_exponent) + data[alt, 27])

    reduced_resistances.append(float(r0))
    reduced_lengths.append(float(l0))
    flows.append(float(data[vessel, 22] + terminal_flow))
    main_idx.append(int(vessel))
    new_scale.append(float(f_changed))
    alt_scale.append(float(f_stagnant))
    if position == 0:
        bifurcations.extend([float(f_changed), float(f_stagnant)])
    else:
        bifurcations.extend([float(f_stagnant), float(f_changed)])

    if vessel == 0:
        l0 = (data[vessel, 20] ** length_exponent) + l0
        r0 = (((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)) ** 0.25
        return (float(r0), float(l0), float(f_terminal), float(f_terminal_sister), float(f_parent),
                np.asarray(point, dtype=float), reduced_resistances, reduced_lengths, bifurcations, flows,
                main_idx, alt_idx, new_scale, alt_scale, float(r_terminal), float(r_terminal_sister))

    previous = int(vessel)
    vessel = int(idxs[vessel, 2])
    while vessel >= 0:
        if int(idxs[vessel, 0]) == previous:
            alt = int(idxs[vessel, 1])
            position = 0
        else:
            alt = int(idxs[vessel, 0])
            position = 1
        alt = -2 if alt == -1 else alt
        alt_idx.append(int(alt))

        if alt == -2:
            f_changed = 1.0
            f_stagnant = 0.0
            r0 = (8.0 * nu / np.pi) * data[vessel, 20] + r0
            l0 = (f_changed ** length_exponent) * (data[previous, 20] + l0)
            f_terminal *= f_changed
            f_terminal_sister *= f_changed
            f_parent *= f_changed
        else:
            ratio = (data[alt, 22] * data[alt, 25]) / ((terminal_flow + data[previous, 22]) * r0)
            f_changed = (1.0 + ratio ** (gamma / 4.0)) ** (-1.0 / gamma)
            f_stagnant = (1.0 + ratio ** (-gamma / 4.0)) ** (-1.0 / gamma)
            f_terminal *= f_changed
            f_terminal_sister *= f_changed
            f_parent *= f_changed
            r0 = (8.0 * nu / np.pi) * data[vessel, 20] + 1.0 / (
                (f_changed ** 4.0) / r0 + (f_stagnant ** 4.0) / data[alt, 25]
            )
            l0 = (f_changed ** radius_exponent) * ((data[previous, 20] ** length_exponent) + l0) + \
                 (f_stagnant ** radius_exponent) * ((data[alt, 20] ** length_exponent) + data[alt, 27])

        # propagate scale to existing entries
        for j in range(len(new_scale)):
            new_scale[j] = float(new_scale[j] * f_changed)
            alt_scale[j] = float(alt_scale[j] * f_stagnant)

        reduced_resistances.append(float(r0))
        reduced_lengths.append(float(l0))
        flows.append(float(data[vessel, 22] + terminal_flow))
        main_idx.append(int(vessel))
        new_scale.append(float(f_changed))
        alt_scale.append(float(f_stagnant))
        if position == 0:
            bifurcations.extend([float(f_changed), float(f_stagnant)])
        else:
            bifurcations.extend([float(f_stagnant), float(f_changed)])

        previous = int(vessel)
        vessel = int(idxs[vessel, 2])

    l0 = (data[0, 20] ** length_exponent) + l0
    r0 = (((data[0, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)) ** 0.25
    return (float(r0), float(l0), float(f_terminal), float(f_terminal_sister), float(f_parent),
            np.asarray(point, dtype=float), reduced_resistances, reduced_lengths, bifurcations, flows,
            main_idx, alt_idx, new_scale, alt_scale, float(r_terminal), float(r_terminal_sister))


def tree_cost(point, data, terminal, idxs, vessel, gamma, nu,
              terminal_flow, terminal_pressure, root_pressure,
              radius_exponent, length_exponent):
    data = np.asarray(data, dtype=float)
    idxs = np.asarray(idxs, dtype=int)
    up_len, down_len, term_len = _initial_lengths(point, data, terminal, vessel)
    if up_len == 0.0 or down_len == 0.0 or term_len == 0.0:
        return float(np.inf)

    r_terminal = (8.0 * nu / np.pi) * term_len
    r_terminal_sister = (8.0 * nu / np.pi) * down_len + (data[vessel, 25] - (8.0 * nu / np.pi) * data[vessel, 20])
    ratio = (data[vessel, 22] * r_terminal_sister) / (terminal_flow * r_terminal)
    f_terminal = (1.0 + ratio ** (gamma / 4.0)) ** (-1.0 / gamma)
    f_terminal_sister = (1.0 + ratio ** (-gamma / 4.0)) ** (-1.0 / gamma)
    r0 = (8.0 * nu / np.pi) * up_len + 1.0 / ((f_terminal ** 4.0) / r_terminal + (f_terminal_sister ** 4.0) / r_terminal_sister)
    l0 = (f_terminal ** radius_exponent) * (term_len ** length_exponent) + \
         (f_terminal_sister ** radius_exponent) * ((down_len ** length_exponent) + data[vessel, 27])
    if vessel == 0:
        l0 = (up_len ** length_exponent) + l0
        r0 = (((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)) ** 0.25
        return float(np.pi * (r0 ** radius_exponent) * l0)

    previous = int(vessel)
    vessel = int(idxs[vessel, 2])
    if int(idxs[vessel, 0]) == previous:
        alt = int(idxs[vessel, 1])
        position = 0
    else:
        alt = int(idxs[vessel, 0])
        position = 1
    alt = -2 if alt == -1 else alt
    if alt == -2:
        f_changed = 1.0
        f_stagnant = 0.0
        f_parent = f_changed
        r0 = (8.0 * nu / np.pi) * data[vessel, 20] + r0
        l0 = (f_changed ** radius_exponent) * ((up_len ** length_exponent) + l0)
    else:
        ratio = (data[alt, 22] * data[alt, 25]) / ((data[previous, 22] + terminal_flow) * r0)
        f_changed = (1.0 + ratio ** (gamma / 4.0)) ** (-1.0 / gamma)
        f_stagnant = (1.0 + ratio ** (-gamma / 4.0)) ** (-1.0 / gamma)
        f_parent = f_changed
        r0 = (8.0 * nu / np.pi) * data[vessel, 20] + 1.0 / (
            (f_changed ** 4.0) / r0 + (f_stagnant ** 4.0) / data[alt, 25]
        )
        l0 = (
            (f_changed ** radius_exponent) * ((up_len ** length_exponent) + l0)
            + (f_stagnant ** radius_exponent) * ((data[alt, 20] ** length_exponent) + data[alt, 27])
        )
    if vessel == 0:
        l0 = (data[vessel, 20] ** length_exponent) + l0
        r0 = (((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)) ** 0.25
        return float(np.pi * (r0 ** radius_exponent) * l0)

    previous = int(vessel)
    vessel = int(idxs[vessel, 2])
    while vessel >= 0:
        if int(idxs[vessel, 0]) == previous:
            alt = int(idxs[vessel, 1])
            position = 0
        else:
            alt = int(idxs[vessel, 0])
            position = 1
        alt = -2 if alt == -1 else alt
        if alt == -2:
            f_changed = 1.0
            f_stagnant = 0.0
            r0 = (8.0 * nu / np.pi) * data[vessel, 20] + r0
            l0 = (f_changed ** length_exponent) * (data[previous, 20] + l0)
        else:
            ratio = (data[alt, 22] * data[alt, 25]) / ((terminal_flow + data[previous, 22]) * r0)
            f_changed = (1.0 + ratio ** (gamma / 4.0)) ** (-1.0 / gamma)
            f_stagnant = (1.0 + ratio ** (-gamma / 4.0)) ** (-1.0 / gamma)
            r0 = (8.0 * nu / np.pi) * data[vessel, 20] + 1.0 / (
                (f_changed ** 4.0) / r0 + (f_stagnant ** 4.0) / data[alt, 25]
            )
            l0 = (f_changed ** radius_exponent) * ((data[previous, 20] ** length_exponent) + l0) + \
                 (f_stagnant ** radius_exponent) * ((data[alt, 20] ** length_exponent) + data[alt, 27])
        previous = int(vessel)
        vessel = int(idxs[vessel, 2])
    l0 = (data[0, 20] ** length_exponent) + l0
    r0 = (((data[0, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)) ** 0.25
    return float(np.pi * (r0 ** radius_exponent) * l0)


def create_new_vessels(point, data, terminal, terminal_vessel,
                       terminal_sister_vessel, terminal_parent_vessel, max_node,
                       max_vessel, idxs, vessel, gamma, nu, terminal_flow,
                       terminal_pressure, root_pressure, radius_exponent, length_exponent):
    data = np.asarray(data, dtype=float)
    idxs = np.asarray(idxs, dtype=int)
    original_vessel = int(vessel)

    up_len, down_len, term_len = _initial_lengths(point, data, terminal, original_vessel)
    r_terminal = (8.0 * nu / np.pi) * term_len
    r_terminal_sister = (8.0 * nu / np.pi) * down_len + (data[original_vessel, 25] - (8.0 * nu / np.pi) * data[original_vessel, 20])

    ratio = (data[original_vessel, 22] * r_terminal_sister) / (terminal_flow * r_terminal)
    f_terminal = (1.0 + ratio ** (gamma / 4.0)) ** (-1.0 / gamma)
    f_terminal_sister = (1.0 + ratio ** (-gamma / 4.0)) ** (-1.0 / gamma)
    terminal_bif = f_terminal
    sister_bif = f_terminal_sister

    f_parent = 1.0
    r0 = (8.0 * nu / np.pi) * up_len + 1.0 / ((f_terminal ** 4.0) / r_terminal + (f_terminal_sister ** 4.0) / r_terminal_sister)
    r_parent = r0
    l0 = (f_terminal ** radius_exponent) * (term_len ** length_exponent) + \
         (f_terminal_sister ** radius_exponent) * ((down_len ** length_exponent) + data[original_vessel, 27])
    rl_parent = l0

    p = np.asarray(point, dtype=float).reshape(3)
    t = np.asarray(terminal, dtype=float).reshape(3)

    if original_vessel == 0:
        l0 = (up_len ** length_exponent) + l0
        r0 = (((data[original_vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)) ** 0.25

        # terminal_vessel
        terminal_vessel[0, 0:3] = p
        terminal_vessel[0, 3:6] = t
        u, v, w = basis3(terminal_vessel[0, 0:3], terminal_vessel[0, 3:6])
        terminal_vessel[0, 6] = u[0]; terminal_vessel[0, 7] = u[1]; terminal_vessel[0, 8] = u[2]
        terminal_vessel[0, 9] = v[0]; terminal_vessel[0,10] = v[1]; terminal_vessel[0,11] = v[2]
        terminal_vessel[0,12] = w[0]; terminal_vessel[0,13] = w[1]; terminal_vessel[0,14] = w[2]
        terminal_vessel[0,17] = float(original_vessel)
        terminal_vessel[0,18] = float(max_node + 1.0)
        terminal_vessel[0,19] = float(max_node + 2.0)
        terminal_vessel[0,20] = float(term_len)
        terminal_vessel[0,21] = float(r0 * f_terminal)
        terminal_vessel[0,22] = float(terminal_flow)
        terminal_vessel[0,25] = float(r_terminal)
        terminal_vessel[0,26] = float(data[original_vessel, 26] + 1.0)
        terminal_vessel[0,27] = 0.0
        terminal_vessel[0,28] = float(f_terminal)

        # terminal_sister_vessel
        terminal_sister_vessel[0, 0:3] = p
        terminal_sister_vessel[0, 3:6] = data[original_vessel, 3:6]
        u, v, w = basis3(terminal_sister_vessel[0, 0:3], terminal_sister_vessel[0, 3:6])
        terminal_sister_vessel[0, 6] = u[0]; terminal_sister_vessel[0, 7] = u[1]; terminal_sister_vessel[0, 8] = u[2]
        terminal_sister_vessel[0, 9] = v[0]; terminal_sister_vessel[0,10] = v[1]; terminal_sister_vessel[0,11] = v[2]
        terminal_sister_vessel[0,12] = w[0]; terminal_sister_vessel[0,13] = w[1]; terminal_sister_vessel[0,14] = w[2]
        terminal_sister_vessel[0,15] = data[original_vessel, 15]
        terminal_sister_vessel[0,16] = data[original_vessel, 16]
        terminal_sister_vessel[0,17] = float(original_vessel)
        terminal_sister_vessel[0,18] = float(max_node + 1.0)
        terminal_sister_vessel[0,19] = float(data[original_vessel, 19])
        terminal_sister_vessel[0,20] = float(down_len)
        terminal_sister_vessel[0,21] = float(r0 * f_terminal_sister)
        terminal_sister_vessel[0,22] = float(data[original_vessel, 22])
        terminal_sister_vessel[0,23] = float(data[original_vessel, 23])
        terminal_sister_vessel[0,24] = float(data[original_vessel, 24])
        terminal_sister_vessel[0,25] = float(r_terminal_sister)
        terminal_sister_vessel[0,26] = float(data[original_vessel, 26] + 1.0)
        terminal_sister_vessel[0,27] = float(data[original_vessel, 27])
        terminal_sister_vessel[0,28] = float(f_terminal_sister)

        # terminal_parent_vessel
        terminal_parent_vessel[0, 0:3] = data[original_vessel, 0:3]
        terminal_parent_vessel[0, 3:6] = p
        u, v, w = basis3(terminal_parent_vessel[0, 0:3], terminal_parent_vessel[0, 3:6])
        terminal_parent_vessel[0, 6] = u[0]; terminal_parent_vessel[0, 7] = u[1]; terminal_parent_vessel[0, 8] = u[2]
        terminal_parent_vessel[0, 9] = v[0]; terminal_parent_vessel[0,10] = v[1]; terminal_parent_vessel[0,11] = v[2]
        terminal_parent_vessel[0,12] = w[0]; terminal_parent_vessel[0,13] = w[1]; terminal_parent_vessel[0,14] = w[2]
        terminal_parent_vessel[0,15] = float(max_vessel)
        terminal_parent_vessel[0,16] = float(max_vessel + 1.0)
        terminal_parent_vessel[0,17] = float(data[original_vessel, 17])
        terminal_parent_vessel[0,18] = float(data[original_vessel, 18])
        terminal_parent_vessel[0,19] = float(max_node + 1.0)
        terminal_parent_vessel[0,20] = float(up_len)
        terminal_parent_vessel[0,21] = float(r0 * f_parent)
        terminal_parent_vessel[0,22] = float(data[original_vessel, 22] + terminal_flow)
        terminal_parent_vessel[0,23] = float(terminal_bif)
        terminal_parent_vessel[0,24] = float(sister_bif)
        terminal_parent_vessel[0,25] = float(r_parent)
        terminal_parent_vessel[0,26] = float(data[original_vessel, 26])
        terminal_parent_vessel[0,27] = float(rl_parent)
        terminal_parent_vessel[0,28] = float(f_parent)
        return

    previous = original_vessel
    vessel = int(idxs[previous, 2])
    while vessel >= 0:
        if int(idxs[vessel, 0]) == previous:
            alt = int(idxs[vessel, 1])
            position = 0
        else:
            alt = int(idxs[vessel, 0])
            position = 1
        alt = -2 if alt == -1 else alt
        if alt == -2:
            f_changed = 1.0
            f_stagnant = 0.0
            r0 = (8.0 * nu / np.pi) * data[vessel, 20] + r0
            l0 = (f_changed ** length_exponent) * (data[previous, 20] + l0)
            f_terminal *= f_changed
            f_terminal_sister *= f_changed
            f_parent *= f_changed
        else:
            ratio = (data[alt, 22] * data[alt, 25]) / ((terminal_flow + data[previous, 22]) * r0)
            f_changed = (1.0 + ratio ** (gamma / 4.0)) ** (-1.0 / gamma)
            f_stagnant = (1.0 + ratio ** (-gamma / 4.0)) ** (-1.0 / gamma)
            f_terminal *= f_changed
            f_terminal_sister *= f_changed
            f_parent *= f_changed
            r0 = (8.0 * nu / np.pi) * data[vessel, 20] + 1.0 / (
                (f_changed ** 4.0) / r0 + (f_stagnant ** 4.0) / data[alt, 25]
            )
            l0 = (f_changed ** radius_exponent) * ((data[previous, 20] ** length_exponent) + l0) + \
                 (f_stagnant ** radius_exponent) * ((data[alt, 20] ** length_exponent) + data[alt, 27])
        previous = int(vessel)
        vessel = int(idxs[vessel, 2])

    l0 = (data[0, 20] ** length_exponent) + l0
    r0 = (((data[0, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)) ** 0.25

    # Write outputs (same as the vessel==0 block but now after climbing to root)
    terminal_vessel[0, 0:3] = p
    terminal_vessel[0, 3:6] = t
    u, v, w = basis3(terminal_vessel[0, 0:3], terminal_vessel[0, 3:6])
    terminal_vessel[0, 6] = u[0]; terminal_vessel[0, 7] = u[1]; terminal_vessel[0, 8] = u[2]
    terminal_vessel[0, 9] = v[0]; terminal_vessel[0,10] = v[1]; terminal_vessel[0,11] = v[2]
    terminal_vessel[0,12] = w[0]; terminal_vessel[0,13] = w[1]; terminal_vessel[0,14] = w[2]
    terminal_vessel[0,17] = float(original_vessel)
    terminal_vessel[0,18] = float(max_node + 1.0)
    terminal_vessel[0,19] = float(max_node + 2.0)
    terminal_vessel[0,20] = float(term_len)
    terminal_vessel[0,21] = float(r0 * f_terminal)
    terminal_vessel[0,22] = float(terminal_flow)
    terminal_vessel[0,25] = float(r_terminal)
    terminal_vessel[0,26] = float(data[original_vessel, 26] + 1.0)
    terminal_vessel[0,27] = 0.0
    terminal_vessel[0,28] = float(f_terminal)

    terminal_sister_vessel[0, 0:3] = p
    terminal_sister_vessel[0, 3:6] = data[original_vessel, 3:6]
    u, v, w = basis3(terminal_sister_vessel[0, 0:3], terminal_sister_vessel[0, 3:6])
    terminal_sister_vessel[0, 6] = u[0]; terminal_sister_vessel[0, 7] = u[1]; terminal_sister_vessel[0, 8] = u[2]
    terminal_sister_vessel[0, 9] = v[0]; terminal_sister_vessel[0,10] = v[1]; terminal_sister_vessel[0,11] = v[2]
    terminal_sister_vessel[0,12] = w[0]; terminal_sister_vessel[0,13] = w[1]; terminal_sister_vessel[0,14] = w[2]
    terminal_sister_vessel[0,15] = data[original_vessel, 15]
    terminal_sister_vessel[0,16] = data[original_vessel, 16]
    terminal_sister_vessel[0,17] = float(original_vessel)
    terminal_sister_vessel[0,18] = float(max_node + 1.0)
    terminal_sister_vessel[0,19] = float(data[original_vessel, 19])
    terminal_sister_vessel[0,20] = float(down_len)
    terminal_sister_vessel[0,21] = float(r0 * f_terminal_sister)
    terminal_sister_vessel[0,22] = float(data[original_vessel, 22])
    terminal_sister_vessel[0,23] = float(data[original_vessel, 23])
    terminal_sister_vessel[0,24] = float(data[original_vessel, 24])
    terminal_sister_vessel[0,25] = float(r_terminal_sister)
    terminal_sister_vessel[0,26] = float(data[original_vessel, 26] + 1.0)
    terminal_sister_vessel[0,27] = float(data[original_vessel, 27])
    terminal_sister_vessel[0,28] = float(f_terminal_sister)

    terminal_parent_vessel[0, 0:3] = data[original_vessel, 0:3]
    terminal_parent_vessel[0, 3:6] = p
    u, v, w = basis3(terminal_parent_vessel[0, 0:3], terminal_parent_vessel[0, 3:6])
    terminal_parent_vessel[0, 6] = u[0]; terminal_parent_vessel[0, 7] = u[1]; terminal_parent_vessel[0, 8] = u[2]
    terminal_parent_vessel[0, 9] = v[0]; terminal_parent_vessel[0,10] = v[1]; terminal_parent_vessel[0,11] = v[2]
    terminal_parent_vessel[0,12] = w[0]; terminal_parent_vessel[0,13] = w[1]; terminal_parent_vessel[0,14] = w[2]
    terminal_parent_vessel[0,15] = float(max_vessel)
    terminal_parent_vessel[0,16] = float(max_vessel + 1.0)
    terminal_parent_vessel[0,17] = float(data[original_vessel, 17])
    terminal_parent_vessel[0,18] = float(data[original_vessel, 18])
    terminal_parent_vessel[0,19] = float(max_node + 1.0)
    terminal_parent_vessel[0,20] = float(up_len)
    terminal_parent_vessel[0,21] = float(r0 * f_parent)
    terminal_parent_vessel[0,22] = float(data[original_vessel, 22] + terminal_flow)
    terminal_parent_vessel[0,23] = float(terminal_bif)
    terminal_parent_vessel[0,24] = float(sister_bif)
    terminal_parent_vessel[0,25] = float(r_parent)
    terminal_parent_vessel[0,26] = float(data[original_vessel, 26])
    terminal_parent_vessel[0,27] = float(rl_parent)
    terminal_parent_vessel[0,28] = float(f_parent)
    return


def update_vessels(point, data, terminal, idxs,
                   vessel, gamma, nu, terminal_flow, terminal_pressure,
                   root_pressure, radius_exponent, length_exponent):
    data = np.asarray(data, dtype=float)
    idxs = np.asarray(idxs, dtype=int)
    up_len, down_len, term_len = _initial_lengths(point, data, terminal, vessel)

    r_terminal = (8.0 * nu / np.pi) * term_len
    r_terminal_sister = (8.0 * nu / np.pi) * down_len + (data[vessel, 25] - (8.0 * nu / np.pi) * data[vessel, 20])
    ratio = (data[vessel, 22] * r_terminal_sister) / (terminal_flow * r_terminal)
    f_terminal = (1.0 + ratio ** (gamma / 4.0)) ** (-1.0 / gamma)
    f_terminal_sister = (1.0 + ratio ** (-gamma / 4.0)) ** (-1.0 / gamma)
    f_parent = 1.0
    r0 = (8.0 * nu / np.pi) * up_len + 1.0 / ((f_terminal ** 4.0) / r_terminal + (f_terminal_sister ** 4.0) / r_terminal_sister)
    r_parent = r0
    l0 = (f_terminal ** radius_exponent) * (term_len ** length_exponent) + \
         (f_terminal_sister ** radius_exponent) * ((down_len ** length_exponent) + data[vessel, 27])

    bifurcation_ratios = []  # list of [left, right] or [right, left] depending on position
    tmp_bifurcation_ratios = [0.0, 0.0]
    main_idx = []
    alt_idx = []
    main_scale = []
    alt_scale = []
    reduced_resistance = []
    reduced_length = []
    flows = []

    if vessel == 0:
        l0 = (up_len ** length_exponent) + l0
        r0 = (((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)) ** 0.25
        return reduced_resistance, reduced_length, main_idx, main_scale, alt_idx, alt_scale, bifurcation_ratios, flows, float(r0), float(l0)

    previous = int(vessel)
    vessel = int(idxs[vessel, 2])
    if int(idxs[vessel, 0]) == previous:
        alt = int(idxs[vessel, 1]); position = 0; alt_position = 1
    else:
        alt = int(idxs[vessel, 0]); position = 1; alt_position = 0
    alt = -2 if alt == -1 else alt

    if alt == -2:
        f_changed = 1.0
        f_stagnant = 0.0
        f_terminal *= f_changed
        f_terminal_sister *= f_changed
        f_parent = f_changed
        r0 = (8.0 * nu / np.pi) * data[vessel, 20] + r0
        l0 = (f_changed ** radius_exponent) * ((up_len ** length_exponent) + l0)
    else:
        ratio = (data[alt, 22] * data[alt, 25]) / ((data[previous, 22] + terminal_flow) * r0)
        f_changed = (1.0 + ratio ** (gamma / 4.0)) ** (-1.0 / gamma)
        f_stagnant = (1.0 + ratio ** (-gamma / 4.0)) ** (-1.0 / gamma)
        f_terminal *= f_changed
        f_terminal_sister *= f_changed
        f_parent = f_changed
        r0 = (8.0 * nu / np.pi) * data[vessel, 20] + 1.0 / (
            (f_changed ** 4.0) / r0 + (f_stagnant ** 4.0) / data[alt, 25]
        )
        l0 = (f_changed ** radius_exponent) * ((up_len ** length_exponent) + l0) + \
             (f_stagnant ** radius_exponent) * ((data[alt, 20] ** length_exponent) + data[alt, 27])

    tmp_bifurcation_ratios[position] = float(f_changed)
    tmp_bifurcation_ratios[alt_position] = float(f_stagnant)
    if vessel == 0:
        main_idx.append(int(vessel))
        main_scale.append(1.0)
        alt_idx.append(int(alt))
        alt_scale.append(float(f_stagnant))
        reduced_resistance.append(float(r0))
        reduced_length.append(float(l0))
        bifurcation_ratios.append(list(tmp_bifurcation_ratios))
        flows.append(float(data[vessel, 22] + terminal_flow))
        l0 = (data[vessel, 20] ** length_exponent) + l0
        r0 = (((data[vessel, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)) ** 0.25
        return reduced_resistance, reduced_length, main_idx, main_scale, alt_idx, alt_scale, bifurcation_ratios, flows, float(r0), float(l0)
    else:
        main_idx.append(int(vessel))
        alt_idx.append(int(alt))
        main_scale.append(1.0)
        alt_scale.append(float(f_stagnant))
        reduced_resistance.append(float(r0))
        reduced_length.append(float(l0))
        bifurcation_ratios.append(list(tmp_bifurcation_ratios))
        flows.append(float(data[vessel, 22] + terminal_flow))

    previous = int(vessel)
    vessel = int(idxs[vessel, 2])
    while vessel >= 0:
        if int(idxs[vessel, 0]) == previous:
            alt = int(idxs[vessel, 1]); position = 0; alt_position = 1
        else:
            alt = int(idxs[vessel, 0]); position = 1; alt_position = 0
        alt = -2 if alt == -1 else alt
        if alt == -2:
            f_changed = 1.0
            f_stagnant = 0.0
            r0 = (8.0 * nu / np.pi) * data[vessel, 20] + r0
            l0 = (f_changed ** length_exponent) * (data[previous, 20] + l0)
            f_terminal *= f_changed
            f_terminal_sister *= f_changed
            f_parent *= f_changed
        else:
            ratio = (data[alt, 22] * data[alt, 25]) / ((terminal_flow + data[previous, 22]) * r0)
            f_changed = (1.0 + ratio ** (gamma / 4.0)) ** (-1.0 / gamma)
            f_stagnant = (1.0 + ratio ** (-gamma / 4.0)) ** (-1.0 / gamma)
            f_terminal *= f_changed
            f_terminal_sister *= f_changed
            f_parent *= f_changed
            r0 = (8.0 * nu / np.pi) * data[vessel, 20] + 1.0 / (
                (f_changed ** 4.0) / r0 + (f_stagnant ** 4.0) / data[alt, 25]
            )
            l0 = (f_changed ** radius_exponent) * ((data[previous, 20] ** length_exponent) + l0) + \
                 (f_stagnant ** radius_exponent) * ((data[alt, 20] ** length_exponent) + data[alt, 27])
        for j in range(len(main_scale)):
            main_scale[j] = float(main_scale[j] * f_changed)
            alt_scale[j] = float(alt_scale[j] * f_changed)
        main_idx.append(int(vessel))
        alt_idx.append(int(alt))
        main_scale.append(1.0)
        alt_scale.append(float(f_stagnant))
        reduced_resistance.append(float(r0))
        reduced_length.append(float(l0))
        tmp_bifurcation_ratios[position] = float(f_changed)
        tmp_bifurcation_ratios[alt_position] = float(f_stagnant)
        bifurcation_ratios.append(list(tmp_bifurcation_ratios))
        flows.append(float(data[vessel, 22] + terminal_flow))
        previous = int(vessel)
        vessel = int(idxs[vessel, 2])

    l0 = (data[0, 20] ** length_exponent) + l0
    r0 = (((data[0, 22] + terminal_flow) * r0) / (root_pressure - terminal_pressure)) ** 0.25
    return reduced_resistance, reduced_length, main_idx, main_scale, alt_idx, alt_scale, bifurcation_ratios, flows, float(r0), float(l0)


def tree_cost_2(point, data, terminal, idxs, vessel, gamma, nu,
                terminal_flow, terminal_pressure, root_pressure, radius_exponent,
                length_exponent):
    # Same as tree_cost with slightly different guard comments; keep behavior consistent
    return tree_cost(point, data, terminal, idxs, vessel, gamma, nu,
                     terminal_flow, terminal_pressure, root_pressure,
                     radius_exponent, length_exponent)


def update_vessels_2(point, data, terminal, idxs,
                     vessel, gamma, nu, terminal_flow, terminal_pressure,
                     root_pressure, radius_exponent, length_exponent):
    # Mirror update_vessels; pyx version duplicates logic with same returns
    return update_vessels(point, data, terminal, idxs,
                          vessel, gamma, nu, terminal_flow, terminal_pressure,
                          root_pressure, radius_exponent, length_exponent)
