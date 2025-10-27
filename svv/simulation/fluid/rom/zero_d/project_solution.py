import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from svv.simulation.utils.close_segments import close_exact_points

def closest_point_on_segment(A, B, P):
    # Convert the points to numpy arrays
    A, B, P = np.array(A), np.array(B), np.array(P)

    # Vector from A to B
    AB = B - A

    # Vector from A to P
    AP = P - A

    # Project AP onto AB to find the parametric point on AB
    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:  # A and B are the same point
        t = 0
    else:
        t = np.dot(AP, AB) / AB_squared

    # Clamp t to the range [0, 1]
    t = max(0, min(1, t))

    # Find the closest point on the segment
    Q = A + t * AB
    return Q, t


def closest_point(points, segments):
    min_distance = float('inf')
    closest_segment_index = -1
    closest_t = -1
    closest_point_on_seg = None

    for idx, (A, B) in enumerate(segments):
        for P in points:
            Q, t = closest_point_on_segment(A, B, P)
            distance = np.linalg.norm(Q - P)
            if distance < min_distance:
                min_distance = distance
                closest_segment_index = idx
                closest_t = t
                closest_point_on_seg = Q

    return closest_segment_index, closest_t



def closest_point_for_each(points, segments):
    results = []
    for P in points:
        closest_segment_index, closest_t = closest_point([P], segments)
        results.append((closest_segment_index, closest_t))
    return results


def visualize_closest_points(points, segments, closest_segments):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    points_np = np.array(points)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c='red', label='Points')

    # Plot the line segments
    for A, B in segments:
        A_np = np.array(A)
        B_np = np.array(B)
        ax.plot([A_np[0], B_np[0]], [A_np[1], B_np[1]], [A_np[2], B_np[2]], 'k-', label='Line Segments')

    # Plot the closest points and connections
    for idx, (P, (segment_index, t)) in enumerate(zip(points, closest_segments)):
        A, B = segments[segment_index]
        Q, _ = closest_point_on_segment(A, B, P)
        Q_np = np.array(Q)
        P_np = np.array(P)
        ax.scatter(Q_np[0], Q_np[1], Q_np[2], c='blue', label=f'Closest Point for P{idx}')
        ax.plot([P_np[0], Q_np[0]], [P_np[1], Q_np[1]], [P_np[2], Q_np[2]], 'g--', label=f'Connection for P{idx}')

    # Set labels and show plot
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.legend(loc='upper left')
    plt.show()


def project_solution(surface, results, time_index, output_file=None):
    segments = results[:, 0:6, time_index]
    points = surface.points
    indices, t = close_exact_points(segments, points)
    surface.point_data["Pressure"] = results[indices, 9, time_index]*(1 - t) + results[indices, 10, time_index]*t
    surface.point_data["Flow"] = results[indices, 11, time_index]*(1 - t) + results[indices, 12, time_index]*t
    if not isinstance(output_file, type(None)):
        with open(output_file, 'w+') as f:
            for i in range(surface.n_points):
                f.write('{} {}\n'.format(surface.point_data["GlobalNodeID"][i], surface.point_data["Pressure"][i]))
        f.close()
    return surface

# Example usage:
#points = [(1, 2, 3), (4, 5, 6), (0.5,0.25,0.5)]
#segments = [(0, 0, 0, 1, 1, 1), (1, 1, 1, 2, 2, 2), (0,0,1,1,0,1)]
#index, t = closest_point(points, segments)

#points = np.array(points)
#segments = np.array(segments)
#closest_segments = close_exact_points(points, segments)

#closest_segments = closest_point_for_each(points, segments)
#visualize_closest_points(points, segments, closest_segments)