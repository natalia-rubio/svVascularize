import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_points(points, values=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if not isinstance(values, type(None)):
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, cmap='viridis')
        cbar = fig.colorbar(scatter, ax=ax)
    else:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
