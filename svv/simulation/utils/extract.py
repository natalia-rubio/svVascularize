import numpy as np

def build_vertex_map(faces: np.ndarray, n_vertices: int):
    """
    Build a vertex map from face connectivity.

    Parameters
    ----------
    faces : np.ndarray
        Face connectivity array of shape (n_faces, n_vertices_per_face)
    n_vertices : int
        Total number of vertices

    Returns
    -------
    list of list of int
        For each vertex index, a list of face indices that contain that vertex
    """
    vertex_map = [[] for _ in range(n_vertices)]
    for face_idx in range(faces.shape[0]):
        for vertex_idx in faces[face_idx]:
            vertex_map[vertex_idx].append(face_idx)
    return vertex_map
