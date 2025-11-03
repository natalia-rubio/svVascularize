import numpy

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def geodesic_constructor(domain, **kwargs):
    """
    Construct a general geodesic function solver for a given domain.

    Parameters
    ----------
    domain : svtoolkit.domain.Domain
        The domain object that defines the spatial region in which vascular
        trees are generated.
    kwargs : dict
        Additional keyword arguments.
        Keyword arguments include:


    Returns
    -------
    get_geodesic : function
        A function that computes the geodesic path between two points.
    """
    idx = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 1], [2, 3]]
    tetra = domain.mesh.cells.reshape(-1, 5)[:, 1:]
    lengths = []
    nodes = []
    added_nodes = set([])
    tetra_node_tree = cKDTree(domain.mesh.points)
    for i in range(tetra.shape[0]):
        tet = tetra[i, :]
        for cx in idx:
            if tuple([tet[cx[0]], tet[cx[1]]]) in added_nodes:
                continue
            added_nodes.add(tuple([tet[cx[0]], tet[cx[1]]]))
            added_nodes.add(tuple([tet[cx[1]], tet[cx[0]]]))
            nodes.append([tet[cx[0]], tet[cx[1]]])
            nodes.append([tet[cx[1]], tet[cx[0]]])
            length = numpy.linalg.norm(domain.mesh.points[tet[cx[0]], :] - domain.mesh.points[tet[cx[1]], :])
            lengths.append(length)
            lengths.append(length)
    M = numpy.array(nodes)
    L = numpy.array(lengths)
    graph = csr_matrix((L, (M[:, 0], M[:, 1])), shape=(numpy.max(M[:, 0]) + 1, numpy.max(M[:, 1]) + 1))

    def get_path(start, end, graph=graph, shortest_path=shortest_path):
        dist, pred = shortest_path(csgraph=graph, directed=False, indices=start, return_predecessors=True)
        path = [end]
        dists = []
        k = end
        while pred[k] != -9999:
            path.append(pred[k])
            dists.append(dist[k])
            k = pred[k]
        path = path[::-1]
        dists = dists[::-1]
        lines = []
        for i in range(len(path) - 1):
            lines.append([path[i], path[i + 1]])
        return path, dists, lines

    def get_geodesic(start, end, tetra_node_tree=tetra_node_tree, get_path=get_path):
        """
        Get the geodesic path between two points

        Parameters
        ----------
        start : numpy.ndarray
            The starting point.
        end : numpy.ndarray
            The ending point.
        tetra_node_tree : scipy.spatial.cKDTree
            The tree of the tetrahedral nodes. This parameter is partially applied and
            should not be modified.
        get_path : function
            The function that computes the path between two nodes. This parameter is
            partially applied and should not be modified.

        Returns
        -------
        path : list
            The list of nodes (mesh indices) in the path.
        dists : list
            The list of distances between nodes.
        lines : list
            The list of lines between nodes.
        """
        ind = tetra_node_tree.query(start)[1]
        jnd = tetra_node_tree.query(end)[1]
        path, dists, lines = get_path(ind, jnd)
        return path, dists, lines
    return get_geodesic

