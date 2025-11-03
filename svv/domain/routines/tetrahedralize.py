import tetgen
import pymeshfix
from svv.utils.remeshing import remesh


def triangulate(curve, verbose=False, **kwargs):
    """
    Triangulate a curve using VTK.

    Parameters
    ----------
    curve : Pyvista.PolyData PolyLine object
        The boundary curve within which the triangulation will
        be performed.
    verbose : bool
        A flag to indicate if mesh fixing should be verbose.
    kwargs : dict
        A dictionary of keyword arguments to be passed to VTK.

    Returns
    -------
    mesh : PyMesh mesh object
        A triangular mesh representing the triangulated region bounded by
        the curve.
    nodes : ndarray
        An array of node coordinates for the triangular mesh.
    vertices : ndarray
        An array of vertex indices for the triangular mesh.
    """
    mesh = curve.delaunay_2d(**kwargs)
    mesh = remesh.remesh_surface(mesh)
    nodes = mesh.points
    vertices = mesh.faces.reshape(-1, 4)[:, 1:]
    return mesh, nodes, vertices


def tetrahedralize(surface_mesh, verbose=False, **kwargs):
    """
    Tetrahedralize a surface mesh using TetGen.

    Parameters
    ----------
    surface_mesh : PyMesh mesh object
        The surface mesh to tetrahedralize.
    verbose : bool
        A flag to indicate if mesh fixing should be verbose.
    kwargs : dict
        A dictionary of keyword arguments to be passed to TetGen.

    Returns
    -------
    mesh : PyMesh mesh object
        An unstructured grid mesh representing the tetrahedralized
        volume enclosed by the surface mesh manifold.
    """
    mesh = pymeshfix.MeshFix(surface_mesh)
    mesh.repair(verbose=verbose)
    tet = tetgen.TetGen(mesh.mesh)
    nodes, vertices = tet.tetrahedralize(**kwargs)
    mesh = tet.grid
    return mesh, nodes, vertices
