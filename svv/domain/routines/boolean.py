import numpy
import trimesh
import pyvista
import pymeshfix


def convert_to_trimesh(pyvista_object):
    """
    Convert a PyVista triangle surface mesh
    to a Trimesh object.
    :param pyvista_object:
    :return:
    """
    msh = pyvista_object.triangulate()
    faces = msh.faces.reshape((-1, 4))[:, 1:].astype(int)
    vertices = msh.points.astype(float)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def convert_to_pyvista(trimesh_object):
    """
    Convert a Trimesh object to a PyVista
    triangle surface mesh.
    :param trimesh_object:
    :return:
    """
    faces = numpy.hstack([numpy.full((len(trimesh_object.faces), 1), 3),
                          trimesh_object.faces])
    vertices = trimesh_object.vertices
    return pyvista.PolyData(vertices, faces.flatten())


def boolean(pyvista_object_1, pyvista_object_2, operation='union', fix_mesh=True, engine='manifold'):
    """
    Perform a boolean operation between two
    PyVista triangle surface meshes.
    :param pyvista_object_1:
    :param pyvista_object_2:
    :param operation:
    :return:
    """
    trimesh_object_1 = convert_to_trimesh(pyvista_object_1)
    trimesh_object_2 = convert_to_trimesh(pyvista_object_2)
    if operation == 'union':
        result = trimesh_object_1.union(trimesh_object_2, engine=engine)
    elif operation == 'intersection':
        result = trimesh_object_1.intersection(trimesh_object_2, engine=engine)
    elif operation == 'difference':
        result = trimesh_object_1.difference(trimesh_object_2, engine=engine)
    else:
        raise ValueError("Unsupported boolean operation.")
    result = convert_to_pyvista(result)
    if not result.is_manifold and fix_mesh:
        fix = pymeshfix.MeshFix(result)
        fix.repair(verbose=False)
        result = fix.mesh
    return result
