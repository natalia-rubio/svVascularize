import os
import pyvista
import numpy
from xml.dom import minidom


class GeneralMeshBase:
    def __init__(self):
        self.mesh = "mesh"
        self.name = None
        self.faces = {}

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(vars(self))

    def __eq__(self, other):
        check = ["mesh", "faces"]
        attributes = vars(self)
        other_attributes = vars(other)
        return all(other_attributes[key] == attributes[key] for key in check)

    def __ne__(self, other):
        return not self.__eq__(other)

    def add_mesh(self, mesh, name=None):
        if isinstance(mesh, str):
            if os.path.exists(mesh):
                if mesh.endswith(".vtu") or mesh.endswith(".vtp"):
                    self.mesh = pyvista.read(mesh)
                    if not isinstance(name, type(None)):
                        self.name = name
                    else:
                        self.name = "msh"
                else:
                    raise ValueError("Mesh file must be a .vtu or .vtp file.")
            else:
                raise ValueError("Mesh file not found.")
        elif isinstance(mesh, pyvista.UnstructuredGrid):
            if not isinstance(name, type(None)):
                self.name = name
            else:
                self.name = "msh"
            self.mesh = mesh
        elif isinstance(mesh, pyvista.PolyData):
            if not isinstance(name, type(None)):
                self.name = name
            else:
                self.name = "msh"
            self.mesh = mesh
        else:
            raise ValueError("Mesh must be a str type file path or a pyvista mesh object.")

    def add_face(self, face, name=None):
        """
        Add a face to the mesh.
        """
        if isinstance(self.mesh, type(None)):
            raise ValueError("Mesh must be added first.")
        if isinstance(face, str):
            if os.path.exists(face):
                if face.endswith(".vtp"):
                    face = pyvista.read(face)
                    check_nodes = numpy.all(numpy.in1d(face.points.ravel(), self.mesh.points.ravel()))
                    if check_nodes:
                        if not isinstance(name, type(None)):
                            self.faces[name] = face
                        else:
                            self.faces["face_{}".format(len(self.faces))] = face
                    else:
                        raise ValueError("Face nodes do not match mesh nodes.")
                else:
                    raise ValueError("Face file must be a .vtp file.")
            else:
                raise ValueError("Face file not found.")
        elif isinstance(face, pyvista.PolyData):
            check_nodes = numpy.all(numpy.in1d(face.points.ravel(), self.mesh.points.ravel()))
            if check_nodes:
                if not isinstance(name, type(None)):
                    self.faces[name] = face
                else:
                    self.faces["face_{}".format(len(self.faces))] = face
            else:
                raise ValueError("Face nodes do not match mesh nodes.")
        else:
            raise ValueError("Face must be a str type file path or a pyvista mesh object.")

    def rename_face(self, name, new_name):
        if name in self.faces.keys():
            self.faces[new_name] = self.faces.pop(name)
        elif isinstance(name, int):
            if len(self.faces) > name and name >= 0:
                self.faces[new_name] = self.faces.pop("face_{}".format(name))
            else:
                raise ValueError("Face index not found.")
        else:
            raise ValueError("Face name not found.")

    def rename_mesh(self, new_name):
        if isinstance(new_name, str):
            self.mesh = self.mesh.rename(new_name)
        else:
            raise ValueError("New name must be a string.")

    def check_mesh(self):
        if isinstance(self.mesh, type(None)):
            raise ValueError("Mesh must be added first.")
        if isinstance(self.mesh, pyvista.UnstructuredGrid):
            surface = self.mesh.extract_surface()
            surface_set = set([tuple(pt) for pt in surface.points.tolist()])
            for name, face in self.faces.items():
                face_set = set([tuple(pt) for pt in face.points.tolist()])
                surface_set = surface_set - face_set
            if len(surface_set) > 0:
                raise ValueError("Faces do not cover the entire boundary of the mesh.")
        elif isinstance(self.mesh, pyvista.PolyData):
            surface = self.mesh.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                      feature_edges=False, non_manifold_edges=False)
            surface_set = set([tuple(pt) for pt in surface.points.tolist()])
            for name, face in self.faces.items():
                face_set = set([tuple(pt) for pt in face.points.tolist()])
                surface_set = surface_set - face_set
            if len(surface_set) > 0:
                raise ValueError("Faces do not cover the entire boundary of the mesh.")


class GeneralMesh(GeneralMeshBase):
    def __init__(self):
        super().__init__()
        self.file = minidom.Document()
        self.directory = ''
        if self.directory == '':
            self.folder = "mesh"
        else:
            self.folder = self.directory + '/' + "mesh"

    def __str__(self):
        return self.file.toxml().toprettyxml()

    def __repr__(self):
        return self

    def toxml(self):
        add_mesh = self.file.createElement("Add_mesh")
        if not isinstance(self.name, type(None)):
            add_mesh.setAttribute("name", self.name)

        if not isinstance(self.mesh, type(None)):
            mesh_element = self.file.createElement("Mesh_file_path")
            if isinstance(self.mesh, pyvista.UnstructuredGrid):
                path = self.folder + '/' + '{}'.format(self.name) + '/' + "{}.vtu".format(self.name)
            elif isinstance(self.mesh, pyvista.PolyData):
                path = self.folder + '/' + '{}'.format(self.name) + '/' + "{}.vtp".format(self.name)
            else:
                raise ValueError("Mesh must be a pyvista mesh object.")
            mesh_element.appendChild(self.file.createTextNode(path))
            add_mesh.appendChild(mesh_element)

        for name, face in self.faces.items():
            face_element = self.file.createElement("Add_face")
            face_element.setAttribute("name", name)
            face_file_path = self.file.createElement("Face_file_path")
            if isinstance(face, pyvista.PolyData):
                path = self.folder + '/' + '{}'.format(self.name) + '/' + "mesh-surfaces" + '/' + "{}.vtp".format(name)
            else:
                raise ValueError("Face must be a pyvista mesh object.")
            face_file_path.appendChild(self.file.createTextNode(path))
            face_element.appendChild(face_file_path)
            add_mesh.appendChild(face_element)
        return add_mesh

    def change_directory(self, new_directory):
        if isinstance(new_directory, str):
            if not os.path.exists(new_directory):
                raise ValueError("Directory does not exist.")
            self.directory = new_directory
        else:
            raise ValueError("Directory must be a string.")
