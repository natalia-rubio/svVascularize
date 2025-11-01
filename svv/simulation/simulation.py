import os
import uuid
from xml.dom import minidom
from copy import deepcopy

import pyvista
import tetgen
import svv
import tqdm
import numpy
import pymeshfix
import vtk

import svv.tree.tree
from svv.simulation.utils.extract_faces import extract_faces
from svv.domain.routines.boolean import boolean
from svv.utils.remeshing.remesh import remesh_surface, remesh_volume
from svv.simulation.general_parameters import GeneralSimulationParameters
from svv.simulation.mesh import GeneralMesh
from svv.simulation.fluid.fluid_equation import FluidEquation
from svv.simulation.utils.boundary_layer import BoundaryLayer


from svv.simulation.fluid.rom import one_d
from svv.simulation.fluid.rom import zero_d
from svv.simulation.fluid.rom.zero_d.zerod_forest import export_0d_simulation
from svv.simulation.fluid.rom.zero_d import project_solution


class Simulation(object):
    def __init__(self, synthetic_object, name=None, directory=None, set_name=None):
        """
        The Simulation class defines a simulation object that
        is used to generate the files to run a physics simulation
        using a synthetic vascular network.
        
        Parameters:
        -----------
        synthetic_object : Tree or Forest
            The synthetic vascular network object
        name : str, optional
            Name of the geometry/simulation (used as geo_name for folder organization)
        directory : str, optional
            Base directory for simulation files
        set_name : str, optional
            Name of the set (used for folder organization). If None, uses "default_set"
        """
        self.synthetic_object = synthetic_object
        if name is None:
            name = "simulations_" + uuid.uuid4().hex
        if directory is None:
            directory = os.getcwd()
        if set_name is None:
            set_name = "default_set"
        
        self.file_path = os.path.join(directory, name)
        self.name = name  # This is now treated as geo_name
        self.set_name = set_name  # Store the set name for folder organization
        self.tissue_domain_faces = []
        self.fluid_domain_faces = []
        self.tissue_domain_surface_meshes = []
        self.fluid_domain_surface_meshes = []
        self.tissue_domain_volume_meshes = []
        self.fluid_domain_volume_meshes = []
        self.tissue_domain_meshes = []
        self.fluid_domain_meshes = []
        self.fluid_domain_boundary_layers = []
        self.fluid_domain_interiors = []
        self.fluid_domain_wall_layers = []
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            self.fluid_3d_simulations = [None]
            self.fluid_1d_simulations = [None]
            self.fluid_0d_simulations = [None]
            self.tissue_simulations = [None]
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest):
            self.fluid_3d_simulations = [[None]*len(network) for network in self.synthetic_object.networks]
            self.fluid_1d_simulations = [[None]*len(network) for network in self.synthetic_object.networks]
            self.fluid_0d_simulations = [[None] * len(network) for network in self.synthetic_object.networks]
            self.tissue_simulations = [None]

    def _create_folder_structure(self):
        """
        Create the new folder structure with set_name and geo_name subdirectories.
        Structure: threeD/set_name/geo_name/ and zeroD/set_name/geo_name/
        """
        # Get the base directory (parent of current file_path)
        base_dir = os.path.dirname(self.file_path)
        
        # Use the stored set_name and name (as geo_name)
        set_name = self.set_name
        geo_name = self.name
        
        # Create threeD and zeroD directories if they don't exist
        threeD_dir = os.path.join(base_dir, "threeD")
        zeroD_dir = os.path.join(base_dir, "zeroD")
        
        if not os.path.exists(threeD_dir):
            os.makedirs(threeD_dir)
            print(f"Created threeD directory: {threeD_dir}")
            
        if not os.path.exists(zeroD_dir):
            os.makedirs(zeroD_dir)
            print(f"Created zeroD directory: {zeroD_dir}")
        
        # Create set-specific directories
        threeD_set_dir = os.path.join(threeD_dir, set_name)
        zeroD_set_dir = os.path.join(zeroD_dir, set_name)
        
        if not os.path.exists(threeD_set_dir):
            os.makedirs(threeD_set_dir)
            print(f"Created threeD set directory: {threeD_set_dir}")
            
        if not os.path.exists(zeroD_set_dir):
            os.makedirs(zeroD_set_dir)
            print(f"Created zeroD set directory: {zeroD_set_dir}")
        
        # Create simulation-specific directories within set directories
        self.threeD_path = os.path.join(threeD_set_dir, geo_name)
        self.zeroD_path = os.path.join(zeroD_set_dir, geo_name)
        
        if not os.path.exists(self.threeD_path):
            os.makedirs(self.threeD_path)
            print(f"Created threeD simulation directory: {self.threeD_path}")
            
        if not os.path.exists(self.zeroD_path):
            os.makedirs(self.zeroD_path)
            print(f"Created zeroD simulation directory: {self.zeroD_path}")


    def build_meshes(self, fluid=True, tissue=False, hausd=0.0001, hsize=None, minratio=1.1, mindihedral=10.0,
                     order=1, remesh_vol=False, smooth_junctions=True, boundary_layer=True, layer_thickness_ratio=0.25,
                     layer_thickness_ratio_adjustment=0.5, boundary_layer_attempts=5, wall_layers=False,
                     wall_thickness=None, upper_num_triangles=1000, lower_num_triangles=100):
        """
        Build the mesh objects for 3D simulations.
        :return:
        [NOTE] Boolean operations and remeshing with of the interface for
        the fluid and tissue domains may need to be redone to ensure mesh
        conformation at the interface.
        """
        self.tissue_domain_surface_meshes = []
        self.fluid_domain_surface_meshes = []
        self.tissue_domain_volume_meshes = []
        self.fluid_domain_volume_meshes = []
        self.fluid_domain_boundary_layers = []
        self.fluid_domain_interiors = []
        self.fluid_domain_wall_layers = []
        basepath = "temp_mesh"
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            if fluid:
                if tissue:
                    extension_scale = 4.0
                    for i in range(5):
                        new_root = self.synthetic_object.data[0, 0:3] - extension_scale * self.synthetic_object.data[0, 21]*self.synthetic_object.data.get('w_basis', 0)
                        if self.synthetic_object.domain(new_root.reshape(1, 3)) > 0:
                            break
                        else:
                            extension_scale += 1.0
                    root_extension = self.synthetic_object.data[0, 21] * extension_scale
                    self.synthetic_object.data[0, 0:3] -= root_extension * self.synthetic_object.data.get('w_basis', 0)
                fluid_surface_mesh = self.synthetic_object.export_solid(watertight=True, smooth_junctions=smooth_junctions, hsize=hsize, cap_resolution=20)
                tet_fluid = tetgen.TetGen(fluid_surface_mesh)

                try:
                    
                    #tet_fluid.tetrahedralize(minratio=minratio, mindihedral=10.0, steinerleft=-1, order=order, nobisect=True, verbose=2, switches='M')
                    nodes, elem = tet_fluid.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    with open(basepath + ".ele", "w") as f:
                        f.write(f"{len(elem)} 4 0\n") # Number of elements, nodes per element, attributes per element
                        for i, element in enumerate(elem):
                            # TetGen elements are 0-indexed, but .ele files often use 1-indexed node IDs
                            # Add 1 to each node ID for 1-based indexing if desired for compatibility
                            f.write(f"{i+1} {element[0]+1} {element[1]+1} {element[2]+1} {element[3]+1}\n")

                    # Save the node data to the .node file (optional, but often needed with .ele)
                    with open(basepath + ".node", "w") as f:
                        f.write(f"{len(nodes)} 3 0 0\n") # Number of nodes, dimensions, attributes, boundary markers
                        for i, node in enumerate(nodes):
                            f.write(f"{i+1} {node[0]} {node[1]} {node[2]}\n")
                    fluid_volume_mesh = tet_fluid.grid
                except:
                    tet_fluid.make_manifold(verbose=True)
                    #tet_fluid.tetrahedralize(minratio=minratio, mindihedral=10.0, steinerleft=-1, order=order, nobisect=True, verbose=2, switches='M')
                    tet_fluid.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    fluid_volume_mesh = tet_fluid.grid
                if isinstance(fluid_volume_mesh, type(None)):
                    print("Failed to generate fluid volume mesh.")


                else:
                    
                    nodes, elem = tet_fluid.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    hsize = fluid_surface_mesh.hsize
                    fluid_surface_mesh = fluid_volume_mesh.extract_surface()
                    fluid_surface_faces = extract_faces(fluid_surface_mesh, fluid_volume_mesh)
                    
                    if boundary_layer:
                        #fluid_surface_mesh = fluid_volume_mesh.extract_surface()
                        #fluid_surface_faces = extract_faces(fluid_surface_mesh, fluid_volume_mesh)
                        wall = fluid_surface_faces[3][0]
                        caps = fluid_surface_faces[1]
                        max_distance = 0.0
                        for i, cap in enumerate(caps):
                            bounds = cap.bounds
                            xmin, xmax, ymin, ymax, zmin, zmax = bounds

                            # Calculate the dimensions along each axis
                            dx = xmax - xmin;dy = ymax - ymin;dz = zmax - zmin

                            # Calculate the maximum distance (diagonal of the bounding box)
                            max_distance = max((max_distance, numpy.sqrt(dx**2 + dy**2 + dz**2)))
                            
                            
                            
                        for i in range(boundary_layer_attempts):
                            try:
                                # wall looks good here
                                fluid_boundary_layers = BoundaryLayer(wall, caps, 
                                                                      layer_thickness=max_distance*0.1,
                                                                      layer_thickness_ratio=1, 
                                                                      number_of_sublayers= 5,
                                                                      sublayer_ratio=0.75,
                                                                      remesh_vol=remesh_vol)
                                fluid_volume_mesh, fluid_interior, fluid_boundary = fluid_boundary_layers.generate()
                                success = True
                                print("Generated boundary layers on attempt {}/{}.".format(i+1, boundary_layer_attempts))

                            except:
                                print("Failed to generate boundary layers {}/{}.\n".format(i+1, boundary_layer_attempts))
                                fluid_boundary = None
                                fluid_interior = None
                                success = False
                                layer_thickness_ratio *= layer_thickness_ratio_adjustment
                            if success:
                                break
                        self.fluid_domain_boundary_layers.append(fluid_boundary)
                        self.fluid_domain_interiors.append(fluid_interior)
                    else:
                        if remesh_vol:
                            fluid_volume_mesh = remesh_volume(fluid_volume_mesh, hsiz=fluid_surface_mesh.hsize)
                        self.fluid_domain_boundary_layers.append(None)
                        self.fluid_domain_interiors.append(None)
                    if wall_layers:
                        if isinstance(wall_thickness, type(None)):
                            wall_thickness = 2*layer_thickness_ratio*hsize
                        wall = fluid_surface_faces[3][0]
                        fluid_boundary_layers = BoundaryLayer(wall, negate_warp_vectors=False,
                                                              layer_thickness=wall_thickness,
                                                              remesh_vol=False, combine=False)
                        _, _, fluid_wall = fluid_boundary_layers.generate()
                        

                        # Perform tetrahedron re-orientation to ensure positive Jacobian
                        fluid_wall = remesh_volume(fluid_wall, nomove=True, noinsert=True, nosurf=True, verbosity=4)
                        if remesh_vol:
                            fluid_wall = remesh_volume(fluid_wall, hausd=hausd, nosurf=True, verbosity=4)
                        self.fluid_domain_wall_layers.append(fluid_wall)
                    fluid_surface_mesh = fluid_volume_mesh.extract_surface()
                    fluid_surface_mesh.hsize = hsize
                    self.fluid_domain_surface_meshes.append(fluid_surface_mesh)
                    self.fluid_domain_volume_meshes.append(fluid_volume_mesh)
                    if tissue:
                        self.synthetic_object.data[0, 0:3] += root_extension * self.synthetic_object.data.get('w_basis',0)
            if tissue and not isinstance(self.synthetic_object.domain, type(None)):
                # Extrude the root of the tree to ensure proper intersection with the tissue domain.
                if not fluid:
                    root_extension = max(self.synthetic_object.data[0, 21] * 4, self.synthetic_object.data[0, 20] * 0.5)
                    self.synthetic_object.data[0, 0:3] -= root_extension * self.synthetic_object.data.get('w_basis', 0)
                    # Should check to see that the extended point does not intersect with another fluid or tissue domain.
                    fluid_surface_boolean_mesh = self.synthetic_object.export_solid(watertight=True, hsize=hsize)
                else:
                    if not wall_layers:
                        fluid_surface_boolean_mesh = deepcopy(self.fluid_domain_surface_meshes[-1])
                    else:
                        fluid_surface_boolean_mesh = deepcopy(self.fluid_domain_wall_layers[-1])
                hsize = fluid_surface_boolean_mesh.hsize
                tissue_domain = remesh_surface(self.synthetic_object.domain.boundary, hausd=hausd) # Check if this should be remeshed
                area = tissue_domain.area
                tissue_domain = boolean(tissue_domain, fluid_surface_boolean_mesh, operation='difference')
                if fluid:
                    fluid_faces = extract_faces(tissue_domain, None)
                    face_sizes = [len(face) for face in fluid_faces[0]]
                    wall = numpy.argmax(face_sizes)
                    low_tri_area = area / upper_num_triangles
                    hmin = ((4.0*low_tri_area)/3.0**0.5) ** (0.5)
                    upper_tri_area = area / lower_num_triangles
                    hmax = ((4.0*upper_tri_area)/3.0**0.5) ** (0.5)
                    tissue_domain = remesh_surface(tissue_domain, hausd=hausd)
                else:
                    tissue_domain = remesh_surface(tissue_domain, hausd=hausd)
                tet_tissue = tetgen.TetGen(tissue_domain)
                if not fluid:
                    self.synthetic_object.data[0, 0:3] += root_extension * self.synthetic_object.data.get('w_basis', 0)
                try:
                    tet_tissue.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    #tet_tissue.tetrahedralize(minratio=minratio, order=order)
                    tissue_volume_mesh = tet_tissue.grid
                except:
                    if fluid:
                        print('Mesh interface may be corrupted after mesh fixing for tetrahedralization.')
                    tet_tissue.make_manifold(verbose=True)
                    #tet_tissue.tetrahedralize(minratio=minratio, order=order)
                    tet_tissue.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    tissue_volume_mesh = tet_tissue.grid
                if isinstance(tissue_volume_mesh, type(None)):
                    print("Failed to generate tissue volume mesh.")
                else:
                    if remesh_volume:
                        tissue_volume_mesh = remesh_volume(tissue_volume_mesh, hausd=hausd, nosurf=True)
                    tissue_domain = tissue_volume_mesh.extract_surface()
                    self.tissue_domain_surface_meshes.append(tissue_domain)
                    self.tissue_domain_volume_meshes.append(tissue_volume_mesh)
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest) and isinstance(self.synthetic_object.connections, type(None)):
            for network in self.synthetic_object.networks:
                network_fluid_surface_meshes = []
                network_fluid_volume_meshes = []
                network_tissue_surface_meshes = []
                network_tissue_volume_meshes = []
                for tree in network:
                    if fluid:
                        fluid_surface_mesh = tree.export_solid(watertight=True, hsize=hsize)
                        tet_fluid = tetgen.TetGen(fluid_surface_mesh)
                        try:
                            tet_fluid.make_manifold(verbose=False)
                            tet_tissue.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                            fluid_volume_mesh = tet_fluid.grid
                        except:
                            try:
                                tet_fluid.make_manifold(verbose=True)
                                tet_tissue.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                                fluid_volume_mesh = tet_fluid.grid
                            except:
                                fluid_volume_mesh = None
                        if isinstance(fluid_volume_mesh, type(None)):
                            print("Failed to generate fluid volume mesh.")
                            network_fluid_surface_meshes.append(None)
                            network_fluid_volume_meshes.append(None)
                        else:
                            fluid_volume_mesh = remesh_volume(fluid_volume_mesh, hausd=hausd)
                            fluid_surface_mesh = fluid_volume_mesh.extract_surface()
                            network_fluid_surface_meshes.append(fluid_surface_mesh)
                            network_fluid_volume_meshes.append(fluid_volume_mesh)
                    if tissue:
                        # Extrude the root of the tree to ensure proper intersection with the tissue domain.
                        root_extension = max(tree.data[0, 21] * 4, tree.data[0, 20] * 0.5)
                        tree.data[0, 0:3] -= root_extension * tree.data.get('w_basis', 0)
                        # Should check to see that the extended point does not intersect with another fluid or tissue domain.
                        fluid_surface_boolean_mesh = tree.export_solid(watertight=True, hsize=hsize)
                        if len(self.tissue_domain_surface_meshes) > 0:
                            tissue_domain = self.tissue_domain_surface_meshes[-1]
                        else:
                            tissue_domain = tree.domain.boundary
                        tissue_domain = boolean(tissue_domain, fluid_surface_boolean_mesh, operation='difference')
                        tissue_domain = remesh_surface(tissue_domain, hausd=hausd)
                        tet_tissue = tetgen.TetGen(tissue_domain)
                        tree.data[0, 0:3] += root_extension * tree.data.get('w_basis', 0)
                        try:
                            tet_tissue.make_manifold(verbose=False)
                            tet_tissue.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                            tissue_volume_mesh = tet_tissue.grid
                        except:
                            try:
                                tet_tissue.make_manifold(verbose=True)
                                tet_tissue.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                                tissue_volume_mesh = tet_tissue.grid
                            except:
                                tissue_volume_mesh = None
                        if isinstance(tissue_volume_mesh, type(None)):
                            print("Failed to generate tissue volume mesh.")
                            network_tissue_surface_meshes.append(None)
                            network_tissue_volume_meshes.append(None)
                        else:
                            tissue_volume_mesh = remesh_volume(tissue_volume_mesh, hausd=hausd)
                            tissue_domain = tissue_volume_mesh.extract_surface()
                            network_tissue_surface_meshes.append(tissue_domain)
                            network_tissue_volume_meshes.append(tissue_volume_mesh)
                self.fluid_domain_surface_meshes.append(network_fluid_surface_meshes)
                self.fluid_domain_volume_meshes.append(network_fluid_volume_meshes)
                self.tissue_domain_surface_meshes.append(network_tissue_surface_meshes)
                self.tissue_domain_volume_meshes.append(network_tissue_volume_meshes)
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest) and not isinstance(self.synthetic_object.connections, type(None)):
            if fluid or tissue:
                if tissue:
                    network_solids, _, _ = self.synthetic_object.connections.export_solid(extrude_roots=True)
                else:
                    network_solids, _, _ = self.synthetic_object.connections.export_solid(extrude_roots=False)
                for i, fluid_surface in enumerate(network_solids):
                    if fluid:
                        tet_fluid = tetgen.TetGen(fluid_surface)
                        try:
                            tet_fluid.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                            fluid_volume = tet_fluid.grid
                        except:
                            tet_fluid.make_manifold(verbose=True)
                            tet_fluid.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                            fluid_volume = tet_fluid.grid
                        if isinstance(fluid_volume, type(None)):
                            print("Failed to generate fluid volume mesh.")
                            self.fluid_domain_surface_meshes.append(fluid_surface)
                            self.fluid_domain_volume_meshes.append(None)
                        else:
                            hsize = fluid_surface.hsize
                            if (boundary_layer or wall_layers) and fluid:
                                fluid_surface = fluid_volume.extract_surface()
                                fluid_surface_faces = extract_faces(fluid_surface, fluid_volume)
                            if boundary_layer and fluid:
                                if len(fluid_surface_faces[1]) > 1:
                                    print("Boundary layer generation with more than one wall mesh is ambiguous.")
                                    print("Only the first wall mesh will be used.")
                                elif len(fluid_surface_faces[1]) == 0:
                                    print("No wall mesh found for boundary layer generation.")
                                wall = fluid_surface_faces[1][0]
                                for j in range(boundary_layer_attempts):
                                    try:
                                        fluid_boundary_layers = BoundaryLayer(wall,
                                                                              layer_thickness=layer_thickness_ratio * hsize,
                                                                              remesh_vol=remesh_vol)
                                        fluid_volume, fluid_interior, fluid_boundary = fluid_boundary_layers.generate()
                                        fluid_surface = fluid_volume.extract_surface()
                                        success = True
                                        
                                        print("Generated boundary layers on attempt {}/{}.".format(i + 1,
                                                                                                   boundary_layer_attempts))
                                    except:
                                        print("Failed to generate boundary layers {}/{}.\n".format(i + 1,
                                                                                                   boundary_layer_attempts))
                                        fluid_boundary = None
                                        fluid_interior = None
                                        success = False
                                        layer_thickness_ratio *= layer_thickness_ratio_adjustment
                                        
                                    if success:
                                        break
                                
                                self.fluid_domain_boundary_layers.append(fluid_boundary)
                                self.fluid_domain_interiors.append(fluid_interior)
                            else:
                                if remesh_vol:
                                    fluid_volume = remesh_volume(fluid_volume, hsiz=hsize)
                                self.fluid_domain_boundary_layers.append(None)
                                self.fluid_domain_interiors.append(None)
                            if wall_layers and fluid:
                                if isinstance(wall_thickness, type(None)):
                                    wall_thickness = 2 * layer_thickness_ratio * hsize
                                wall = fluid_surface_faces[1][0]
                                fluid_boundary_layers = BoundaryLayer(wall, negate_warp_vectors=False,
                                                                      layer_thickness=wall_thickness,
                                                                      remesh_vol=False, combine=False)
                                _, _, fluid_wall = fluid_boundary_layers.generate()
                                # Perform tetrahedron re-orientation to ensure positive Jacobian
                                fluid_wall = remesh_volume(fluid_wall, nomove=True, noinsert=True, nosurf=True, verbosity=4)
                                if remesh_vol:
                                    fluid_wall = remesh_volume(fluid_wall, hausd=hausd, nosurf=True, verbosity=4)
                                self.fluid_domain_wall_layers.append(fluid_wall)
                            else:
                                self.fluid_domain_wall_layers.append(None)
                            fluid_surface.hsize = hsize
                        self.fluid_domain_surface_meshes.append(fluid_surface)
                        self.fluid_domain_volume_meshes.append(fluid_volume)
                    else:
                        self.fluid_domain_surface_meshes.append(fluid_surface)
                        self.fluid_domain_volume_meshes.append(None)
            if tissue:
                tissue_domain = deepcopy(self.synthetic_object.domain.boundary)
                tissue_domain = tissue_domain.compute_normals(auto_orient_normals=True)
                fluid_hsize = min([mesh.hsize for mesh in self.fluid_domain_surface_meshes])
                radii = []
                for net in range(len(self.synthetic_object.networks)):
                    for tr in range(len(self.synthetic_object.networks[net])):
                        radii.append(self.synthetic_object.networks[net][tr].data[0, 21])
                hsize = min(radii) * 2.0
                print("Remeshing tissue domain with edge size {}.".format(hsize))
                tissue_domain = remesh_surface(tissue_domain, hsiz=hsize)
                for i, fluid_surface in enumerate(self.fluid_domain_surface_meshes):
                    fluid_surface_normals = fluid_surface.compute_normals(auto_orient_normals=True)
                    print("Performing boolean operation with fluid surface mesh {}.".format(i))
                    tissue_domain = boolean(tissue_domain, fluid_surface_normals, operation='difference', engine='blender')
                    tissue_domain = tissue_domain.compute_normals(auto_orient_normals=True)
                    print("Remeshing tissue domain with edge size {}.".format(fluid_hsize))
                    #tissue_domain = remesh_surface(tissue_domain, hmin=fluid_hsize, hmax=hsize)
                    tissue_domain = remesh_surface(tissue_domain, optim=True)
                self.tissue_domain_surface_meshes.append(tissue_domain)
                #tissue_domain = remesh_surface(tissue_domain, hausd=hausd)
                print("Tetrahedralizing tissue domain.")
                tet_tissue = tetgen.TetGen(tissue_domain)
                try:
                    tet_tissue.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    tissue_volume_mesh = tet_tissue.grid
                except:
                    tet_tissue.make_manifold(verbose=True)
                    tet_tissue.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    tissue_volume_mesh = tet_tissue.grid
                if isinstance(tissue_volume_mesh, type(None)):
                    print("Failed to generate tissue volume mesh.")
                else:
                    if remesh_vol:
                        tissue_volume_mesh = remesh_volume(tissue_volume_mesh, hausd=hausd, nosurf=True)
                    tissue_surface = tissue_volume_mesh.extract_surface()
                    self.tissue_domain_surface_meshes[-1] = tissue_surface
                    self.tissue_domain_volume_meshes.append(tissue_volume_mesh)
        else:
            raise ValueError("Unsupported synthetic object type.")

    def extract_faces(self, crease_angle=60.0, verbose=False):
        """
        Extract the faces from the mesh objects.
        :return:
        """
        self.tissue_domain_faces = []
        self.fluid_domain_faces = []
        self.fluid_domain_meshes = []
        self.tissue_domain_meshes = []
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):

            if len(self.fluid_domain_surface_meshes) > 0 and len(self.fluid_domain_volume_meshes) > 0:
                # something is switched here... wall_surfaces are caps, lumen_surfaces are walls
                assert (len(self.fluid_domain_surface_meshes) == 1)
                assert (len(self.fluid_domain_volume_meshes) == 1)
                
                faces, caps, should_be_empty, walls, shared_boundaries = extract_faces(self.fluid_domain_surface_meshes[0],
                                                                      self.fluid_domain_volume_meshes[0],
                                                                      crease_angle=crease_angle, verbose=verbose)
                self.fluid_domain_faces.append({'walls': walls, 'caps': caps, 'shared_boundaries': shared_boundaries})
                fluid_mesh = GeneralMesh()
                fluid_mesh.add_mesh(self.fluid_domain_volume_meshes[0], name='fluid_msh_0')
                for i, wall in enumerate(walls):
                    fluid_mesh.add_face(wall, name='wall_{}'.format(i))
                for i, cap in enumerate(caps):
                    fluid_mesh.add_face(cap, name='cap_{}'.format(i))
                # for i, lumen in enumerate(lumen_surfaces):
                #     fluid_mesh.add_face(lumen, name='lumen_{}'.format(i))
                fluid_mesh.check_mesh()
                self.fluid_domain_meshes.append(fluid_mesh)
            if len(self.tissue_domain_surface_meshes) > 0 and len(self.tissue_domain_volume_meshes) > 0:
                faces, caps, should_be_empty, walls, shared_boundaries = extract_faces(self.tissue_domain_surface_meshes[0],
                                                                      self.tissue_domain_volume_meshes[0],
                                                                      crease_angle=crease_angle, verbose=verbose)
                self.tissue_domain_faces.append({'walls': walls, 'caps': caps, 'shared_boundaries': shared_boundaries})
                tissue_mesh = GeneralMesh()
                tissue_mesh.add_mesh(self.tissue_domain_volume_meshes[0], name='tissue_msh_0')
                for i, wall in enumerate(walls):
                    tissue_mesh.add_face(wall, name='lumen_{}'.format(i))
                for i, cap in enumerate(caps):
                    tissue_mesh.add_face(cap, name='wall_{}'.format(i))
                # for i, lumen in enumerate(lumen_surfaces):
                #     tissue_mesh.add_face(lumen, name='lumen_{}'.format(i))
                tissue_mesh.check_mesh()
                self.tissue_domain_meshes.append(tissue_mesh)
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest):
            network_tissue_faces = []
            network_tissue_domains = []
            network_fluid_faces = []
            network_fluid_domains = []
            for i in range(len(self.fluid_domain_surface_meshes)):
                for j in range(len(self.fluid_domain_surface_meshes[i])):
                    surface = self.fluid_domain_surface_meshes[i][j]
                    mesh = self.fluid_domain_volume_meshes[i][j]
                    if isinstance(surface, type(None)) or isinstance(mesh, type(None)):
                        network_fluid_faces.append(None)
                        network_fluid_domains.append(None)
                        continue
                    nodes, walls, caps, shared_boundaries = extract_faces(surface, mesh, crease_angle=crease_angle,
                                                                          verbose=verbose)
                    network_fluid_faces.append({'walls': walls, 'caps': caps, 'shared_boundaries': shared_boundaries})
                    fluid_mesh = GeneralMesh()
                    fluid_mesh.add_mesh(mesh, name='fluid_msh_{}'.format(len(self.fluid_domain_meshes)))
                    for k, wall in enumerate(walls):
                        fluid_mesh.add_face(wall, name='wall_{}'.format(k))
                    for k, cap in enumerate(caps):
                        fluid_mesh.add_face(cap, name='cap_{}'.format(k))
                    fluid_mesh.check_mesh()
                    network_fluid_domains.append(fluid_mesh)
                self.fluid_domain_faces.append(network_fluid_faces)
                self.fluid_domain_meshes.append(network_fluid_domains)
            for i in range(len(self.tissue_domain_surface_meshes)):
                for j in range(len(self.tissue_domain_surface_meshes[i])):
                    surface = self.tissue_domain_surface_meshes[i][j]
                    mesh = self.tissue_domain_volume_meshes[i][j]
                    nodes, walls, caps, lumens, shared_boundaries = extract_faces(surface, mesh, crease_angle=crease_angle,
                                                                          verbose=False)
                    network_tissue_faces.append({'walls': walls, 'caps': caps, 'shared_boundaries': shared_boundaries})
                    tissue_mesh = GeneralMesh()
                    tissue_mesh.add_mesh(mesh, name='tissue_msh_{}'.format(len(self.tissue_domain_meshes)))
                    for k, wall in enumerate(walls):
                        tissue_mesh.add_face(wall, name='lumen_{}'.format(k))
                    for k, cap in enumerate(caps):
                        tissue_mesh.add_face(cap, name='wall_{}'.format(k))
                    tissue_mesh.check_mesh()
                    network_tissue_domains.append(tissue_mesh)
                self.tissue_domain_faces.append(network_tissue_faces)
                self.tissue_domain_meshes.append(network_tissue_domains)

    def construct_3d_fluid_equation(self, *args, target_reynolds_number=None):
        """
        Construct the equations for the simulation.
        
        Parameters:
        -----------
        *args : tuple
            Network and tree IDs for Forest objects
        target_reynolds_number : float, optional
            If provided, calculates inlet flow to achieve this Reynolds number
            using the inlet diameter. If None, uses the flow from the tree data.
        
        :return:
        """
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            fluid_mesh = self.fluid_domain_meshes[0]
            fluid_equation = FluidEquation()
            fluid_equation.add_mesh(fluid_mesh)
            inlet = None
            best = numpy.inf
            inlet_center = self.synthetic_object.data[0, 0:3]
            for name, face in fluid_mesh.faces.items():
                if 'cap' in name or 'lumen' in name:
                    if numpy.linalg.norm(face.center - inlet_center) < best:
                        best = numpy.linalg.norm(face.center)
                        inlet = name
            if isinstance(inlet, type(None)):
                raise ValueError("Inlet not found.")
            
            # Get inlet radius and calculate flow based on Reynolds number if provided
            inlet_radius = self.synthetic_object.data[0, 21]
            
            if target_reynolds_number is not None:
                # Calculate flow from Reynolds number: Re = ρ*V*D/μ = 2*ρ*Q/(π*r*μ)
                # Solving for Q: Q = Re * π * r * μ / (2 * ρ)
                # Since μ = ν * ρ (dynamic viscosity = kinematic viscosity * density):
                # Q = Re * π * r * ν / 2
                kinematic_viscosity = self.synthetic_object.parameters.kinematic_viscosity
                characteristic_flow = -1.0 * target_reynolds_number * numpy.pi * inlet_radius * kinematic_viscosity / 2.0
                print(f"Calculated inlet flow for Re={target_reynolds_number}:")
                print(f"  Inlet diameter: {2*inlet_radius:.6f}")
                print(f"  Inlet radius: {inlet_radius:.6f}")
                print(f"  Kinematic viscosity: {kinematic_viscosity:.6f}")
                print(f"  Target Reynolds number: {target_reynolds_number:.1f}")
                print(f"  Calculated flow: {-characteristic_flow:.6f}")
            else:
                # Use flow from tree data (negative for inlet)
                characteristic_flow = -1*self.synthetic_object.data[0, 22]
            
            # Create folder structure if not already created
            if not hasattr(self, 'threeD_path'):
                self._create_folder_structure()
                
            # Write flow file for unsteady inlet boundary condition
            flow_filename = f"{inlet}.flow"
            flow_filepath = os.path.join(self.threeD_path, flow_filename)
            self.write_flow(
                file_path=flow_filepath,
                max_reynolds_number=target_reynolds_number if target_reynolds_number is not None else 1000,
                profile_type='physiological',
                num_time_steps=self.number_of_time_steps,
                num_fourier_modes=10,
                num_repeats=2
            )
            
            # Add inlet with time-varying flow file
            fluid_equation.add_inlet(
                inlet, 
                value=None,  # Not used for unsteady BC
                time_varying_file=flow_filename,
                profile='Parabolic',
                impose_flux=True
            )
            fluid_equation.set_viscosity('Constant', self.synthetic_object.parameters.kinematic_viscosity)
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest):
            if len(args) == 0:
                network_id = 0
                tree_id = 0
            elif len(args) == 1:
                network_id = args[0]
                tree_id = 0
            elif len(args) == 2:
                network_id = args[0]
                tree_id = args[1]
            else:
                raise ValueError("Too many arguments.")
            fluid_mesh = self.fluid_domain_meshes[network_id][tree_id]
            fluid_equation = FluidEquation()
            fluid_equation.add_mesh(fluid_mesh)
            # Verify cap inlet
            inlet = None
            for name, face in fluid_mesh.faces.items():
                best = numpy.inf
                inlet_center = self.synthetic_object.networks[network_id][tree_id].data[0, 0:3]
                if 'cap' in name or 'lumen' in name:
                    if numpy.linalg.norm(face.center - inlet_center) < best:
                        best = numpy.linalg.norm(face.center)
                        inlet = name
            if isinstance(inlet, type(None)):
                raise ValueError("Inlet not found.")
            
            # Get inlet radius and calculate flow based on Reynolds number if provided
            inlet_radius = self.synthetic_object.networks[network_id][tree_id].data[0, 21]
            
            if target_reynolds_number is not None:
                # Calculate flow from Reynolds number: Re = ρ*V*D/μ = 2*ρ*Q/(π*r*μ)
                # Solving for Q: Q = Re * π * r * μ / (2 * ρ)
                # Since μ = ν * ρ (dynamic viscosity = kinematic viscosity * density):
                # Q = Re * π * r * ν / 2
                kinematic_viscosity = self.synthetic_object.networks[network_id][tree_id].parameters.kinematic_viscosity
                characteristic_flow = -1.0 * target_reynolds_number * numpy.pi * inlet_radius * kinematic_viscosity / 2.0
                print(f"Calculated inlet flow for Re={target_reynolds_number} (network {network_id}, tree {tree_id}):")
                print(f"  Inlet diameter: {2*inlet_radius:.6f}")
                print(f"  Inlet radius: {inlet_radius:.6f}")
                print(f"  Kinematic viscosity: {kinematic_viscosity:.6f}")
                print(f"  Target Reynolds number: {target_reynolds_number:.1f}")
                print(f"  Calculated flow: {-characteristic_flow:.6f}")
            else:
                # Use flow from tree data (negative for inlet)
                characteristic_flow = -1*self.synthetic_object.networks[network_id][tree_id].data[0, 22]
            
            # Write flow file for unsteady inlet boundary condition
            flow_filename = f"{inlet}.flow"
            flow_filepath = os.path.join(self.file_path, flow_filename)
            self.write_flow(
                file_path=flow_filepath,
                max_reynolds_number=target_reynolds_number if target_reynolds_number is not None else 1000,
                profile_type='physiological',
                num_time_steps=self.number_of_time_steps,
                num_fourier_modes=10
            )
            
            # Add inlet with time-varying flow file
            fluid_equation.add_inlet(
                inlet,
                value=None,  # Not used for unsteady BC
                time_varying_file=flow_filename,
                profile='Parabolic',
                impose_flux=True
            )
            fluid_equation.set_viscosity('Constant', self.synthetic_object.networks[network_id][tree_id].parameters.kinematic_viscosity)
        else:
            raise ValueError("Unsupported synthetic object type.")
        for face in fluid_mesh.faces:
            if face == inlet:
                continue
            if 'cap' in face:
                fluid_equation.add_outlet(face, value=0.0)
            if 'wall' in face:
                fluid_equation.add_wall(face)
        fluid_equation.check_bcs()
        return fluid_equation

    def construct_3d_fluid_simulation(self, *args, number_of_time_steps=100, time_step_size=0.001, 
                                       increment_in_saving_vtk_files=10, target_reynolds_number=None):
        """
        Construct the 3D simulations.
        
        Parameters:
        -----------
        *args : tuple
            Network and tree IDs for Forest objects
        number_of_time_steps : int
            Number of time steps to run (default: 100)
        time_step_size : float
            Size of each time step (default: 0.001)
        increment_in_saving_vtk_files : int
            How often to save VTK output files (default: 10)
        target_reynolds_number : float, optional
            If provided, calculates inlet flow to achieve this Reynolds number
            using the inlet diameter. If None, uses the flow from the tree data.
        
        :return:
        """
        self.number_of_time_steps = number_of_time_steps  
        if len(args) == 0:
            network_id = 0
            tree_id = 0
        elif len(args) == 1:
            network_id = args[0]
            tree_id = 0
        elif len(args) == 2:
            network_id = args[0]
            tree_id = args[1]
        else:
            raise ValueError("Too many arguments.")
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            simulation_file = minidom.Document()
            svfsi_file = simulation_file.createElement("svMultiPhysicsFile")
            svfsi_file.setAttribute("version", "0.1")
            general_simulation_parameters = GeneralSimulationParameters()
            general_simulation_parameters.number_of_time_steps = self.number_of_time_steps
            general_simulation_parameters.time_step_size = time_step_size
            general_simulation_parameters.increment_in_saving_vtk_files = increment_in_saving_vtk_files
            fluid_mesh = self.fluid_domain_meshes[0]
            fluid_equation = self.construct_3d_fluid_equation(target_reynolds_number=target_reynolds_number)
            svfsi_file.appendChild(general_simulation_parameters.toxml())
            svfsi_file.appendChild(fluid_mesh.toxml())
            svfsi_file.appendChild(fluid_equation.toxml())
            simulation_file.appendChild(svfsi_file)
            self.fluid_3d_simulations[0] = tuple([simulation_file, fluid_mesh])
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest):
            simulation_file = minidom.Document()
            svfsi_file = simulation_file.createElement("svMultiPhysicsFile")
            svfsi_file.setAttribute("version", "0.1")
            general_simulation_parameters = GeneralSimulationParameters()
            general_simulation_parameters.number_of_time_steps = self.number_of_time_steps
            general_simulation_parameters.time_step_size = time_step_size
            general_simulation_parameters.increment_in_saving_vtk_files = increment_in_saving_vtk_files
            fluid_mesh = self.fluid_domain_meshes[network_id][tree_id]
            fluid_equation = self.construct_3d_fluid_equation(network_id, tree_id, target_reynolds_number=target_reynolds_number)
            svfsi_file.appendChild(general_simulation_parameters.toxml())
            svfsi_file.appendChild(fluid_mesh.toxml())
            svfsi_file.appendChild(fluid_equation.toxml())
            simulation_file.appendChild(svfsi_file)
            self.fluid_3d_simulations[network_id][tree_id] = tuple([simulation_file, fluid_mesh])
        else:
            raise ValueError("Index out of range.")
        return

    def write_3d_fluid_simulation(self, *args, write_centerlines=False):
        """
        Write the fluid simulation to disk.
        
        Parameters:
        -----------
        *args : tuple
            Network and tree IDs for Forest objects
        write_centerlines : bool, optional
            Whether to write centerlines to VTP file (default: False)
        
        Returns:
        --------
        None
        """
        if len(args) == 0:
            network_id = 0
            tree_id = 0
        elif len(args) == 1:
            network_id = args[0]
            tree_id = 0
        elif len(args) == 2:
            network_id = args[0]
            tree_id = args[1]
        else:
            raise ValueError("Too many arguments.")
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            simulation_file, fluid_mesh = self.fluid_3d_simulations[0]
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest):
            simulation_file, fluid_mesh = self.fluid_3d_simulations[network_id][tree_id]
        else:
            raise ValueError("Index out of range.")
        if not isinstance(simulation_file, type(None)) and not isinstance(fluid_mesh, type(None)):
            # Create the new folder structure
            self._create_folder_structure()
            
            # Create mesh subdirectories in threeD folder
            mesh_dir = os.path.join(self.threeD_path, "mesh")
            if not os.path.exists(mesh_dir):
                os.makedirs(mesh_dir)
                
            mesh_name_dir = os.path.join(mesh_dir, fluid_mesh.name)
            if not os.path.exists(mesh_name_dir):
                os.makedirs(mesh_name_dir)
                
            mesh_surfaces_dir = os.path.join(mesh_name_dir, "mesh-surfaces")
            if not os.path.exists(mesh_surfaces_dir):
                os.makedirs(mesh_surfaces_dir)
            if isinstance(fluid_mesh.mesh, pyvista.UnstructuredGrid):
                
                fluid_mesh.mesh.point_data["GlobalNodeID"] = numpy.arange(1, 1 + fluid_mesh.mesh.n_points, dtype=numpy.int32)
                fluid_mesh.mesh.cell_data['GlobalElementID'] = numpy.arange(1, 1 + fluid_mesh.mesh.n_cells, dtype=numpy.int32)
                fluid_mesh.mesh.save(os.path.join(mesh_name_dir, "{}.vtu".format(fluid_mesh.name)))
                
            elif isinstance(fluid_mesh.mesh, pyvista.PolyData):
                fluid_mesh.mesh.save(os.path.join(mesh_name_dir, "{}.vtp".format(fluid_mesh.name)))
            else:
                raise ValueError("Mesh must be a pyvista mesh object.")
            for name, face in fluid_mesh.faces.items():
                print("Writing face: {}".format(name))
                if isinstance(face, pyvista.PolyData):
                    # Ensure cell data arrays are int32 type and validate they contain positive values
                    for array_name, array in face.cell_data.items():
                        print("checking array: {}".format(array_name))
                        # Convert to int32 but preserve the actual values (don't overwrite!)
                        face.cell_data[array_name] = face.cell_data[array_name].astype(numpy.int32)
                        if not numpy.all(face.cell_data[array_name] > 0):
                            print("WARNING: Array {} contains non-positive values".format(array_name))

                        #assert(numpy.all(face.cell_data[array_name] > 0))
                    face.save(os.path.join(mesh_surfaces_dir, "{}.vtp".format(name)))
                else:
                    raise ValueError("Face must be a pyvista mesh object.")

            # Save centerlines as VTP file (optional)
            if write_centerlines:
                try:
                    centerlines, _ = self.synthetic_object.export_centerlines()
                    centerlines_file = os.path.join(self.threeD_path, "centerlines_svVascularize.vtp")
                    centerlines.save(centerlines_file)
                    print(f"Saved centerlines: {centerlines_file}")
                except Exception as e:
                    print(f"Warning: Could not save centerlines: {e}")
            
            # Save XML file to threeD directory
            xml_file_name = os.path.join(self.threeD_path, "fluid_simulation_{}-{}.xml".format(network_id, tree_id))
            with open(xml_file_name, 'w') as f:
                f.write(simulation_file.toprettyxml())
            print(f"Saved 3D simulation XML to: {xml_file_name}")
        else:
            raise ValueError("Simulation file or mesh object not found.")

    def construct_1d_fluid_simulation(self, *args, viscosity=None, density=None, time_step_size=0.01,
                                      number_time_steps=100, olufsen_material_exponent=2):
        if len(args) == 0:
            network_id = 0
            tree_id = 0
        elif len(args) == 1:
            network_id = args[0]
            tree_id = 0
        elif len(args) == 2:
            network_id = args[0]
            tree_id = args[1]
        else:
            raise ValueError("Too many positional input arguments")
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            centerlines, _ = self.synthetic_object.export_centerlines()
            material = one_d.parameters.MaterialModel()
            params = one_d.parameters.Parameters()
            params.output_directory = self.file_path + os.sep + "fluid" + os.sep + "1d"
            params.solver_output_file = self.file_path + os.sep + "fluid" + os.sep + "1d" + os.sep + "1d_simulation_input.json"
            params.centerlines_input_file = self.file_path + os.sep + "fluid" + os.sep + "1d" + os.sep + "centerlines.vtp"
            params.outlet_face_names_file = self.file_path + os.sep + "fluid" + os.sep + "1d" + "outlets"
            params.seg_size_adaptive = True
            params.model_order = 1  # Since this is strictly a 1d ROM simulation it is not an exposed parameter
            params.uniform_bc = False
            params.inflow_input_file = self.file_path + os.sep + "fluid" + os.sep + "1d" + "inflow_1d.flow"
            params.outflow_bc_type = ["rcrt.dat"]
            params.outflow_bc_file = self.file_path + os.sep + "fluid" + os.sep + "1d"
            params.model_name = "1d_model_{}-{}".format(network_id, tree_id)
            params.compute_mesh = True
            params.time_step = time_step_size
            params.num_time_steps = number_time_steps
            params.olufsen_material_exponent = olufsen_material_exponent
            params.material_model = material.OLUFSEN
            params.viscosity = viscosity
            params.density = density
            mesh = one_d.mesh.Mesh()
            self.fluid_1d_simulations[0] = tuple([centerlines, mesh, params])
        else:
            raise ValueError("Index out of range.")

    def write_1d_fluid_simulation(self, *args):
        pass

    def construct_0d_fluid_equation(self, *args):
        pass

    def construct_0d_fluid_simulation(self, *args):
        pass

    def write_0d_fluid_simulation(self, *args, **kwargs):
        """
        Write 0D simulation files to the zeroD directory.
        """
        # Create folder structure if not already created
        if not hasattr(self, 'zeroD_path'):
            self._create_folder_structure()
            
        # Import the 0D export functions
        from svv.simulation.fluid.rom.zero_d.zerod_tree import export_0d_simulation as export_0d_tree
        from svv.simulation.fluid.rom.zero_d.zerod_forest import export_0d_simulation as export_0d_forest
        
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            # For single tree, use the tree export function
            export_0d_tree(
                self.synthetic_object, 
                outdir=self.zeroD_path, 
                folder="0d_tmp",
                **kwargs
            )
            print(f"Saved 0D simulation files to: {self.zeroD_path}/0d_tmp")
            
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest):
            # For forest, use the forest export function
            # You may need to specify network_id and inlets
            network_id = kwargs.get('network_id', 0)
            inlets = kwargs.get('inlets', [0])
            
            export_0d_forest(
                self.synthetic_object,
                network_id,
                inlets,
                outdir=self.zeroD_path,
                folder="0d_tmp",
                **kwargs
            )
            print(f"Saved 0D simulation files to: {self.zeroD_path}/0d_tmp")
        else:
            raise ValueError("Unsupported synthetic object type for 0D simulation.")

    def construct_3d_tissue_perfusion_equation(self, *args):
        pass

    def construct_3d_tissue_perfusion_simulation(self, *args):
        pass

    def write_3d_tissue_perfusion_simulation(self, *args):
        pass

    def pulsatile_waveform(self, *args):
        """
        Generate a pulsatile waveform for fluid simulations. This function should accept either an average flow rate
        or an array of flow rate values over the nominal cardiac cycle t -> [0, 1].

        This function should check that the waveform is properly formatted and that the values are within a reasonable
        range; otherwise it should warn the user that values of the waveform might result in unrealistic/erroneous
        results. (e.g. high reynolds number flows, reversed flow, etc.)
        :param args:
        :return:
        """
        pass

    def generate_inflow(self, *args, filename=None):
        pass

    def write_flow(self, file_path, max_reynolds_number, profile_type='ramp', num_time_steps=100, num_fourier_modes=None, num_repeats=1, inlet_radius=None, kinematic_viscosity=None):
        """
        Write a .flow file for unsteady boundary conditions.
        
        Parameters:
        -----------
        file_path : str
            Path to the .flow file to write
        max_reynolds_number : float
            Maximum Reynolds number for the inlet flow
        profile_type : str
            Type of temporal profile. Options:
            - 'ramp': Ramp up from 0 to max flow over first half, then constant
            - 'constant': Constant value equal to max flow
            - 'sinusoidal': Sinusoidal waveform with peak at max flow
            - 'physiological': Physiological cardiac waveform scaled to max flow
        num_time_steps : int
            Number of time steps in the cardiac cycle
        num_fourier_modes : int, optional
            Number of Fourier modes for FFT (default: num_time_steps // 2)
        num_repeats : int, optional
            Number of repeats for the first half of physiological profile (default: 1)
            Only applies to 'physiological' profile_type
        inlet_radius : float, optional
            Inlet radius for Reynolds number calculation. If None, uses tree data.
        kinematic_viscosity : float, optional
            Kinematic viscosity for Reynolds number calculation. If None, uses tree data.
        
        Returns:
        --------
        None
        """
        import numpy
        from scipy import interpolate
        
        if num_fourier_modes is None:
            num_fourier_modes = num_time_steps // 2
        
        # Calculate characteristic flow rate from Reynolds number
        if inlet_radius is None:
            # Get inlet radius from tree data
            if hasattr(self.synthetic_object, 'data'):
                # Tree object
                inlet_radius = self.synthetic_object.data[0, 21]
            else:
                # Forest object - use first network/tree
                inlet_radius = self.synthetic_object.networks[0][0].data[0, 21]
        
        if kinematic_viscosity is None:
            # Get kinematic viscosity from tree data
            if hasattr(self.synthetic_object, 'parameters'):
                # Tree object
                kinematic_viscosity = self.synthetic_object.parameters.kinematic_viscosity
            else:
                # Forest object - use first network/tree
                kinematic_viscosity = self.synthetic_object.networks[0][0].parameters.kinematic_viscosity
        
        # Calculate characteristic flow rate from Reynolds number
        # Q = Re * π * r * ν / 2
        characteristic_value = -1.0 * max_reynolds_number * numpy.pi * inlet_radius * kinematic_viscosity / 2.0
        
        print(f"Calculated inlet flow for Re={max_reynolds_number}:")
        print(f"  Inlet diameter: {2*inlet_radius:.6f} cm")
        print(f"  Inlet radius: {inlet_radius:.6f} cm")
        print(f"  Kinematic viscosity: {kinematic_viscosity:.6f} cm²/s")
        print(f"  Target Reynolds number: {max_reynolds_number:.1f}")
        print(f"  Calculated flow: {-characteristic_value:.6f} cm³/s")
        
        # Generate time array (normalized to [0, 1] for one cardiac cycle)
        # For physiological profile, limit to first half of cycle (0 to 0.5) and repeat if needed
        if profile_type == 'physiological':
            # Create time array for one repeat of the first half
            time_single = numpy.linspace(0.0, 0.5, num_time_steps + 1)
            # Repeat the time array for the specified number of repeats
            time = numpy.tile(time_single, num_repeats)
            # Adjust time values for each repeat
            for i in range(num_repeats):
                start_idx = i * (num_time_steps + 1)
                end_idx = (i + 1) * (num_time_steps + 1)
                time[start_idx:end_idx] += i * 0.5
        else:
            time = numpy.linspace(0.0, 1.0, num_time_steps + 1)
        
        # Generate flow values based on profile type
        if profile_type == 'ramp':
            # Ramp up from 0 to characteristic_value over first half
            flow = numpy.zeros_like(time)
            mid_point = len(time) // 2
            flow[:mid_point] = numpy.linspace(0, characteristic_value, mid_point)
            flow[mid_point:] = characteristic_value
            
        elif profile_type == 'constant':
            # Constant flow
            flow = numpy.full_like(time, characteristic_value)
            
        elif profile_type == 'sinusoidal':
            # Sinusoidal waveform with peak at characteristic_value
            # Using sin wave that starts at 0, peaks at 0.25*period
            flow = characteristic_value * numpy.sin(2 * numpy.pi * time)
            # Shift to make it non-negative if characteristic_value > 0
            if characteristic_value > 0:
                flow = numpy.maximum(flow, 0.0)
        
        elif profile_type == 'physiological':
            # Physiological cardiac waveform from reference data
            # Reference waveform data (time, flow)
            ref_time = numpy.array([0.000000000000000000e+00, 3.799999999999999992e-03, 7.499999999999999722e-03, 
                1.129999999999999928e-02, 1.510000000000000057e-02, 1.880000000000000074e-02, 2.259999999999999856e-02, 
                2.630000000000000046e-02, 3.009999999999999828e-02, 3.389999999999999958e-02, 3.760000000000000148e-02, 
                4.139999999999999930e-02, 4.519999999999999712e-02, 4.889999999999999902e-02, 5.269999999999999685e-02, 
                5.639999999999999875e-02, 6.019999999999999657e-02, 6.400000000000000133e-02, 6.769999999999999629e-02, 
                7.149999999999999412e-02, 7.530000000000000582e-02, 7.900000000000000078e-02, 8.279999999999999860e-02, 
                8.659999999999999643e-02, 9.030000000000000526e-02, 9.410000000000000309e-02, 9.779999999999999805e-02, 
                1.015999999999999959e-01, 1.053999999999999937e-01, 1.091000000000000025e-01, 1.129000000000000004e-01, 
                1.166999999999999982e-01, 1.203999999999999931e-01, 1.242000000000000048e-01, 1.279000000000000137e-01, 
                1.317000000000000115e-01, 1.355000000000000093e-01, 1.391999999999999904e-01, 1.429999999999999882e-01, 
                1.468000000000000138e-01, 1.504999999999999949e-01, 1.542999999999999927e-01, 1.580000000000000016e-01, 
                1.617999999999999994e-01, 1.655999999999999972e-01, 1.693000000000000060e-01, 1.731000000000000039e-01, 
                1.769000000000000017e-01, 1.806000000000000105e-01, 1.844000000000000083e-01, 1.882000000000000062e-01, 
                1.918999999999999873e-01, 1.957000000000000128e-01, 1.993999999999999939e-01, 2.031999999999999917e-01, 
                2.069999999999999896e-01, 2.106999999999999984e-01, 2.144999999999999962e-01, 2.182999999999999940e-01, 
                2.220000000000000029e-01, 2.258000000000000007e-01, 2.295000000000000095e-01, 2.333000000000000074e-01, 
                2.371000000000000052e-01, 2.407999999999999863e-01, 2.446000000000000119e-01, 2.484000000000000097e-01, 
                2.520999999999999908e-01, 2.559000000000000163e-01, 2.596999999999999864e-01, 2.634000000000000230e-01, 
                2.671999999999999931e-01, 2.708999999999999742e-01, 2.746999999999999997e-01, 2.785000000000000253e-01, 
                2.822000000000000064e-01, 2.859999999999999765e-01, 2.898000000000000020e-01, 2.934999999999999831e-01, 
                2.973000000000000087e-01, 3.009999999999999898e-01, 3.048000000000000154e-01, 3.085999999999999854e-01, 
                3.123000000000000220e-01, 3.160999999999999921e-01, 3.199000000000000177e-01, 3.235999999999999988e-01, 
                3.274000000000000243e-01, 3.311000000000000054e-01, 3.348999999999999755e-01, 3.387000000000000011e-01, 
                3.423999999999999821e-01, 3.462000000000000077e-01, 3.499999999999999778e-01, 3.537000000000000144e-01, 
                3.574999999999999845e-01, 3.613000000000000100e-01, 3.649999999999999911e-01, 3.688000000000000167e-01, 
                3.724999999999999978e-01, 3.763000000000000234e-01, 3.800999999999999934e-01, 3.837999999999999745e-01, 
                3.876000000000000001e-01, 3.914000000000000257e-01, 3.951000000000000068e-01, 3.988999999999999768e-01, 
                4.026000000000000134e-01, 4.063999999999999835e-01, 4.102000000000000091e-01, 4.138999999999999901e-01, 
                4.177000000000000157e-01, 4.214999999999999858e-01, 4.252000000000000224e-01, 4.289999999999999925e-01, 
                4.328000000000000180e-01, 4.364999999999999991e-01, 4.403000000000000247e-01, 4.440000000000000058e-01, 
                4.477999999999999758e-01, 4.516000000000000014e-01, 4.552999999999999825e-01, 4.591000000000000081e-01, 
                4.628999999999999782e-01, 4.666000000000000147e-01, 4.703999999999999848e-01, 4.741000000000000214e-01, 
                4.778999999999999915e-01, 4.817000000000000171e-01, 4.853999999999999981e-01, 4.892000000000000237e-01, 
                4.929999999999999938e-01, 4.966999999999999749e-01, 5.004999999999999449e-01, 5.041999999999999815e-01, 
                5.080000000000000071e-01, 5.118000000000000327e-01, 5.154999999999999583e-01, 5.192999999999999838e-01, 
                5.231000000000000094e-01, 5.268000000000000460e-01, 5.305999999999999606e-01, 5.343999999999999861e-01, 
                5.381000000000000227e-01, 5.419000000000000483e-01, 5.455999999999999739e-01, 5.493999999999999995e-01, 
                5.532000000000000250e-01, 5.568999999999999506e-01, 5.606999999999999762e-01, 5.645000000000000018e-01, 
                5.682000000000000384e-01, 5.719999999999999529e-01, 5.756999999999999895e-01, 5.795000000000000151e-01, 
                5.833000000000000407e-01, 5.869999999999999662e-01, 5.907999999999999918e-01, 5.946000000000000174e-01, 
                5.983000000000000540e-01, 6.020999999999999686e-01, 6.058999999999999941e-01, 6.096000000000000307e-01, 
                6.133999999999999453e-01, 6.170999999999999819e-01, 6.209000000000000075e-01, 6.247000000000000330e-01, 
                6.283999999999999586e-01, 6.321999999999999842e-01, 6.360000000000000098e-01, 6.397000000000000464e-01, 
                6.434999999999999609e-01, 6.471999999999999975e-01, 6.510000000000000231e-01, 6.548000000000000487e-01, 
                6.584999999999999742e-01, 6.622999999999999998e-01, 6.661000000000000254e-01, 6.697999999999999510e-01, 
                6.735999999999999766e-01, 6.773000000000000131e-01, 6.811000000000000387e-01, 6.848999999999999533e-01, 
                6.885999999999999899e-01, 6.924000000000000155e-01, 6.962000000000000410e-01, 6.998999999999999666e-01, 
                7.036999999999999922e-01, 7.075000000000000178e-01, 7.112000000000000544e-01, 7.149999999999999689e-01, 
                7.187000000000000055e-01, 7.225000000000000311e-01, 7.262999999999999456e-01, 7.299999999999999822e-01, 
                7.338000000000000078e-01, 7.376000000000000334e-01, 7.412999999999999590e-01, 7.450999999999999845e-01, 
                7.488000000000000211e-01, 7.526000000000000467e-01, 7.563999999999999613e-01, 7.600999999999999979e-01, 
                7.639000000000000234e-01, 7.677000000000000490e-01, 7.713999999999999746e-01, 7.752000000000000002e-01, 
                7.790000000000000258e-01, 7.826999999999999513e-01, 7.864999999999999769e-01, 7.902000000000000135e-01, 
                7.940000000000000391e-01, 7.977999999999999536e-01, 8.014999999999999902e-01, 8.053000000000000158e-01, 
                8.091000000000000414e-01, 8.127999999999999670e-01, 8.165999999999999925e-01, 8.203000000000000291e-01, 
                8.241000000000000547e-01, 8.278999999999999693e-01, 8.316000000000000059e-01, 8.354000000000000314e-01, 
                8.391999999999999460e-01, 8.428999999999999826e-01, 8.467000000000000082e-01, 8.504000000000000448e-01, 
                8.541999999999999593e-01, 8.579999999999999849e-01, 8.617000000000000215e-01, 8.655000000000000471e-01, 
                8.692999999999999616e-01, 8.729999999999999982e-01, 8.768000000000000238e-01, 8.806000000000000494e-01, 
                8.842999999999999750e-01, 8.881000000000000005e-01, 8.918000000000000371e-01, 8.955999999999999517e-01, 
                8.993999999999999773e-01, 9.031000000000000139e-01, 9.069000000000000394e-01, 9.106999999999999540e-01, 
                9.143999999999999906e-01, 9.182000000000000162e-01, 9.219000000000000528e-01, 9.256999999999999673e-01, 
                9.294999999999999929e-01, 9.332000000000000295e-01, 9.370000000000000551e-01])
            
            ref_flow = numpy.array([-1.379357119734847004e+01, -2.319314179284629773e+01, -3.431056154859179941e+01, 
                -4.715446630665641692e+01, -6.169774258390032884e+01, -7.787783100315029117e+01, -9.559783927812581794e+01, 
                -1.147273533063476521e+02, -1.351068943074041613e+02, -1.565500561382174851e+02, -1.788490266648585987e+02, 
                -2.017797333116055825e+02, -2.251061318112393792e+02, -2.485868753620300140e+02, -2.719808299300773342e+02, 
                -2.950515014985616631e+02, -3.175738056628919139e+02, -3.393366475501418904e+02, -3.601485974043342821e+02, 
                -3.798400895443295440e+02, -3.982661692904551387e+02, -4.153076005749235264e+02, -4.308725528821120747e+02, 
                -4.448953398159042081e+02, -4.573362102063240968e+02, -4.681803845857408533e+02, -4.774345787859867301e+02, 
                -4.851263965257277277e+02, -4.912997072339014721e+02, -4.960132128061205208e+02, -4.993368702558612426e+02, 
                -5.013487622604354783e+02, -5.021326870605877275e+02, -5.017753060692037366e+02, -5.003650292758651403e+02, 
                -4.979882485213101404e+02, -4.947296964674292781e+02, -4.906702867857731007e+02, -4.858867146168708473e+02, 
                -4.804507428559647906e+02, -4.744290813316129061e+02, -4.678834251788047709e+02, -4.608704134255666531e+02, 
                -4.534412595023198378e+02, -4.456433050463742802e+02, -4.375187028206174205e+02, -4.291046792367867511e+02, 
                -4.204346038556851113e+02, -4.115366964053906713e+02, -4.024339125077595440e+02, -3.931447637215191548e+02, 
                -3.836817917084674150e+02, -3.740517250806576044e+02, -3.642559402778479694e+02, -3.542892485039337771e+02, 
                -3.441413745744511630e+02, -3.337960300646334986e+02, -3.232323144998787825e+02, -3.124256982170069818e+02, 
                -3.013484082667265511e+02, -2.899717222402797461e+02, -2.782671117114882691e+02, -2.662082370653419616e+02, 
                -2.537730552786051703e+02, -2.409458156233008026e+02, -2.277186323642351056e+02, -2.140940132490720487e+02, 
                -2.000860737172318977e+02, -1.857218757400020763e+02, -1.710426434065859098e+02, -1.561038960639083371e+02, 
                -1.409750347879601406e+02, -1.257401612957777672e+02, -1.104949764964283361e+02, -9.534647113674643037e+01, 
                -8.041030645379116493e+01, -6.580766942089428539e+01, -5.166321313662206194e+01, -3.810103645208668866e+01, 
                -2.524125108239019255e+01, -1.319742367798500027e+01, -2.072117602566660111e+00, 8.045043273947593221e+00, 
                1.708073937350365412e+01, 2.498006262779241737e+01, 3.170769366379050425e+01, 3.724930675901719468e+01, 
                4.161193554332903943e+01, 4.482311991361553538e+01, 4.693046315146652603e+01, 4.799949070238392324e+01, 
                4.811174455922664350e+01, 4.736190582207393618e+01, 4.585488022116327045e+01, 4.370239778355168170e+01, 
                4.101963714725864918e+01, 3.792197487392882493e+01, 3.452167210134788178e+01, 3.092489820597587169e+01, 
                2.722936212795273647e+01, 2.352194965734193488e+01, 1.987748574144684000e+01, 1.635745826507214673e+01, 
                1.300969013579367406e+01, 9.868602444353925307e+00, 6.955723398346298936e+00, 4.280832368832765411e+00, 
                1.843645385355177080e+00, -3.646317924702477953e-01, -2.359085062251911058e+00, -4.159177273720876755e+00, 
                -5.786692188009296522e+00, -7.263738451250087991e+00, -8.611166054281758520e+00, -9.847150746764889107e+00, 
                -1.098607820239864807e+01, -1.203797115740670520e+01, -1.300816650755663595e+01, -1.389738809390468433e+01, 
                -1.470230204907924509e+01, -1.541620217023033135e+01, -1.603001962921158352e+01, -1.653353868455684506e+01, 
                -1.691657432159151142e+01, -1.717026240307729168e+01, -1.728822392718000955e+01, -1.726751570905097566e+01, 
                -1.710941496074873669e+01, -1.681995300277056415e+01, -1.641006307726832247e+01, -1.589543378255583583e+01, 
                -1.529614249980285301e+01, -1.463580994077887354e+01, -1.394067468637341101e+01, -1.323841845742598089e+01, 
                -1.255681069285197715e+01, -1.192240025775462442e+01, -1.135910789981361191e+01, -1.088707535517418457e+01, 
                -1.052156080540194694e+01, -1.027212663167964912e+01, -1.014210973891120027e+01, -1.012843301806211826e+01, 
                -1.022165982618676594e+01, -1.040654911897675028e+01, -1.066274182429476447e+01, -1.096585298403124042e+01, 
                -1.128870861976882267e+01, -1.160278768440995734e+01, -1.187970877850710849e+01, -1.209273290620379449e+01, 
                -1.221816152408105260e+01, -1.223665591651839968e+01, -1.213420226960090709e+01, -1.190294213482905405e+01, 
                -1.154154990557681870e+01, -1.105533940713902652e+01, -1.045602250621437435e+01, -9.761094644366833606e+00, 
                -8.992908431068235231e+00, -8.177508275874316723e+00, -7.343293435188847695e+00, -6.519496054158594234e+00, 
                -5.734630520774326001e+00, -5.015051387198162125e+00, -4.383494623897464137e+00, -3.857963183121243755e+00, 
                -3.450742459203953949e+00, -3.167788550890556998e+00, -3.008465218315385936e+00, -2.965588729858885664e+00, 
                -3.025870667333082409e+00, -3.170668117477439818e+00, -3.377035372012296310e+00, -3.618995503959061999e+00, 
                -3.868979237305238517e+00, -4.099376063486214861e+00, -4.284021179554657444e+00, -4.399677742028394789e+00, 
                -4.427293898876108358e+00, -4.353061860532345229e+00, -4.169181658830558135e+00, -3.874265207696101498e+00, 
                -3.473417023487221034e+00, -2.977917552714464122e+00, -2.404596757724654310e+00, -1.774853994707286109e+00, 
                -1.113428233522042010e+00, -4.470153356420084600e-01, 1.972947015320831166e-01, 7.935781725234993811e-01, 
                1.318537804176509010e+00, 1.752917969203144244e+00, 2.082575122123758060e+00, 2.299358781913793504e+00, 
                2.401592403334920078e+00, 2.394187642457217713e+00, 2.288391027535819688e+00, 2.101126405663306329e+00, 
                1.854014113295764821e+00, 1.572067336214832300e+00, 1.282199086562630708e+00, 1.011550453580732034e+00, 
                7.857859111110966355e-01, 6.274500027553118198e-01, 5.544445726169864308e-01, 5.787696171886005381e-01, 
                7.055729603877869405e-01, 9.325964461209681478e-01, 1.250048811208041677e+00, 1.640929241462077748e+00, 
                2.081834519534012884e+00, 2.544138789372190068e+00, 2.995612487145912439e+00, 3.402298089782714197e+00, 
                3.730595421475901841e+00, 3.949452310985989367e+00, 4.032536638040204124e+00, 3.960233425253367745e+00, 
                3.721388813544581176e+00, 3.314672927904266153e+00, 2.749444350986970598e+00, 2.046050517418056991e+00, 
                1.235553195863076548e+00, 3.587590916545446706e-01, -5.353673203580494588e-01, -1.391812534064463369e+00, 
                -2.152289873975785639e+00, -2.758277031514521305e+00, -3.154485045977560631e+00, -3.292421517152531063e+00, 
                -3.133945051615595556e+00, -2.654693836774178806e+00, -1.847137555815473231e+00, -7.231915056121085428e-01, 
                6.838554961476898120e-01, 2.318161666486830530e+00, 4.100889199405957974e+00, 5.930879412534071804e+00, 
                7.685976973959263603e+00, 9.225516436921326502e+00, 1.039353178688869406e+01, 1.102267992970502242e+01, 
                1.093895904578267775e+01, 9.966761983699232275e+00, 7.934339543080232815e+00, 4.679242717501269411e+00, 
                5.389867159821015719e-02, -6.069324561590091704e+00, -1.379357119734853399e+01])
            
            # Normalize reference time to [0, 1]
            ref_time_norm = ref_time / ref_time.max()
            
            # For physiological profile, only use first half of the cardiac cycle (0 to 0.5)
            # Find the midpoint of the reference data
            mid_point = len(ref_time_norm) // 2
            ref_time_half = ref_time_norm[:mid_point]
            ref_flow_half = ref_flow[:mid_point]
            
            # Find the peak (maximum absolute value) in the first half
            ref_peak = numpy.max(numpy.abs(ref_flow_half))
            
            # Scale the reference flow to match characteristic_value
            # Flip sign so negative values represent inflow (into the inlet)
            ref_flow_scaled = -ref_flow_half * (characteristic_value / ref_peak)
            
            # Interpolate to desired number of time steps for one repeat
            time_single = numpy.linspace(0.0, 0.5, num_time_steps + 1)
            interp_func = interpolate.interp1d(ref_time_half, ref_flow_scaled, kind='cubic', 
                                               fill_value='extrapolate')
            flow_single = interp_func(time_single)
            
            # Repeat the flow values for the specified number of repeats
            flow = numpy.tile(flow_single, num_repeats)
            
            # Add smoothing between repeats if num_repeats > 1
            if num_repeats > 1:
                # Create smoothing window for transitions
                transition_points = num_time_steps // 20  # 5% of time steps for transition
                if transition_points < 2:
                    transition_points = 2
                
                # Smooth transitions between repeats
                for i in range(1, num_repeats):
                    # Find transition region
                    start_idx = i * (num_time_steps + 1) - transition_points
                    end_idx = i * (num_time_steps + 1) + transition_points
                    
                    # Ensure indices are within bounds
                    start_idx = max(0, start_idx)
                    end_idx = min(len(flow), end_idx)
                    
                    if end_idx > start_idx:
                        # Create smooth transition using cubic interpolation with derivative matching
                        transition_time = numpy.linspace(0, 1, end_idx - start_idx)
                        
                        # Get values and derivatives at endpoints
                        flow_start = flow[start_idx]
                        flow_end = flow[end_idx - 1]
                        
                        # Calculate derivatives at endpoints (using finite differences)
                        derivative_scaler = time[end_idx] - time[start_idx]
                        
                        if start_idx > 0:
                            deriv_start = derivative_scaler * (flow[start_idx] - flow[start_idx - 1]) / (time[start_idx] - time[start_idx - 1])
                        else:
                            deriv_start = 0.0
                            
                        if end_idx < len(flow):
                            deriv_end = derivative_scaler * (flow[end_idx] - flow[end_idx - 1]) / (time[end_idx] - time[end_idx - 1])
                        else:
                            deriv_end = 0.0
                        
                        
                        # Cubic Hermite interpolation: matches values and derivatives at endpoints
                        # P(t) = P0(2t³-3t²+1) + P1(-2t³+3t²) + m0(t³-2t²+t) + m1(t³-t²)
                        # where P0, P1 are values and m0, m1 are derivatives
                        t = transition_time
                        smooth_transition = (flow_start * (2*t**3 - 3*t**2 + 1) + 
                                           flow_end * (-2*t**3 + 3*t**2) + 
                                           deriv_start * (t**3 - 2*t**2 + t) + 
                                           deriv_end * (t**3 - t**2))
                        
                        # Apply the smooth transition directly
                        flow[start_idx:end_idx] = smooth_transition
            
        else:
            raise ValueError(f"Unknown profile_type: {profile_type}. Choose 'ramp', 'constant', 'sinusoidal', or 'physiological'.")
        
        # Write to file
        with open(file_path, 'w') as f:
            # Calculate actual number of time steps (accounting for repeats)
            actual_time_steps = len(time) - 1
            # Write header: num_time_steps num_fourier_modes
            f.write(f"{actual_time_steps + 1}    {num_fourier_modes}\n")
            
            # Write time-value pairs
            for t, q in zip(time, flow):
                f.write(f"{t:.6f}    {q:.6f}\n")
        
        print(f"Flow file written to: {file_path}")
        print(f"  Profile type: {profile_type}")
        print(f"  Max Reynolds number: {max_reynolds_number}")
        print(f"  Characteristic flow: {characteristic_value:.6f}")
        print(f"  Number of time steps: {num_time_steps}")
        print(f"  Number of Fourier modes: {num_fourier_modes}")
        if profile_type == 'physiological':
            print(f"  Number of repeats: {num_repeats}")
            print(f"  Total time steps: {actual_time_steps + 1}")
            print(f"  Time range: 0.0 to {time.max():.1f}")

    def write(self):
        """
        Save the simulation file to disk using the new folder structure.
        :return:
        """
        # Create the new folder structure
        self._create_folder_structure()
        print(f"Created folder structure:")
        print(f"  threeD: {self.threeD_path}")
        print(f"  zeroD: {self.zeroD_path}")

    def write_input_file(self):
        """
        Write the simulation input file.
        :return:
        """
        pass
