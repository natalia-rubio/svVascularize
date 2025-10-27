from xml.dom import minidom

from svv.simulation.output import Output
from svv.simulation.mesh import GeneralMesh
from svv.simulation.boundary_conditions import BoundaryCondition
from svv.simulation.linear_solver import LinearSolver


class EquationBase:
    def __init__(self):
        self.equation_type = None
        self.coupled = False
        self.min_iterations = None
        self.max_iterations = None
        self.tolerance = None
        self.backflow_stabilization_coefficient = None
        self.density = None
        self.conductivity = None
        self.source_term = None
        self.viscosity = None
        self.linear_solver = LinearSolver()
        self.outputs = []
        self.alias = None

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(vars(self))

    def __eq__(self, other):
        check = ["equation_type", "coupled", "min_iterations", "max_iterations", "tolerance",
                 "backflow_stabilization_coefficient", "density", "conductivity", "source_term"]
        attributes = vars(self)
        other_attributes = vars(other)
        return all(other_attributes[key] == attributes[key] for key in check)

    def __ne__(self, other):
        return not self.__eq__(other)


class Equation(EquationBase):
    def __init__(self):
        super().__init__()
        self.file = minidom.Document()
        self.meshes = []
        self.boundary_conditions = []
        self.faces = []

    def __str__(self):
        return self.toxml().toprettyxml()

    def __repr__(self):
        return self.toxml().toprettyxml()

    def set_type(self, equation_type):
        if equation_type not in ["fluid", "scalarTransport"]:
            raise ValueError("Equation type must be 'fluid' or 'scalarTransport'.")
        self.equation_type = equation_type

    def set_coupled(self, coupled):
        if not isinstance(coupled, bool):
            raise ValueError("Coupled must be a boolean.")
        self.coupled = coupled

    def set_min_iterations(self, min_iterations):
        if not isinstance(min_iterations, int):
            raise ValueError("Min iterations must be an integer.")
        self.min_iterations = min_iterations

    def set_max_iterations(self, max_iterations):
        if not isinstance(max_iterations, int):
            raise ValueError("Max iterations must be an integer.")
        self.max_iterations = max_iterations

    def set_tolerance(self, tolerance):
        if not isinstance(tolerance, float):
            raise ValueError("Tolerance must be a float.")
        self.tolerance = tolerance

    def set_backflow_stabilization_coefficient(self, backflow_stabilization_coefficient):
        if not isinstance(backflow_stabilization_coefficient, float):
            raise ValueError("Backflow stabilization coefficient must be a float.")
        self.backflow_stabilization_coefficient = backflow_stabilization_coefficient

    def set_density(self, density):
        if not isinstance(density, float):
            raise ValueError("Density must be a float.")
        self.density = density

    def set_viscosity(self, viscosity_type, viscosity_value):
        if not isinstance(viscosity_type, str):
            raise ValueError("Viscosity type must be a string.")
        if viscosity_type not in ["Constant"]:
            raise ValueError("Viscosity type must be 'Constant'.")
        if not isinstance(viscosity_value, float):
            raise ValueError("Viscosity value must be a float.")
        self.viscosity = {"type": viscosity_type, "value": viscosity_value}

    def set_conductivity(self, conductivity):
        if not isinstance(conductivity, float):
            raise ValueError("Conductivity must be a float.")
        self.conductivity = conductivity

    def set_source_term(self, source_term):
        if not isinstance(source_term, float):
            raise ValueError("Source term must be a float.")
        self.source_term = source_term

    def add_mesh(self, mesh):
        if not isinstance(mesh, GeneralMesh):
            raise ValueError("Mesh must be a GeneralMesh object.")
        self.meshes.append(mesh)
        self.faces.extend(list(mesh.faces.keys()))

    def add_bc(self, face, bc_type, value, time_varying=False,
               time_varying_file=None, impose_flux=False, profile=None,
               zero_out_perimeter=True):
        if face not in self.faces:
            raise ValueError("Face not found in mesh.")
        for bc in self.boundary_conditions:
            if bc.name == face:
                raise ValueError("Boundary condition already exists for face.")
        bc = BoundaryCondition()
        bc.set_name(face)
        bc.set_type(bc_type)
        if not time_varying:
            bc.set_value(value)
        else:
            if isinstance(time_varying_file, type(None)):
                raise ValueError("Time varying file must be set.")
            bc.set_time_varying(time_varying)
            bc.set_time_varying_file(time_varying_file)
        bc.impose_flux = impose_flux
        bc.profile = profile
        self.boundary_conditions.append(bc)

    def reassign_bc(self, face, bc_type, value, time_varying=False,
                    time_varying_file=None, impose_flux=False, profile=None):
        if face not in self.faces:
            raise ValueError("Face not found in mesh.")
        bc = BoundaryCondition()
        bc.set_name(face)
        bc.set_type(bc_type)
        if not time_varying:
            bc.set_value(value)
        else:
            if isinstance(time_varying_file, type(None)):
                raise ValueError("Time varying file must be set.")
            bc.set_time_varying(time_varying)
            bc.set_time_varying_file(time_varying_file)
        bc.impose_flux = impose_flux
        bc.profile = profile
        for i, b in enumerate(self.boundary_conditions):
            if b.name == face:
                self.boundary_conditions[i] = bc
                break

    def toxml(self):
        add_equation = self.file.createElement("Add_equation")
        if not isinstance(self.equation_type, type(None)):
            add_equation.setAttribute("type", self.equation_type)
        else:
            raise ValueError("Equation type must be set.")

        if not isinstance(self.coupled, type(None)):
            coupled = self.file.createElement("Coupled")
            if self.coupled:
                coupled.appendChild(self.file.createTextNode("true"))
            else:
                coupled.appendChild(self.file.createTextNode("false"))
            add_equation.appendChild(coupled)

        if not isinstance(self.min_iterations, type(None)):
            min_iterations = self.file.createElement("Min_iterations")
            min_iterations.appendChild(self.file.createTextNode(str(self.min_iterations)))
            add_equation.appendChild(min_iterations)

        if not isinstance(self.max_iterations, type(None)):
            max_iterations = self.file.createElement("Max_iterations")
            max_iterations.appendChild(self.file.createTextNode(str(self.max_iterations)))
            add_equation.appendChild(max_iterations)

        if not isinstance(self.tolerance, type(None)):
            tolerance = self.file.createElement("Tolerance")
            tolerance.appendChild(self.file.createTextNode(str(self.tolerance)))
            add_equation.appendChild(tolerance)

        if not isinstance(self.backflow_stabilization_coefficient, type(None)):
            backflow_stabilization_coefficient = self.file.createElement("Backflow_stabilization_coefficient")
            backflow_stabilization_coefficient.appendChild(self.file.createTextNode(str(self.backflow_stabilization_coefficient)))
            add_equation.appendChild(backflow_stabilization_coefficient)

        if not isinstance(self.density, type(None)):
            density = self.file.createElement("Density")
            density.appendChild(self.file.createTextNode(str(self.density)))
            add_equation.appendChild(density)

        if not isinstance(self.conductivity, type(None)):
            conductivity = self.file.createElement("Conductivity")
            conductivity.appendChild(self.file.createTextNode(str(self.conductivity)))
            add_equation.appendChild(conductivity)

        if not isinstance(self.source_term, type(None)):
            source_term = self.file.createElement("Source_term")
            source_term.appendChild(self.file.createTextNode(str(self.source_term)))
            add_equation.appendChild(source_term)

        if not isinstance(self.viscosity, type(None)):
            viscosity = self.file.createElement("Viscosity")
            viscosity.setAttribute("model", self.viscosity["type"])
            value = self.file.createElement("Value")
            value.appendChild(self.file.createTextNode(str(self.viscosity["value"])))
            viscosity.appendChild(value)
            add_equation.appendChild(viscosity)

        for out in self.outputs:
            if isinstance(out, Output):
                add_equation.appendChild(out.toxml())

        if isinstance(self.alias, Output) and self.alias.output_type == "Alias":
            add_equation.appendChild(self.alias.toxml())

        if isinstance(self.linear_solver, LinearSolver):
            add_equation.appendChild(self.linear_solver.toxml())

        for boundary_condition in self.boundary_conditions:
            add_equation.appendChild(boundary_condition.toxml())

        return add_equation
