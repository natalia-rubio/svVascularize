from svv.simulation.equation import Equation
from svv.simulation.output import Output


class FluidEquation(Equation):
    def __init__(self):
        super().__init__()
        self.equation_type = "fluid"
        self.density = 1.06
        self.tolerance = 1e-4
        self.coupled = True
        self.min_iterations = 3
        self.max_iterations = 10
        self.backflow_stabilization_coefficient = 0.2
        output = Output()
        output.set_type("Spatial")
        output.pressure = True
        output.velocity = True
        output.traction = True
        output.wss = True
        self.outputs.append(output)
        self.linear_solver.set_type("NS")
        self.linear_solver.ns_cg_max_iterations = 500
        self.linear_solver.ns_gm_max_iterations = 3
        self.linear_solver.ns_cg_tolerance = 1e-4
        self.linear_solver.ns_gm_tolerance = 1e-4
        self.linear_solver.krylov_space_dimension = 100
        self.linear_solver.absolute_tolerance = 1e-12
        self.linear_solver.tolerance = 1e-4
        self.linear_solver.max_iterations = 10

    def add_inlet(self, face, value=None, bc_type='Dirichlet', time_varying_file=None, profile='Parabolic',
                  impose_flux=True):
        """
        Add an inlet boundary condition to the fluid equation.
        """
        if isinstance(time_varying_file, type(None)):
            time_varying = False
        else:
            time_varying = True
        self.add_bc(face, bc_type, value, time_varying=time_varying,
                    profile=profile, impose_flux=impose_flux)

    def add_outlet(self, face, value=None, bc_type='Neumann', time_varying_file=None,
                   impose_flux=False):
        """
        Add an outlet boundary condition to the fluid equation.
        """
        if isinstance(time_varying_file, type(None)):
            time_varying = False
        else:
            time_varying = True
        self.add_bc(face, bc_type, value, time_varying=time_varying, impose_flux=impose_flux)

    def add_wall(self, face, value=0.0, bc_type='Dirichlet', time_varying_file=None):
        """
        Add a wall boundary condition to the fluid equation.
        """
        if isinstance(time_varying_file, type(None)):
            time_varying = False
        else:
            time_varying = True
        self.add_bc(face, bc_type, value, time_varying=time_varying)

    def check_bcs(self):
        """
        Check that all boundary conditions are set.
        """
        for face in self.faces:
            found = False
            for bc in self.boundary_conditions:
                if bc.name == face:
                    found = True
                    break
            if not found:
                raise ValueError("Boundary condition not set for face: " + face)