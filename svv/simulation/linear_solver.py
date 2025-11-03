from xml.dom import minidom


class LinearSolverBase:
    def __init__(self):
        self.solver_type = "NS"
        self.absolute_tolerance = 1.0e-10
        self.tolerance = 0.5
        self.max_iterations = 1000
        self.preconditioner = None
        self.ns_cg_max_iterations = 1000
        self.ns_cg_tolerance = 1.0e-2
        self.ns_gm_max_iterations = 1000
        self.ns_gm_tolerance = 1.0e-2
        self.use_trilinos_for_assembly = False
        self.krylov_space_dimension = 50

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(vars(self))

    def __eq__(self, other):
        check = ["solver_type", "absolute_tolerance", "tolerance", "max_iterations", "preconditioner",
                 "ns_cg_max_iterations", "ns_cg_tolerance", "ns_gm_max_iterations", "ns_gm_tolerance",
                 "use_trilinos_for_assembly"]
        attributes = vars(self)
        other_attributes = vars(other)
        return all(other_attributes[key] == attributes[key] for key in check)

    def __ne__(self, other):
        return not self.__eq__(other)


class LinearSolver(LinearSolverBase):
    def __init__(self):
        super().__init__()
        self.file = minidom.Document()

    def __str__(self):
        return self.toxml().toprettyxml()

    def __repr__(self):
        return self.toxml().toprettyxml()

    def set_type(self, solver_type):
        """
        Set the linear solver type. Options are 'CG', 'GMRES', 'NS', 'BICGS'.
        Note that not all linear solvers and preconditioners are compatible.
        """
        if solver_type not in ["CG", "GMRES", "NS", "BICGS"]:
            raise ValueError("Solver type must be 'CG' or 'GMRES'.")
        self.solver_type = solver_type

    def toxml(self):
        ls = self.file.createElement("LS")
        if not isinstance(self.solver_type, type(None)):
            ls.setAttribute("type", self.solver_type)
        else:
            raise ValueError("Solver type must be set.")

        if not isinstance(self.absolute_tolerance, type(None)):
            absolute_tolerance = self.file.createElement("Absolute_tolerance")
            absolute_tolerance.appendChild(self.file.createTextNode(str(self.absolute_tolerance)))
            ls.appendChild(absolute_tolerance)

        if not isinstance(self.tolerance, type(None)):
            tolerance = self.file.createElement("Tolerance")
            tolerance.appendChild(self.file.createTextNode(str(self.tolerance)))
            ls.appendChild(tolerance)

        if not isinstance(self.max_iterations, type(None)):
            max_iterations = self.file.createElement("Max_iterations")
            max_iterations.appendChild(self.file.createTextNode(str(self.max_iterations)))
            ls.appendChild(max_iterations)

        if not isinstance(self.preconditioner, type(None)):
            preconditioner = self.file.createElement("Preconditioner")
            preconditioner.appendChild(self.file.createTextNode(self.preconditioner))
            ls.appendChild(preconditioner)

        if not isinstance(self.ns_cg_max_iterations, type(None)):
            ns_cg_max_iterations = self.file.createElement("NS_CG_max_iterations")
            ns_cg_max_iterations.appendChild(self.file.createTextNode(str(self.ns_cg_max_iterations)))
            ls.appendChild(ns_cg_max_iterations)

        if not isinstance(self.ns_cg_tolerance, type(None)):
            ns_cg_tolerance = self.file.createElement("NS_CG_tolerance")
            ns_cg_tolerance.appendChild(self.file.createTextNode(str(self.ns_cg_tolerance)))
            ls.appendChild(ns_cg_tolerance)

        if not isinstance(self.ns_gm_max_iterations, type(None)):
            ns_gm_max_iterations = self.file.createElement("NS_GM_max_iterations")
            ns_gm_max_iterations.appendChild(self.file.createTextNode(str(self.ns_gm_max_iterations)))
            ls.appendChild(ns_gm_max_iterations)

        if not isinstance(self.ns_gm_tolerance, type(None)):
            ns_gm_tolerance = self.file.createElement("NS_GM_tolerance")
            ns_gm_tolerance.appendChild(self.file.createTextNode(str(self.ns_gm_tolerance)))
            ls.appendChild(ns_gm_tolerance)

        if self.use_trilinos_for_assembly:
            use_trilinos_for_assembly = self.file.createElement("Use_trilinos_for_assembly")
            use_trilinos_for_assembly.appendChild(self.file.createTextNode("true"))
            ls.appendChild(use_trilinos_for_assembly)

        if not isinstance(self.krylov_space_dimension, type(None)):
            krylov_space_dimension = self.file.createElement("Krylov_space_dimension")
            krylov_space_dimension.appendChild(self.file.createTextNode(str(self.krylov_space_dimension)))
            ls.appendChild(krylov_space_dimension)
        return ls
