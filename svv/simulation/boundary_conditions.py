from xml.dom import minidom


class BoundaryConditionBase:
    def __init__(self):
        self.name = None
        self.bc_type = None
        self.value = None
        self.time_varying = False
        self.time_varying_file = None
        self.impose_flux = False
        self.profile = None
        self.zero_out_perimeter = False

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(vars(self))

    def __eq__(self, other):
        check = ["bc_type", "value", "time_varying", "time_varying_file", "impose_flux", "profile"]
        attributes = vars(self)
        other_attributes = vars(other)
        return all(other_attributes[key] == attributes[key] for key in check)

    def __ne__(self, other):
        return not self.__eq__(other)


class BoundaryCondition(BoundaryConditionBase):
    def __init__(self):
        super().__init__()
        self.file = minidom.Document()

    def __str__(self):
        return self.toxml().toprettyxml()

    def __repr__(self):
        return self.toxml().toprettyxml()

    def set_type(self, bc_type):
        if bc_type not in ["Dirichlet", "Neumann", "Dir", "Neu"]:
            raise ValueError("Boundary condition type must be 'Dirichlet' or 'Neumann'.")
        self.bc_type = bc_type

    def set_name(self, name):
        if not isinstance(name, str):
            raise ValueError("Name must be a string.")
        self.name = name

    def set_value(self, value):
        if not isinstance(value, float) and not isinstance(value, int):
            raise ValueError("Value must be a float of int.")
        self.value = value

    def set_time_varying(self, time_varying):
        if not isinstance(time_varying, bool):
            raise ValueError("Time varying must be a boolean.")
        self.time_varying = time_varying

    def set_time_varying_file(self, time_varying_file):
        if not isinstance(time_varying_file, str):
            raise ValueError("Time varying file must be a string.")
        self.time_varying_file = time_varying_file

    def set_impose_flux(self, impose_flux):
        if not isinstance(impose_flux, bool):
            raise ValueError("Imposed flux must be a boolean.")
        self.impose_flux = impose_flux

    def set_profile(self, profile):
        if not isinstance(profile, str):
            raise ValueError("Profile must be a string.")
        if profile not in ["Parabolic", "Plug"]:
            raise ValueError("Profile must be 'Parabolic' or 'Plug' flow.")
        self.profile = profile

    def set_zero_out_perimeter(self, zero_out_perimeter):
        if not isinstance(zero_out_perimeter, bool):
            raise ValueError("Zero out perimeter must be a boolean.")
        self.zero_out_perimeter = zero_out_perimeter

    def toxml(self):
        add_bc = self.file.createElement("Add_BC")
        if not isinstance(self.name, type(None)):
            add_bc.setAttribute("name", self.name)

        if not isinstance(self.bc_type, type(None)):
            type_element = minidom.Element("Type")
            type_element.appendChild(self.file.createTextNode(str(self.bc_type)))
            add_bc.appendChild(type_element)

        if not isinstance(self.value, type(None)):
            value_element = minidom.Element("Value")
            value_element.appendChild(self.file.createTextNode(str(self.value)))
            add_bc.appendChild(value_element)

        time_dependence_element = minidom.Element("Time_dependence")

        if self.time_varying:
            time_dependence_element.appendChild(self.file.createTextNode("Unsteady"))
            add_bc.appendChild(time_dependence_element)
            time_varying_file_element = minidom.Element("Temporal_values_file_path")
            time_varying_file_element.appendChild(self.file.createTextNode(self.time_varying_file))
            add_bc.appendChild(time_varying_file_element)
        else:
            time_dependence_element.appendChild(self.file.createTextNode("Steady"))

        if self.impose_flux:
            impose_flux_element = minidom.Element("Impose_flux")
            impose_flux_element.appendChild(self.file.createTextNode("true"))
            add_bc.appendChild(impose_flux_element)

        if not isinstance(self.profile, type(None)):
            profile_element = minidom.Element("Profile")
            profile_element.appendChild(self.file.createTextNode(str(self.profile)))
            add_bc.appendChild(profile_element)

        zero_out_perimeter_element = minidom.Element("Zero_out_perimeter")
        if self.zero_out_perimeter:
            zero_out_perimeter_element.appendChild(self.file.createTextNode("true"))
            add_bc.appendChild(zero_out_perimeter_element)
        else:
            zero_out_perimeter_element.appendChild(self.file.createTextNode("false"))
            add_bc.appendChild(zero_out_perimeter_element)
        return add_bc
