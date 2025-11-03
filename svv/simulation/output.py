from xml.dom import minidom


class OutputBase:
    def __init__(self):
        self.output_type = None
        self.pressure = False
        self.velocity = False
        self.traction = False
        self.displacement = False
        self.wss = False
        self.temperature = False
        self.heat_flux = False
        self.alias = None

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(vars(self))

    def __eq__(self, other):
        check = ["output_type", "pressure", "velocity", "traction", "displacement", "wss", "temperature", "heat_flux"]
        attributes = vars(self)
        other_attributes = vars(other)
        return all(other_attributes[key] == attributes[key] for key in check)

    def __ne__(self, other):
        return not self.__eq__(other)


class Output(OutputBase):
    def __init__(self):
        super().__init__()
        self.file = minidom.Document()

    def __str__(self):
        return self.toxml().toprettyxml()

    def __repr__(self):
        return self.toxml().toprettyxml()

    def set_type(self, output_type):
        if output_type not in ["Boundary_integral", "Spatial", "Volume_integral", "Alias"]:
            raise ValueError("Output type must be 'Boundary_integral', 'Spatial', or 'Volume_integral'.")
        self.output_type = output_type

    def toxml(self):
        output = self.file.createElement("Output")
        if not isinstance(self.output_type, type(None)):
            output.setAttribute("type", self.output_type)
        else:
            raise ValueError("Output type must be set.")

        if self.pressure:
            pressure = self.file.createElement("Pressure")
            pressure.appendChild(self.file.createTextNode("true"))
            output.appendChild(pressure)

        if self.velocity:
            velocity = self.file.createElement("Velocity")
            velocity.appendChild(self.file.createTextNode("true"))
            output.appendChild(velocity)

        if self.traction:
            traction = self.file.createElement("Traction")
            traction.appendChild(self.file.createTextNode("true"))
            output.appendChild(traction)

        if self.displacement:
            displacement = self.file.createElement("Displacement")
            displacement.appendChild(self.file.createTextNode("true"))
            output.appendChild(displacement)

        if self.wss:
            wss = self.file.createElement("WSS")
            wss.appendChild(self.file.createTextNode("true"))
            output.appendChild(wss)

        if self.temperature:
            temperature = self.file.createElement("Temperature")
            temperature.appendChild(self.file.createTextNode("true"))
            output.appendChild(temperature)

        if self.heat_flux:
            heat_flux = self.file.createElement("Heat_flux")
            heat_flux.appendChild(self.file.createTextNode("true"))
            output.appendChild(heat_flux)

        if self.output_type == 'Alias':
            if isinstance(self.alias, type(None)):
                raise ValueError("Alias must be set.")
            valid_outputs = ["Pressure", "Velocity", "Traction", "Displacement", "WSS", "Temperature", "Heat_flux"]
            for key, value in self.alias.items():
                if key not in valid_outputs:
                    raise ValueError("Alias key must be a valid output type. {} is not valid.".format(key))
                alias = self.file.createElement(key)
                alias.appendChild(self.file.createTextNode(value))
                output.appendChild(alias)
        return output
