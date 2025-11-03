from xml.dom import minidom


class GeneralSimulationParameters:
    def __init__(self):
        self.file = minidom.Document()
        self.continue_previous_simulation = False
        self.number_of_spatial_dimensions = 3
        self.number_of_time_steps = 1
        self.time_step_size = 0.001
        self.precomputed_time_step_size = None
        self.spectral_radius_of_infinite_time_step = 0.5
        self.searched_file_name_to_trigger_stop = "STOP_SIM"
        self.save_results_to_vtk_format = True
        self.name_prefix_of_saved_vtk_files = "result"
        self.increment_in_saving_vtk_files = 1
        self.start_saving_after_time_step = 0
        self.increment_in_saving_restart_files = 100
        self.convert_bin_to_vtk_format = False
        self.use_precomputed_solution = False
        self.precomputed_solution_file_path = None
        self.precomputed_solution_field_name = None
        self.overwrite_restart_file = False
        self.save_averaged_results = False
        self.simulation_initialization_file_path = None
        self.save_results_in_folder = None
        self.restart_file_name = None
        self.check_ien_order = False
        self.verbose = True
        self.warning = False
        self.debug = True

    def __str__(self):
        return self.toxml().toprettyxml()

    def __repr__(self):
        return self.toxml().toprettyxml()

    def toxml(self):
        general_simulation_parameters = self.file.createElement("GeneralSimulationParameters")
        continue_previous_simulation = self.file.createElement("Continue_previous_simulation")
        if self.continue_previous_simulation:
            continue_previous_simulation.appendChild(self.file.createTextNode(str(1)))
        else:
            continue_previous_simulation.appendChild(self.file.createTextNode(str(0)))
        general_simulation_parameters.appendChild(continue_previous_simulation)

        number_of_spatial_dimensions = self.file.createElement("Number_of_spatial_dimensions")
        number_of_spatial_dimensions.appendChild(self.file.createTextNode(str(self.number_of_spatial_dimensions)))
        general_simulation_parameters.appendChild(number_of_spatial_dimensions)

        number_of_time_steps = self.file.createElement("Number_of_time_steps")
        number_of_time_steps.appendChild(self.file.createTextNode(str(self.number_of_time_steps)))
        general_simulation_parameters.appendChild(number_of_time_steps)

        time_step_size = self.file.createElement("Time_step_size")
        time_step_size.appendChild(self.file.createTextNode(str(self.time_step_size)))
        general_simulation_parameters.appendChild(time_step_size)

        spectral_radius_of_infinite_time_step = self.file.createElement("Spectral_radius_of_infinite_time_step")
        spectral_radius_of_infinite_time_step.appendChild(self.file.createTextNode(str(self.spectral_radius_of_infinite_time_step)))
        general_simulation_parameters.appendChild(spectral_radius_of_infinite_time_step)

        searched_file_name_to_trigger_stop = self.file.createElement("Searched_file_name_to_trigger_stop")
        searched_file_name_to_trigger_stop.appendChild(self.file.createTextNode(str(self.searched_file_name_to_trigger_stop)))
        general_simulation_parameters.appendChild(searched_file_name_to_trigger_stop)

        save_results_to_vtk_format = self.file.createElement("Save_results_to_VTK_format")
        if self.save_results_to_vtk_format:
            save_results_to_vtk_format.appendChild(self.file.createTextNode(str(1)))
        else:
            save_results_to_vtk_format.appendChild(self.file.createTextNode(str(0)))
        general_simulation_parameters.appendChild(save_results_to_vtk_format)

        name_prefix_of_saved_vtk_files = self.file.createElement("Name_prefix_of_saved_VTK_files")
        name_prefix_of_saved_vtk_files.appendChild(self.file.createTextNode(str(self.name_prefix_of_saved_vtk_files)))
        general_simulation_parameters.appendChild(name_prefix_of_saved_vtk_files)

        increment_in_saving_vtk_files = self.file.createElement("Increment_in_saving_VTK_files")
        increment_in_saving_vtk_files.appendChild(self.file.createTextNode(str(self.increment_in_saving_vtk_files)))
        general_simulation_parameters.appendChild(increment_in_saving_vtk_files)

        start_saving_after_time_step = self.file.createElement("Start_saving_after_time_step")
        start_saving_after_time_step.appendChild(self.file.createTextNode(str(self.start_saving_after_time_step)))
        general_simulation_parameters.appendChild(start_saving_after_time_step)

        increment_in_saving_restart_files = self.file.createElement("Increment_in_saving_restart_files")
        increment_in_saving_restart_files.appendChild(self.file.createTextNode(str(self.increment_in_saving_restart_files)))
        general_simulation_parameters.appendChild(increment_in_saving_restart_files)

        convert_bin_to_vtk_format = self.file.createElement("Convert_BIN_to_VTK_format")
        if self.convert_bin_to_vtk_format:
            convert_bin_to_vtk_format.appendChild(self.file.createTextNode(str(1)))
        else:
            convert_bin_to_vtk_format.appendChild(self.file.createTextNode(str(0)))
        general_simulation_parameters.appendChild(convert_bin_to_vtk_format)

        if self.use_precomputed_solution:
            use_precomputed_solution = self.file.createElement("Use_precomputed_solution")
            use_precomputed_solution.appendChild(self.file.createTextNode('true'))
            general_simulation_parameters.appendChild(use_precomputed_solution)

            if not isinstance(self.precomputed_time_step_size, type(None)):
                precomputed_time_step_size = self.file.createElement("Precomputed_time_step_size")
                precomputed_time_step_size.appendChild(self.file.createTextNode(str(self.precomputed_time_step_size)))
                general_simulation_parameters.appendChild(precomputed_time_step_size)

            if not isinstance(self.precomputed_solution_file_path, type(None)):
                precomputed_solution_file_path = self.file.createElement("Precomputed_solution_file_path")
                precomputed_solution_file_path.appendChild(self.file.createTextNode(str(self.precomputed_solution_file_path)))
                general_simulation_parameters.appendChild(precomputed_solution_file_path)
            else:
                raise ValueError("Precomputed solution file path must be set.")

            if not isinstance(self.precomputed_solution_field_name, type(None)):
                precomputed_solution_field_name = self.file.createElement("Precomputed_solution_field_name")
                precomputed_solution_field_name.appendChild(self.file.createTextNode(str(self.precomputed_solution_field_name)))
                general_simulation_parameters.appendChild(precomputed_solution_field_name)
            else:
                raise ValueError("Precomputed solution field name must be set.")

        if self.overwrite_restart_file:
            overwrite_restart_file = self.file.createElement("Overwrite_restart_file")
            overwrite_restart_file.appendChild(self.file.createTextNode(str(1)))
            general_simulation_parameters.appendChild(overwrite_restart_file)

        if self.save_averaged_results:
            save_averaged_results = self.file.createElement("Save_averaged_results")
            save_averaged_results.appendChild(self.file.createTextNode(str(1)))
            general_simulation_parameters.appendChild(save_averaged_results)

        if not isinstance(self.simulation_initialization_file_path, type(None)):
            simulation_initialization_file_path = self.file.createElement("Simulation_initialization_file_path")
            simulation_initialization_file_path.appendChild(self.file.createTextNode(str(self.simulation_initialization_file_path)))
            general_simulation_parameters.appendChild(simulation_initialization_file_path)

        if not isinstance(self.save_results_in_folder, type(None)):
            save_results_in_folder = self.file.createElement("Save_results_in_folder")
            save_results_in_folder.appendChild(self.file.createTextNode(str(self.save_results_in_folder)))
            general_simulation_parameters.appendChild(save_results_in_folder)

        if not isinstance(self.restart_file_name, type(None)):
            restart_file_name = self.file.createElement("Restart_file_name")
            restart_file_name.appendChild(self.file.createTextNode(str(self.restart_file_name)))
            general_simulation_parameters.appendChild(restart_file_name)

        if self.check_ien_order:
            check_ien_order = self.file.createElement("Check_IEN_order")
            check_ien_order.appendChild(self.file.createTextNode(str(1)))
            general_simulation_parameters.appendChild(check_ien_order)

        verbose = self.file.createElement("Verbose")
        if self.verbose:
            verbose.appendChild(self.file.createTextNode(str(1)))
        else:
            verbose.appendChild(self.file.createTextNode(str(0)))
        general_simulation_parameters.appendChild(verbose)

        warning = self.file.createElement("Warning")
        if self.warning:
            warning.appendChild(self.file.createTextNode(str(1)))
        else:
            warning.appendChild(self.file.createTextNode(str(0)))
        general_simulation_parameters.appendChild(warning)

        debug = self.file.createElement("Debug")
        if self.debug:
            debug.appendChild(self.file.createTextNode(str(1)))
        else:
            debug.appendChild(self.file.createTextNode(str(0)))
        general_simulation_parameters.appendChild(debug)
        return general_simulation_parameters
