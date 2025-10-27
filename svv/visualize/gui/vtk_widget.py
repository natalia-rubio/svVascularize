"""
VTK/PyVista widget for 3D domain visualization.
"""
import os
import sys

# Configure software rendering only on Linux to ensure in-window rendering
# without imposing Mesa settings on Windows/macOS.
if sys.platform.startswith('linux'):
    # Allow users to opt out of software GL and use system drivers
    gl_mode = os.environ.get('SVV_GUI_GL_MODE', 'software')  # 'software' or 'system'
    if gl_mode != 'system':
        # Prefer software rendering to avoid GPU/driver issues on varied Linux setups
        os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
        os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')
        os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
        # Allow overriding the Mesa software driver
        sw_driver = os.environ.get('SVV_GUI_SOFTWARE_DRIVER', 'llvmpipe')
        os.environ.setdefault('MESA_LOADER_DRIVER_OVERRIDE', sw_driver)

        # Qt side: force software OpenGL and use XCB when on Wayland
        os.environ.setdefault('QT_OPENGL', 'software')
        if 'WAYLAND_DISPLAY' in os.environ:
            os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

        # Allow explicit override of the DRI drivers path
        override_dri = os.environ.get('SVV_LIBGL_DRIVERS_PATH')
        if override_dri and os.path.isdir(override_dri):
            os.environ.setdefault('LIBGL_DRIVERS_PATH', override_dri)
        else:
            # Common conda‑forge location
            conda_prefix = os.environ.get('CONDA_PREFIX', '')
            candidates = []
            if conda_prefix:
                candidates.extend([
                    os.path.join(conda_prefix, 'lib', 'dri'),
                    os.path.join(conda_prefix, 'x86_64-conda-linux-gnu', 'sysroot', 'usr', 'lib64', 'dri'),
                ])
            for dri_path in candidates:
                if os.path.isdir(dri_path):
                    os.environ.setdefault('LIBGL_DRIVERS_PATH', dri_path)
                    break

    # Optional debug output for GL setup
    if os.environ.get('SVV_GUI_DEBUG_GL') == '1':
        debug_state = {
            'gl_mode': gl_mode,
            'SVV_GUI_SOFTWARE_DRIVER': os.environ.get('SVV_GUI_SOFTWARE_DRIVER'),
            'MESA_LOADER_DRIVER_OVERRIDE': os.environ.get('MESA_LOADER_DRIVER_OVERRIDE'),
            'LIBGL_DRIVERS_PATH': os.environ.get('LIBGL_DRIVERS_PATH'),
            'QT_QPA_PLATFORM': os.environ.get('QT_QPA_PLATFORM'),
            'QT_OPENGL': os.environ.get('QT_OPENGL'),
            'WAYLAND_DISPLAY': os.environ.get('WAYLAND_DISPLAY'),
        }
        print('[svv.gui] GL debug config:', debug_state)

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget
from PySide6.QtCore import Signal


class VTKWidget(QWidget):
    """
    Widget for 3D visualization of Domain objects using PyVista.
    """

    # Signals
    point_picked = Signal(object)  # Emitted when a point is picked (numpy array)

    def __init__(self, parent=None):
        """
        Initialize the VTK widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.domain = None
        self.domain_actor = None
        self.points_actors = []
        self.direction_actors = []
        self.tree_actors = []

        # Attempt to import PyVista and PyVistaQt lazily with helpful errors
        try:
            from pyvistaqt import QtInteractor as _QtInteractor
            import pyvista as _pv
        except Exception as e:
            # Provide a clearer error, especially for missing libffi on Linux
            msg = (
                "Failed to import PyVista/PyVistaQt. On Linux, ensure libffi is installed "
                "(conda: `conda install -c conda-forge libffi>=3.4,<3.5`; system: `sudo apt install libffi8` on Ubuntu 22.04 "
                "or `libffi7` on Ubuntu 20.04). Also install dependencies: PySide6, pyvista, pyvistaqt."
            )
            raise RuntimeError(msg) from e

        self._pv = _pv
        self._QtInteractor = _QtInteractor

        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Create PyVista plotter
        self.plotter = self._QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # Enable point picking
        self.plotter.enable_point_picking(
            callback=self._on_point_picked,
            show_message=True,
            color='red',
            point_size=10
        )

        # Set background
        self.plotter.set_background('white')

        # Initial camera setup
        self.plotter.camera_position = 'iso'

    def set_domain(self, domain):
        """
        Set and visualize the domain.

        Parameters
        ----------
        domain : svv.domain.Domain
            Domain object to visualize
        """
        self.domain = domain
        self.clear()

        # Add domain boundary if available
        if hasattr(domain, 'boundary') and domain.boundary is not None:
            self.domain_actor = self.plotter.add_mesh(
                domain.boundary,
                color='lightblue',
                opacity=0.3,
                show_edges=True,
                edge_color='gray',
                name='domain'
            )

        # Reset camera to show full domain
        self.plotter.reset_camera()
        self.plotter.render()

    def add_start_point(self, point, index=None, color='red'):
        """
        Add a start point marker to the visualization.

        Parameters
        ----------
        point : array-like
            3D coordinates of the point [x, y, z]
        index : int, optional
            Index/label for the point
        color : str, optional
            Color of the point marker

        Returns
        -------
        actor
            PyVista actor for the point
        """
        point = np.asarray(point).flatten()

        # Create sphere marker
        sphere = self._pv.Sphere(radius=self._get_marker_size(), center=point)
        actor = self.plotter.add_mesh(
            sphere,
            color=color,
            name=f'start_point_{len(self.points_actors)}'
        )

        # Add label if index provided
        if index is not None:
            self.plotter.add_point_labels(
                [point],
                [f'P{index}'],
                point_size=10,
                font_size=12,
                text_color='black',
                name=f'label_{len(self.points_actors)}'
            )

        self.points_actors.append(actor)
        self.plotter.render()
        return actor

    def add_direction(self, point, direction, length=None, color='blue'):
        """
        Add a direction vector visualization.

        Parameters
        ----------
        point : array-like
            Starting point [x, y, z]
        direction : array-like
            Direction vector [dx, dy, dz]
        length : float, optional
            Length of the arrow. If None, uses domain characteristic length
        color : str, optional
            Color of the arrow

        Returns
        -------
        actor
            PyVista actor for the direction arrow
        """
        point = np.asarray(point).flatten()
        direction = np.asarray(direction).flatten()
        direction = direction / np.linalg.norm(direction)

        if length is None:
            length = self._get_arrow_length()

        # Create arrow
        end_point = point + direction * length
        arrow = self._pv.Line(point, end_point)

        actor = self.plotter.add_mesh(
            arrow,
            color=color,
            line_width=3,
            name=f'direction_{len(self.direction_actors)}'
        )

        # Add arrow head
        cone_height = length * 0.15
        cone = self._pv.Cone(
            center=end_point,
            direction=direction,
            height=cone_height,
            radius=cone_height * 0.5
        )
        cone_actor = self.plotter.add_mesh(
            cone,
            color=color,
            name=f'direction_cone_{len(self.direction_actors)}'
        )

        self.direction_actors.append((actor, cone_actor))
        self.plotter.render()
        return actor

    def add_tree(self, tree, color='red'):
        """
        Add a Tree visualization.

        Parameters
        ----------
        tree : svv.tree.Tree
            Tree object to visualize
        color : str, optional
            Color for the tree vessels

        Returns
        -------
        list
            List of actors for the tree vessels
        """
        actors = []
        for i in range(tree.data.shape[0]):
            center = (tree.data[i, 0:3] + tree.data[i, 3:6]) / 2
            direction = tree.data.get('w_basis', i)
            radius = tree.data.get('radius', i)
            length = tree.data.get('length', i)

            vessel = self._pv.Cylinder(
                center=center,
                direction=direction,
                radius=radius,
                height=length
            )
            actor = self.plotter.add_mesh(
                vessel,
                color=color,
                name=f'tree_vessel_{i}'
            )
            actors.append(actor)

        self.tree_actors.extend(actors)
        self.plotter.render()
        return actors

    def clear_points(self):
        """Clear all start point markers."""
        for actor in self.points_actors:
            self.plotter.remove_actor(actor)
        self.points_actors.clear()
        self.plotter.render()

    def clear_directions(self):
        """Clear all direction arrows."""
        for actor_tuple in self.direction_actors:
            for actor in actor_tuple:
                self.plotter.remove_actor(actor)
        self.direction_actors.clear()
        self.plotter.render()

    def clear_trees(self):
        """Clear all tree visualizations."""
        for actor in self.tree_actors:
            self.plotter.remove_actor(actor)
        self.tree_actors.clear()
        self.plotter.render()

    def clear(self):
        """Clear all visualizations except the domain."""
        self.clear_points()
        self.clear_directions()
        self.clear_trees()

    def reset_camera(self):
        """Reset the camera to show the full domain."""
        self.plotter.reset_camera()
        self.plotter.render()

    def toggle_domain_visibility(self):
        """Toggle the visibility of the domain mesh."""
        if self.domain_actor is not None:
            self.domain_actor.SetVisibility(not self.domain_actor.GetVisibility())
            self.plotter.render()

    def _on_point_picked(self, point):
        """
        Callback for point picking.

        Parameters
        ----------
        point : array-like
            Picked point coordinates
        """
        self.point_picked.emit(np.asarray(point))

    def _get_marker_size(self):
        """
        Calculate appropriate marker size based on domain size.

        Returns
        -------
        float
            Marker radius
        """
        if self.domain is not None and hasattr(self.domain, 'characteristic_length'):
            return self.domain.characteristic_length * 0.05
        return 0.1

    def _get_arrow_length(self):
        """
        Calculate appropriate arrow length based on domain size.

        Returns
        -------
        float
            Arrow length
        """
        if self.domain is not None and hasattr(self.domain, 'characteristic_length'):
            return self.domain.characteristic_length * 0.2
        return 1.0
