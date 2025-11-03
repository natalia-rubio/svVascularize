"""
Widget for selecting and managing start points and directions.
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QListWidget, QListWidgetItem,
    QLabel, QLineEdit, QCheckBox, QSpinBox,
    QDoubleSpinBox, QComboBox
)
from PySide6.QtCore import Signal


class PointSelectorWidget(QWidget):
    """
    Widget for managing start points and directions for Tree/Forest generation.
    """

    # Signals
    points_changed = Signal()

    def __init__(self, parent=None):
        """
        Initialize the point selector widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget (should be VascularizeGUI)
        """
        super().__init__(parent)
        self.parent_gui = parent
        self.domain = None
        self.points = []  # List of dicts with 'point', 'direction', 'network', 'tree_index'

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Group box for point management
        group_box = QGroupBox("Start Points & Directions")
        group_layout = QVBoxLayout()
        group_box.setLayout(group_layout)

        # Network configuration
        network_layout = QHBoxLayout()
        network_layout.addWidget(QLabel("Networks:"))
        self.network_spin = QSpinBox()
        self.network_spin.setMinimum(1)
        self.network_spin.setMaximum(10)
        self.network_spin.setValue(1)
        self.network_spin.valueChanged.connect(self._on_network_changed)
        network_layout.addWidget(self.network_spin)
        network_layout.addStretch()
        group_layout.addLayout(network_layout)

        # Current network/tree selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Network:"))
        self.network_combo = QComboBox()
        self.network_combo.addItem("Network 0")
        selection_layout.addWidget(self.network_combo)

        selection_layout.addWidget(QLabel("Tree:"))
        self.tree_combo = QComboBox()
        self.tree_combo.addItem("Tree 0")
        self.tree_combo.addItem("Tree 1")
        selection_layout.addWidget(self.tree_combo)
        group_layout.addLayout(selection_layout)

        # Point list
        self.point_list = QListWidget()
        self.point_list.itemClicked.connect(self._on_point_selected)
        group_layout.addWidget(QLabel("Points:"))
        group_layout.addWidget(self.point_list)

        # Add point buttons
        button_layout = QHBoxLayout()

        self.pick_mode_btn = QPushButton("Pick Point (Click on Domain)")
        self.pick_mode_btn.setCheckable(True)
        self.pick_mode_btn.clicked.connect(self._toggle_pick_mode)
        button_layout.addWidget(self.pick_mode_btn)

        self.manual_btn = QPushButton("Manual Input...")
        self.manual_btn.clicked.connect(self._manual_input)
        button_layout.addWidget(self.manual_btn)

        group_layout.addLayout(button_layout)

        # Point details
        details_group = QGroupBox("Point Details")
        details_layout = QVBoxLayout()
        details_group.setLayout(details_layout)

        # Coordinates
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("Position:"))
        self.coord_label = QLabel("(0.0, 0.0, 0.0)")
        coord_layout.addWidget(self.coord_label)
        coord_layout.addStretch()
        details_layout.addLayout(coord_layout)

        # Direction checkbox
        self.use_direction_cb = QCheckBox("Use Custom Direction")
        self.use_direction_cb.stateChanged.connect(self._on_direction_toggled)
        details_layout.addWidget(self.use_direction_cb)

        # Direction inputs
        direction_layout = QHBoxLayout()
        direction_layout.addWidget(QLabel("Direction:"))
        self.dir_x = QDoubleSpinBox()
        self.dir_x.setRange(-1.0, 1.0)
        self.dir_x.setSingleStep(0.1)
        self.dir_x.setDecimals(3)
        self.dir_x.valueChanged.connect(self._on_direction_changed)
        direction_layout.addWidget(QLabel("X:"))
        direction_layout.addWidget(self.dir_x)

        self.dir_y = QDoubleSpinBox()
        self.dir_y.setRange(-1.0, 1.0)
        self.dir_y.setSingleStep(0.1)
        self.dir_y.setDecimals(3)
        self.dir_y.valueChanged.connect(self._on_direction_changed)
        direction_layout.addWidget(QLabel("Y:"))
        direction_layout.addWidget(self.dir_y)

        self.dir_z = QDoubleSpinBox()
        self.dir_z.setRange(-1.0, 1.0)
        self.dir_z.setSingleStep(0.1)
        self.dir_z.setDecimals(3)
        self.dir_z.valueChanged.connect(self._on_direction_changed)
        direction_layout.addWidget(QLabel("Z:"))
        direction_layout.addWidget(self.dir_z)

        details_layout.addLayout(direction_layout)

        # Normalize button
        self.normalize_btn = QPushButton("Normalize Direction")
        self.normalize_btn.clicked.connect(self._normalize_direction)
        details_layout.addWidget(self.normalize_btn)

        # Initially disable direction inputs
        self._enable_direction_inputs(False)

        group_layout.addWidget(details_group)

        # Delete and Clear buttons
        action_layout = QHBoxLayout()
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        action_layout.addWidget(self.delete_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_all)
        action_layout.addWidget(self.clear_btn)

        group_layout.addLayout(action_layout)

        layout.addWidget(group_box)

    def set_domain(self, domain):
        """
        Set the domain for point selection.

        Parameters
        ----------
        domain : svv.domain.Domain
            Domain object
        """
        self.domain = domain

    def _toggle_pick_mode(self, checked):
        """Toggle point picking mode."""
        if checked and self.parent_gui:
            self.pick_mode_btn.setText("Picking... (Click Domain)")
            # Connect to VTK widget's point picking signal
            self.parent_gui.vtk_widget.point_picked.connect(self._on_point_picked)
        else:
            self.pick_mode_btn.setText("Pick Point (Click on Domain)")
            if self.parent_gui:
                try:
                    self.parent_gui.vtk_widget.point_picked.disconnect(self._on_point_picked)
                except:
                    pass

    def _on_point_picked(self, point):
        """
        Handle picked point from VTK widget.

        Parameters
        ----------
        point : np.ndarray
            Picked point coordinates
        """
        network = self.network_combo.currentIndex()
        tree_idx = self.tree_combo.currentIndex()

        point_data = {
            'point': point,
            'direction': None,
            'network': network,
            'tree_index': tree_idx
        }

        self.points.append(point_data)
        self._update_point_list()
        self._visualize_point(point_data, len(self.points) - 1)

        # Update status
        if self.parent_gui:
            self.parent_gui.update_status(
                f"Added point at ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"
            )

        # Disable pick mode after picking
        self.pick_mode_btn.setChecked(False)
        self._toggle_pick_mode(False)

        self.points_changed.emit()

    def _manual_input(self):
        """Open dialog for manual point input."""
        from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Manual Point Input")
        layout = QFormLayout()

        x_input = QDoubleSpinBox()
        x_input.setRange(-1000, 1000)
        x_input.setDecimals(3)
        layout.addRow("X:", x_input)

        y_input = QDoubleSpinBox()
        y_input.setRange(-1000, 1000)
        y_input.setDecimals(3)
        layout.addRow("Y:", y_input)

        z_input = QDoubleSpinBox()
        z_input.setRange(-1000, 1000)
        z_input.setDecimals(3)
        layout.addRow("Z:", z_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        dialog.setLayout(layout)

        if dialog.exec_():
            point = np.array([x_input.value(), y_input.value(), z_input.value()])
            self._on_point_picked(point)

    def _on_point_selected(self, item):
        """Handle point selection from list."""
        idx = self.point_list.row(item)
        if idx < len(self.points):
            point_data = self.points[idx]
            point = point_data['point']
            direction = point_data['direction']

            self.coord_label.setText(f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")

            if direction is not None:
                self.use_direction_cb.setChecked(True)
                self.dir_x.setValue(direction[0])
                self.dir_y.setValue(direction[1])
                self.dir_z.setValue(direction[2])
            else:
                self.use_direction_cb.setChecked(False)

    def _on_direction_toggled(self, state):
        """Handle direction checkbox toggle."""
        enabled = state == 2  # Qt.Checked
        self._enable_direction_inputs(enabled)

        if enabled:
            # Set default direction if needed
            if self.dir_x.value() == 0 and self.dir_y.value() == 0 and self.dir_z.value() == 0:
                self.dir_z.setValue(1.0)

        self._update_selected_point_direction()

    def _on_direction_changed(self):
        """Handle direction value changes."""
        self._update_selected_point_direction()

    def _normalize_direction(self):
        """Normalize the current direction vector."""
        direction = np.array([self.dir_x.value(), self.dir_y.value(), self.dir_z.value()])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            self.dir_x.setValue(direction[0])
            self.dir_y.setValue(direction[1])
            self.dir_z.setValue(direction[2])

    def _update_selected_point_direction(self):
        """Update the direction of the currently selected point."""
        current_item = self.point_list.currentItem()
        if current_item is not None:
            idx = self.point_list.row(current_item)
            if idx < len(self.points):
                if self.use_direction_cb.isChecked():
                    direction = np.array([
                        self.dir_x.value(),
                        self.dir_y.value(),
                        self.dir_z.value()
                    ])
                    self.points[idx]['direction'] = direction
                    self._update_direction_visualization(idx)
                else:
                    self.points[idx]['direction'] = None
                    self._clear_direction_visualization(idx)

                self.points_changed.emit()

    def _enable_direction_inputs(self, enabled):
        """Enable or disable direction input fields."""
        self.dir_x.setEnabled(enabled)
        self.dir_y.setEnabled(enabled)
        self.dir_z.setEnabled(enabled)
        self.normalize_btn.setEnabled(enabled)

    def _delete_selected(self):
        """Delete the selected point."""
        current_item = self.point_list.currentItem()
        if current_item is not None:
            idx = self.point_list.row(current_item)
            if idx < len(self.points):
                self.points.pop(idx)
                self._update_point_list()
                self._refresh_visualization()
                self.points_changed.emit()

    def _clear_all(self):
        """Clear all points."""
        self.points.clear()
        self._update_point_list()
        if self.parent_gui:
            self.parent_gui.vtk_widget.clear()
        self.points_changed.emit()

    def _on_network_changed(self, value):
        """Handle network count change."""
        # Update network combo box
        self.network_combo.clear()
        for i in range(value):
            self.network_combo.addItem(f"Network {i}")

    def _update_point_list(self):
        """Update the point list widget."""
        self.point_list.clear()
        for i, point_data in enumerate(self.points):
            point = point_data['point']
            network = point_data['network']
            tree_idx = point_data['tree_index']
            has_dir = point_data['direction'] is not None

            label = f"P{i}: N{network}T{tree_idx} ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
            if has_dir:
                label += " [Dir]"

            self.point_list.addItem(label)

    def _visualize_point(self, point_data, index):
        """Visualize a single point."""
        if self.parent_gui:
            vtk_widget = self.parent_gui.vtk_widget
            point = point_data['point']
            direction = point_data['direction']

            vtk_widget.add_start_point(point, index=index)

            if direction is not None:
                vtk_widget.add_direction(point, direction)

    def _update_direction_visualization(self, index):
        """Update direction visualization for a point."""
        if self.parent_gui and index < len(self.points):
            # For simplicity, refresh all visualizations
            self._refresh_visualization()

    def _clear_direction_visualization(self, index):
        """Clear direction visualization for a point."""
        if self.parent_gui:
            # For simplicity, refresh all visualizations
            self._refresh_visualization()

    def _refresh_visualization(self):
        """Refresh all point and direction visualizations."""
        if self.parent_gui:
            vtk_widget = self.parent_gui.vtk_widget
            vtk_widget.clear()

            for i, point_data in enumerate(self.points):
                self._visualize_point(point_data, i)

    def get_configuration(self):
        """
        Get the current configuration as a dictionary.

        Returns
        -------
        dict
            Configuration with start_points and directions for Forest
        """
        n_networks = self.network_spin.value()

        # Organize points by network and tree
        config = {
            'n_networks': n_networks,
            'n_trees_per_network': [2] * n_networks,  # Default to 2 trees per network
            'start_points': [[None, None] for _ in range(n_networks)],
            'directions': [[None, None] for _ in range(n_networks)]
        }

        for point_data in self.points:
            network = point_data['network']
            tree_idx = point_data['tree_index']
            point = point_data['point']
            direction = point_data['direction']

            if network < n_networks and tree_idx < 2:
                config['start_points'][network][tree_idx] = point.tolist()
                if direction is not None:
                    config['directions'][network][tree_idx] = direction.tolist()

        return config
