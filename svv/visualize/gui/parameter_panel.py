"""
Parameter panel for configuring Tree and Forest generation.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QScrollArea, QFormLayout,
    QMessageBox, QProgressDialog
)
from PySide6.QtCore import Signal, Qt


class ParameterPanel(QWidget):
    """
    Widget for configuring Tree and Forest generation parameters.
    """

    # Signals
    parameters_changed = Signal()

    def __init__(self, parent=None):
        """
        Initialize the parameter panel.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget (should be VascularizeGUI)
        """
        super().__init__(parent)
        self.parent_gui = parent

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Mode selection
        mode_group = QGroupBox("Generation Mode")
        mode_layout = QVBoxLayout()
        mode_group.setLayout(mode_layout)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Single Tree")
        self.mode_combo.addItem("Forest (Multiple Trees)")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)

        scroll_layout.addWidget(mode_group)

        # Tree parameters
        tree_params_group = QGroupBox("Tree Parameters")
        tree_form = QFormLayout()
        tree_params_group.setLayout(tree_form)

        self.n_vessels_spin = QSpinBox()
        self.n_vessels_spin.setRange(1, 100000)
        self.n_vessels_spin.setValue(100)
        tree_form.addRow("Number of Vessels:", self.n_vessels_spin)

        self.physical_clearance_spin = QDoubleSpinBox()
        self.physical_clearance_spin.setRange(0.0, 10.0)
        self.physical_clearance_spin.setSingleStep(0.01)
        self.physical_clearance_spin.setDecimals(4)
        self.physical_clearance_spin.setValue(0.0)
        tree_form.addRow("Physical Clearance:", self.physical_clearance_spin)

        self.convexity_tolerance_spin = QDoubleSpinBox()
        self.convexity_tolerance_spin.setRange(0.0, 1.0)
        self.convexity_tolerance_spin.setSingleStep(0.01)
        self.convexity_tolerance_spin.setDecimals(3)
        self.convexity_tolerance_spin.setValue(0.01)
        tree_form.addRow("Convexity Tolerance:", self.convexity_tolerance_spin)

        scroll_layout.addWidget(tree_params_group)

        # Forest parameters (initially hidden)
        self.forest_params_group = QGroupBox("Forest Parameters")
        forest_form = QFormLayout()
        self.forest_params_group.setLayout(forest_form)

        self.compete_cb = QCheckBox("Enable Competition")
        forest_form.addRow("Competition:", self.compete_cb)

        self.decay_probability_spin = QDoubleSpinBox()
        self.decay_probability_spin.setRange(0.0, 1.0)
        self.decay_probability_spin.setSingleStep(0.05)
        self.decay_probability_spin.setDecimals(2)
        self.decay_probability_spin.setValue(0.9)
        forest_form.addRow("Decay Probability:", self.decay_probability_spin)

        scroll_layout.addWidget(self.forest_params_group)
        self.forest_params_group.hide()

        # Advanced parameters
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_form = QFormLayout()
        advanced_group.setLayout(advanced_form)

        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(-1, 999999)
        self.random_seed_spin.setValue(-1)
        self.random_seed_spin.setSpecialValueText("Random")
        advanced_form.addRow("Random Seed:", self.random_seed_spin)

        scroll_layout.addWidget(advanced_group)

        # Add stretch to push everything to top
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Action buttons
        button_layout = QVBoxLayout()

        self.generate_btn = QPushButton("Generate Tree/Forest")
        self.generate_btn.clicked.connect(self._generate)
        self.generate_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        button_layout.addWidget(self.generate_btn)

        self.export_btn = QPushButton("Export Configuration")
        self.export_btn.clicked.connect(self._export_config)
        button_layout.addWidget(self.export_btn)

        layout.addLayout(button_layout)

    def _on_mode_changed(self, index):
        """Handle mode selection change."""
        if index == 0:  # Single Tree
            self.forest_params_group.hide()
        else:  # Forest
            self.forest_params_group.show()

    def _generate(self):
        """Generate Tree or Forest based on current configuration."""
        if not self.parent_gui or not self.parent_gui.domain:
            QMessageBox.warning(
                self,
                "No Domain",
                "Please load a domain before generating trees."
            )
            return

        # Get configuration
        mode = self.mode_combo.currentIndex()
        n_vessels = self.n_vessels_spin.value()
        physical_clearance = self.physical_clearance_spin.value()
        convexity_tolerance = self.convexity_tolerance_spin.value()
        random_seed = self.random_seed_spin.value() if self.random_seed_spin.value() >= 0 else None

        # Get start points configuration
        point_config = self.parent_gui.point_selector.get_configuration()

        try:
            if mode == 0:  # Single Tree
                self._generate_tree(
                    n_vessels,
                    physical_clearance,
                    convexity_tolerance,
                    random_seed,
                    point_config
                )
            else:  # Forest
                compete = self.compete_cb.isChecked()
                decay_prob = self.decay_probability_spin.value()
                self._generate_forest(
                    n_vessels,
                    physical_clearance,
                    convexity_tolerance,
                    random_seed,
                    compete,
                    decay_prob,
                    point_config
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Generation Error",
                f"Failed to generate tree/forest:\n{str(e)}"
            )

    def _generate_tree(self, n_vessels, physical_clearance, convexity_tolerance, random_seed, config):
        """Generate a single tree."""
        from svv.tree.tree import Tree

        # Create progress dialog
        progress = QProgressDialog("Generating tree...", "Cancel", 0, n_vessels, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        # Create tree
        tree = Tree()
        tree.physical_clearance = physical_clearance
        if random_seed is not None:
            tree.random_seed = random_seed

        # Set domain
        tree.set_domain(self.parent_gui.domain, convexity_tolerance)

        # Get start point and direction
        start_point = None
        direction = None

        if config['start_points'][0][0] is not None:
            start_point = config['start_points'][0][0]
            if config['directions'][0][0] is not None:
                direction = config['directions'][0][0]

        # Set root
        if start_point is not None:
            if direction is not None:
                tree.set_root(start_point, direction)
            else:
                tree.set_root(start_point)
        else:
            tree.set_root()

        # Generate vessels
        for i in range(n_vessels):
            if progress.wasCanceled():
                break

            tree.add()
            progress.setValue(i + 1)

        progress.close()

        # Visualize
        self.parent_gui.vtk_widget.clear_trees()
        self.parent_gui.vtk_widget.add_tree(tree)

        # Store tree
        self.parent_gui.trees = [tree]

        QMessageBox.information(
            self,
            "Success",
            f"Tree generated with {tree.n_terminals} vessels."
        )

    def _generate_forest(self, n_vessels, physical_clearance, convexity_tolerance,
                        random_seed, compete, decay_prob, config):
        """Generate a forest."""
        from svv.forest.forest import Forest

        # Create progress dialog
        progress = QProgressDialog("Generating forest...", "Cancel", 0, n_vessels, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        # Extract forest configuration
        n_networks = config['n_networks']
        n_trees_per_network = config['n_trees_per_network']
        start_points = config['start_points']
        directions = config['directions']

        # Create forest
        forest = Forest(
            domain=self.parent_gui.domain,
            n_networks=n_networks,
            n_trees_per_network=n_trees_per_network,
            start_points=start_points,
            directions=directions,
            physical_clearance=physical_clearance,
            compete=compete
        )

        # Set domain
        forest.set_domain(self.parent_gui.domain, convexity_tolerance)

        # Set roots
        forest.set_roots(start_points, directions)

        # Generate vessels
        for i in range(n_vessels):
            if progress.wasCanceled():
                break

            forest.add(1, decay_probability=decay_prob)
            progress.setValue(i + 1)

        progress.close()

        # Visualize
        self.parent_gui.vtk_widget.clear_trees()
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        color_idx = 0

        for network in forest.networks:
            for tree in network:
                self.parent_gui.vtk_widget.add_tree(tree, color=colors[color_idx % len(colors)])
                color_idx += 1

        # Store forest
        self.parent_gui.forest = forest

        total_vessels = sum(tree.n_terminals for network in forest.networks for tree in network)
        QMessageBox.information(
            self,
            "Success",
            f"Forest generated with {total_vessels} total vessels across {n_networks} networks."
        )

    def _export_config(self):
        """Export the current configuration."""
        if self.parent_gui:
            self.parent_gui.save_configuration()

    def get_parameters(self):
        """
        Get the current parameter configuration.

        Returns
        -------
        dict
            Dictionary of parameters
        """
        return {
            'mode': 'tree' if self.mode_combo.currentIndex() == 0 else 'forest',
            'n_vessels': self.n_vessels_spin.value(),
            'physical_clearance': self.physical_clearance_spin.value(),
            'convexity_tolerance': self.convexity_tolerance_spin.value(),
            'random_seed': self.random_seed_spin.value() if self.random_seed_spin.value() >= 0 else None,
            'compete': self.compete_cb.isChecked(),
            'decay_probability': self.decay_probability_spin.value()
        }
