"""
Main GUI window for svVascularize using PySide6.
"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QStatusBar, QFileDialog,
    QMessageBox
)
from PySide6.QtCore import Qt
from svv.visualize.gui.vtk_widget import VTKWidget
from svv.visualize.gui.point_selector import PointSelectorWidget
from svv.visualize.gui.parameter_panel import ParameterPanel


class VascularizeGUI(QMainWindow):
    """
    Main GUI window for visualizing and manipulating Domain objects
    to configure Tree and Forest vascularization.
    """

    def __init__(self, domain=None):
        """
        Initialize the main GUI window.

        Parameters
        ----------
        domain : svv.domain.Domain, optional
            Initial domain object to visualize
        """
        super().__init__()
        self.domain = domain
        self.trees = []
        self.forest = None

        self.setWindowTitle("svVascularize - Domain Visualization")
        self.setGeometry(100, 100, 1400, 900)

        self._init_ui()
        self._create_menu_bar()
        self._create_status_bar()

        # Load domain if provided
        if domain is not None:
            self.load_domain(domain)

    def _init_ui(self):
        """Initialize the user interface layout."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: 3D visualization
        self.vtk_widget = VTKWidget(self)
        splitter.addWidget(self.vtk_widget)

        # Right panel: Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # Point selector widget
        self.point_selector = PointSelectorWidget(self)
        right_layout.addWidget(self.point_selector)

        # Parameter panel
        self.parameter_panel = ParameterPanel(self)
        right_layout.addWidget(self.parameter_panel)

        splitter.addWidget(right_panel)

        # Set splitter proportions (70% visualization, 30% controls)
        splitter.setSizes([1000, 400])

        main_layout.addWidget(splitter)

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_domain_action = file_menu.addAction("Load Domain...")
        load_domain_action.triggered.connect(self.load_domain_dialog)

        save_config_action = file_menu.addAction("Save Configuration...")
        save_config_action.triggered.connect(self.save_configuration)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

        # View menu
        view_menu = menubar.addMenu("&View")

        reset_camera_action = view_menu.addAction("Reset Camera")
        reset_camera_action.triggered.connect(self.vtk_widget.reset_camera)

        toggle_domain_action = view_menu.addAction("Toggle Domain Visibility")
        toggle_domain_action.triggered.connect(self.vtk_widget.toggle_domain_visibility)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def load_domain(self, domain):
        """
        Load a domain object into the visualization.

        Parameters
        ----------
        domain : svv.domain.Domain
            Domain object to visualize
        """
        self.domain = domain
        self.vtk_widget.set_domain(domain)
        self.point_selector.set_domain(domain)
        self.status_bar.showMessage(f"Domain loaded successfully")

    def load_domain_dialog(self):
        """Open a file dialog to load a domain from a .dmn file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Domain",
            "",
            "Domain Files (*.dmn);;All Files (*)"
        )

        if file_path:
            try:
                from svv.domain.domain import Domain
                domain = Domain.load(file_path)

                # If boundary wasn't saved in the .dmn file, rebuild it
                if domain.boundary is None and hasattr(domain, 'patches') and len(domain.patches) > 0:
                    self.status_bar.showMessage("Building domain boundary...")
                    domain.build()

                self.load_domain(domain)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Domain",
                    f"Failed to load domain:\n{str(e)}"
                )

    def save_configuration(self):
        """Save the current start points and configuration."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                import json
                config = self.point_selector.get_configuration()
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.status_bar.showMessage(f"Configuration saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Saving Configuration",
                    f"Failed to save configuration:\n{str(e)}"
                )

    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About svVascularize",
            "svVascularize Domain Visualization Tool\n\n"
            "Interactive GUI for configuring vascular tree generation.\n\n"
            "Built with PySide6 and PyVista"
        )

    def update_status(self, message):
        """
        Update the status bar message.

        Parameters
        ----------
        message : str
            Status message to display
        """
        self.status_bar.showMessage(message)
