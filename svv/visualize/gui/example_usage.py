"""
Example usage of the svVascularize GUI.

This script demonstrates how to launch the GUI with and without a domain.
"""
import sys
from PySide6.QtWidgets import QApplication
from svv.visualize.gui import VascularizeGUI


def example_empty():
    """Launch GUI without a domain (domain can be loaded from menu)."""
    app = QApplication(sys.argv)
    gui = VascularizeGUI()
    gui.show()
    try:
        sys.exit(app.exec())
    except AttributeError:
        sys.exit(app.exec_())


def example_with_domain():
    """Launch GUI with a pre-loaded domain."""
    import pyvista as pv
    from svv.domain.domain import Domain

    # Create a simple cube domain
    cube_mesh = pv.Cube()
    domain = Domain(cube_mesh)
    domain.create()
    domain.solve()
    domain.build()

    # Launch GUI with domain
    app = QApplication(sys.argv)
    gui = VascularizeGUI(domain=domain)
    gui.show()
    try:
        sys.exit(app.exec())
    except AttributeError:
        sys.exit(app.exec_())


def example_with_sphere():
    """Launch GUI with a sphere domain."""
    import pyvista as pv
    from svv.domain.domain import Domain

    # Create a sphere domain
    sphere = pv.Sphere(radius=5.0, theta_resolution=30, phi_resolution=30)
    domain = Domain(sphere)
    domain.create()
    domain.solve()
    domain.build()

    # Launch GUI with domain
    app = QApplication(sys.argv)
    gui = VascularizeGUI(domain=domain)
    gui.show()
    try:
        sys.exit(app.exec())
    except AttributeError:
        sys.exit(app.exec_())


def example_load_from_file(dmn_file):
    """Launch GUI with a domain loaded from a .dmn file."""
    from svv.domain.domain import Domain

    # Load domain from file
    domain = Domain.load(dmn_file)

    # Launch GUI with domain
    app = QApplication(sys.argv)
    gui = VascularizeGUI(domain=domain)
    gui.show()
    try:
        sys.exit(app.exec())
    except AttributeError:
        sys.exit(app.exec_())


if __name__ == "__main__":
    # Choose which example to run:

    # 1. Empty GUI (load domain from File menu)
    # example_empty()

    # 2. GUI with a cube domain
    example_with_domain()

    # 3. GUI with a sphere domain
    # example_with_sphere()

    # 4. GUI with domain loaded from file
    # example_load_from_file("path/to/your/domain.dmn")
