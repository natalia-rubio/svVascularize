# svVascularize GUI

Interactive GUI for visualizing and manipulating Domain objects to configure Tree and Forest vascularization.

## Features

- **3D Domain Visualization**: View your domain mesh in an interactive 3D viewport
- **Interactive Point Selection**: Click on the domain surface to select start points for trees
- **Direction Control**: Optionally specify custom directions for tree growth
- **Multi-Network Support**: Configure multiple networks with multiple trees per network
- **Parameter Configuration**: Adjust tree/forest generation parameters
- **Real-time Visualization**: See generated trees/forests in 3D
- **Export Configuration**: Save your start points and parameters for later use

## Installation

The GUI requires PySide6 and PyVistaQt (with PyVista).

Recommended (conda):
```bash
conda install -c conda-forge pyside6 pyvista pyvistaqt
```

Alternative (pip):
```bash
pip install PySide6 pyvista PyVistaQt
```

Notes:
- On Linux, the GUI prefers software rendering (Mesa llvmpipe) to avoid GPU/driver issues. These settings are applied only on Linux and do not affect Windows or macOS. You can opt out with `SVV_GUI_GL_MODE=system`.
- On Windows and macOS, no Mesa-specific setup is required.

## Usage

### Quick Launch

Use the provided launcher script:

```bash
./launch_gui.sh
```

### Basic Usage (from Python)

```python
import os
import sys

from svv.visualize.gui import launch_gui

# Launch and block until closed
launch_gui()
```

### With Pre-loaded Domain

```python
from PySide6.QtWidgets import QApplication
from svv.visualize.gui import VascularizeGUI
from svv.domain.domain import Domain
import pyvista as pv
import sys

# Create or load a domain
sphere = pv.Sphere(radius=5.0)
domain = Domain(sphere)
domain.create()
domain.solve()
domain.build()

# Launch GUI with domain (blocking)
from svv.visualize.gui import launch_gui
launch_gui(domain=domain)
```

### Example Script

Run the included example:

```bash
python -m svv.visualize.gui.example_usage
```

### CLI Launch

```bash
python -m svv.visualize.gui
python -m svv.visualize.gui --domain path/to/domain.dmn
```

## Linux Troubleshooting

If you see libGL/Mesa warnings or a segmentation fault on launch, ensure your conda environment has Mesa userspace drivers and that the loader uses them instead of the system drivers.

1. Use conda-forge packages and strict priority:
   - `conda config --env --add channels conda-forge`
   - `conda config --env --set channel_priority strict`
2. Install GL stack (conda-forge):
   - `conda install -y mesalib libglvnd libxcb xorg-libx11 xorg-libxt xorg-libxfixes`
3. Point the driver loader to conda's DRI path and favor software GL:
   - `export SVV_LIBGL_DRIVERS_PATH="$CONDA_PREFIX/lib/dri"`
   - `export QT_OPENGL=software`
   - `export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe`
   - If on Wayland: `export QT_QPA_PLATFORM=xcb`
4. Launch again via Python or `python -m svv.visualize.gui`.

Opt out and try system GL:
- `export SVV_GUI_GL_MODE=system`  # disables Linux-specific software GL setup in the GUI

These settings help avoid mixing system Mesa with conda libraries, which can cause crashes.

## GUI Components

### Main Window

The main window is split into two panels:

1. **Left Panel (70%)**: 3D visualization using PyVista
2. **Right Panel (30%)**: Control widgets

### Control Widgets

#### Point Selector

- **Networks**: Set the number of vascular networks
- **Network/Tree Selection**: Choose which network and tree to add points to
- **Pick Point**: Click on the domain surface to add start points
- **Manual Input**: Enter point coordinates manually
- **Direction Control**: Optionally specify custom growth directions
- **Point List**: View and manage all added points

#### Parameter Panel

- **Generation Mode**: Choose between single tree or forest (multiple trees)
- **Tree Parameters**:
  - Number of vessels to generate
  - Physical clearance between vessels
  - Convexity tolerance
- **Forest Parameters** (when in forest mode):
  - Competition between trees
  - Decay probability
- **Advanced Parameters**:
  - Random seed for reproducibility

### Menu Bar

- **File**:
  - Load Domain: Load a .dmn domain file
  - Save Configuration: Export start points and parameters to JSON
  - Exit: Close the application
- **View**:
  - Reset Camera: Reset the 3D view
  - Toggle Domain Visibility: Show/hide the domain mesh
- **Help**:
  - About: Show information about the application

## Workflow

1. **Load or Create a Domain**:
   - Use `File -> Load Domain` to load a .dmn file, or
   - Pass a domain object when creating the GUI

2. **Configure Start Points**:
   - Select the network and tree index
   - Click "Pick Point" and click on the domain surface, or
   - Use "Manual Input" to enter coordinates
   - Optionally check "Use Custom Direction" and specify a direction vector

3. **Set Parameters**:
   - Choose generation mode (Tree or Forest)
   - Set number of vessels
   - Adjust physical clearance and other parameters

4. **Generate**:
   - Click "Generate Tree/Forest" to start generation
   - View progress in the dialog
   - See results in the 3D viewport

5. **Export** (Optional):
   - Use "File -> Save Configuration" to save your setup

## API Reference

### VascularizeGUI

Main GUI window class.

**Constructor:**
```python
VascularizeGUI(domain=None)
```

**Parameters:**
- `domain` (svv.domain.Domain, optional): Initial domain to visualize

**Methods:**
- `load_domain(domain)`: Load a domain object
- `update_status(message)`: Update the status bar

### VTKWidget

3D visualization widget.

**Methods:**
- `set_domain(domain)`: Set and visualize the domain
- `add_start_point(point, index, color)`: Add a point marker
- `add_direction(point, direction, length, color)`: Add a direction arrow
- `add_tree(tree, color)`: Visualize a tree
- `clear()`: Clear all visualizations except domain
- `reset_camera()`: Reset camera view

### PointSelectorWidget

Widget for managing start points and directions.

**Methods:**
- `set_domain(domain)`: Set the domain
- `get_configuration()`: Get current configuration as dictionary

### ParameterPanel

Widget for setting tree/forest parameters.

**Methods:**
- `get_parameters()`: Get current parameter values

## Notes

- The GUI uses PyVista's Qt integration for 3D rendering
- Point picking is done by clicking directly on the domain surface
- Directions are automatically normalized when using the "Normalize Direction" button
- Generated trees/forests are stored in `gui.trees` or `gui.forest` for programmatic access

## Troubleshooting

### ImportError: No module named 'PySide6'

Install the required dependencies:
```bash
pip install PySide6 pyvista PyVistaQt
```

### GUI not responding during generation

Large tree/forest generation may take time. A progress dialog is shown during generation. For very large generations (>10000 vessels), consider using the non-GUI API.

### Point picking not working

Ensure the domain has a valid boundary mesh. The domain must be created, solved, and built before point picking will work properly.
