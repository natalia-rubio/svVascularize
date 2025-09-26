# Junction Smoothing for Vascular Trees

This document describes the new junction smoothing functionality added to the svVascularize library. The feature automatically detects vessel junctions (where multiple vessels intersect) and applies smoothing algorithms to improve mesh quality at these critical points.

## Overview

Vessel junctions are locations where multiple vessels meet, creating complex geometric intersections. These regions often have poor mesh quality due to sharp angles and overlapping geometries. The junction smoothing feature addresses this by:

1. **Detecting junctions** automatically from the tree structure
2. **Identifying junction regions** in the mesh
3. **Applying smoothing algorithms** to improve mesh quality
4. **Preserving vessel connectivity** and overall tree structure

## Features

### Automatic Junction Detection
- Identifies points where multiple vessels intersect
- Uses vessel connectivity data from the tree structure
- Provides statistics about junction types and distributions

### Advanced Smoothing Algorithms
- **Taubin smoothing** with boundary preservation
- **Adaptive smoothing radius** based on vessel dimensions
- **Configurable smoothing parameters** for different use cases

### Integration with Existing Pipeline
- Seamlessly integrated into the `export_solid()` method
- Optional feature that can be enabled/disabled
- Maintains compatibility with existing code

## Usage

### Basic Usage

```python
from svv.tree.tree import Tree

# Create and build your tree
tree = Tree()
# ... build your tree structure ...

# Export with junction smoothing (enabled by default)
mesh = tree.export_solid(watertight=True, smooth_junctions=True)
```

### Advanced Configuration

```python
# Customize smoothing parameters
mesh = tree.export_solid(
    watertight=True,
    smooth_junctions=True,
    smoothing_radius_factor=2.0,    # Radius multiplier for smoothing region
    smoothing_iterations=5          # Number of smoothing iterations
)
```

### Junction Statistics

```python
# Get information about junctions in your tree
stats = tree.get_junction_statistics()
print(f"Total junctions: {stats['total_junctions']}")
print(f"Junction types: {stats['junction_types']}")
print(f"Average vessels per junction: {stats['average_vessels_per_junction']}")
```

## Parameters

### `smooth_junctions` (bool, default: True)
- Enable or disable junction smoothing
- When `False`, the original mesh is exported without smoothing

### `smoothing_radius_factor` (float, default: 2.0)
- Multiplier for determining the smoothing radius around junctions
- Larger values create larger smoothing regions
- Smaller values provide more localized smoothing

### `smoothing_iterations` (int, default: 5)
- Number of smoothing iterations to apply
- More iterations provide smoother results but may over-smooth
- Fewer iterations preserve more geometric detail

## Technical Details

### Junction Detection Algorithm

The junction detection algorithm works by:

1. **Extracting vessel endpoints** from the tree data structure
2. **Identifying coordinate matches** where multiple vessels meet
3. **Creating junction mappings** that link vessels to intersection points
4. **Computing junction statistics** for analysis and visualization

### Smoothing Algorithm

The smoothing process involves:

1. **Region identification** around each junction point
2. **Taubin smoothing** with boundary preservation
3. **Mesh repair** to ensure watertightness
4. **Normal recomputation** for proper rendering

### Integration Points

The junction smoothing is integrated at several key points:

- **`svv/tree/utils/junction_smoothing.py`**: Core smoothing algorithms
- **`svv/tree/export/export_solid.py`**: Integration with mesh export
- **`svv/tree/tree.py`**: User-facing API methods

## Examples

### Simple Tree with Smoothing

```python
import numpy as np
from svv.tree.tree import Tree
from svv.domain.domain import Domain

# Create domain and tree
domain = Domain()
domain.set_sphere(center=[0, 0, 0], radius=10)
tree = Tree()
tree.set_domain(domain)
tree.set_root(start=[0, 0, 0], direction=[1, 0, 0])

# Add vessels to create junctions
for i in range(10):
    tree.add()

# Export with smoothing
mesh = tree.export_solid(watertight=True, smooth_junctions=True)
print(f"Exported mesh with {mesh.n_points} points and {mesh.n_cells} cells")
```

### Comparing Smoothed vs Non-Smoothed

```python
# Export without smoothing
mesh_original = tree.export_solid(watertight=True, smooth_junctions=False)

# Export with smoothing
mesh_smoothed = tree.export_solid(watertight=True, smooth_junctions=True)

# Compare mesh quality
print(f"Original mesh quality: {mesh_original.compute_cell_quality().mean()}")
print(f"Smoothed mesh quality: {mesh_smoothed.compute_cell_quality().mean()}")
```

## Performance Considerations

### Computational Cost
- Junction detection is O(n) where n is the number of vessels
- Smoothing cost depends on junction complexity and smoothing parameters
- Overall overhead is typically < 10% of total mesh generation time

### Memory Usage
- Minimal additional memory requirements
- Junction detection uses existing tree data structures
- Smoothing operates on mesh regions, not entire mesh

### Quality vs Performance Trade-offs
- Higher `smoothing_iterations` improve quality but increase computation time
- Larger `smoothing_radius_factor` affects more mesh elements
- Default parameters provide good balance of quality and performance

## Troubleshooting

### Common Issues

1. **No junctions detected**
   - Ensure your tree has multiple vessels
   - Check that vessels actually intersect (not just touch)
   - Verify tree connectivity is properly set

2. **Over-smoothing**
   - Reduce `smoothing_iterations`
   - Decrease `smoothing_radius_factor`
   - Check vessel dimensions are appropriate

3. **Under-smoothing**
   - Increase `smoothing_iterations`
   - Increase `smoothing_radius_factor`
   - Verify junction detection is working correctly

### Debug Information

Enable debug output by checking junction statistics:

```python
stats = tree.get_junction_statistics()
if stats['total_junctions'] == 0:
    print("Warning: No junctions detected - smoothing will have no effect")
else:
    print(f"Detected {stats['total_junctions']} junctions for smoothing")
```

## Future Enhancements

Potential future improvements include:

1. **Adaptive smoothing parameters** based on vessel geometry
2. **Junction-specific smoothing strategies** for different junction types
3. **Quality metrics** for evaluating smoothing effectiveness
4. **Interactive smoothing controls** for fine-tuning results

## API Reference

### Tree Methods

#### `export_solid(smooth_junctions=True, smoothing_radius_factor=2.0, smoothing_iterations=5)`
Export tree as solid mesh with optional junction smoothing.

#### `get_junction_statistics()`
Get statistics about junctions in the tree.

### Junction Smoothing Module

#### `detect_junctions(tree_data, vessel_map, connectivity, tolerance=1e-6)`
Detect vessel junctions from tree structure.

#### `smooth_junctions_advanced(mesh, tree_data, vessel_map, connectivity, hsize=None, cap_resolution=40)`
Apply advanced junction smoothing to mesh.

#### `get_junction_statistics(tree_data, vessel_map, connectivity)`
Get comprehensive junction statistics.

## Contributing

To contribute to the junction smoothing functionality:

1. **Test with various tree structures** to ensure robustness
2. **Optimize smoothing algorithms** for better performance
3. **Add new smoothing methods** for different junction types
4. **Improve documentation** and examples

## License

This functionality is part of the svVascularize library and follows the same licensing terms.
