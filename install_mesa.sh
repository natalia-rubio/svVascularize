#!/bin/bash
# Install Mesa drivers for software rendering in conda environment

echo "Installing Mesa drivers for software rendering..."

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please ensure conda is installed and activated."
    exit 1
fi

if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No conda environment activated. Please activate a conda environment first."
    exit 1
fi

echo "Installing Mesa packages in conda environment: $CONDA_PREFIX"

# Install Mesa packages (double-check libffi and mesa compatibility)
conda install -y -c conda-forge \
    mesa-libgl-cos7-x86_64 \
    mesa-dri-drivers-cos7-x86_64 \
    mesalib \
    libglu \
    libffi

# Alternative: Install newer mesa packages if available
conda install -y -c conda-forge mesa-libgl-devel-cos7-x86_64 2>/dev/null || true

echo ""
echo "Mesa installation complete. Checking for drivers..."

# Check if drivers are installed
DRI_LOCATIONS=(
    "$CONDA_PREFIX/lib/dri"
    "$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64/dri"
    "$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib/dri"
)

FOUND=0
for dir in "${DRI_LOCATIONS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Found DRI directory: $dir"
        ls -la "$dir"/*swrast* "$dir"/*llvmpipe* 2>/dev/null && FOUND=1
    fi
done

if [ "$FOUND" -eq 1 ]; then
    echo ""
    echo "Success! Mesa drivers installed successfully."
    echo "You can now run: ./launch_gui.sh"
else
    echo ""
    echo "Warning: Could not verify Mesa driver installation."
    echo "Try running: ./launch_gui.sh --debug-gl for more information"
fi