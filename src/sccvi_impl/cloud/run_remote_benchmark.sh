#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Activate conda environment (try different paths)
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
else
    echo "Warning: Could not find conda.sh in standard locations. Trying conda activate directly."
    # Try direct activation as a fallback
    if ! conda activate base 2>/dev/null; then
        echo "Error: Failed to activate conda. Please check your conda installation."
        exit 1
    fi
fi

# Check if conda environment exists
if ! conda env list | grep -q "sccvi"; then
    echo "Error: conda environment 'sccvi' not found"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Activate the environment
echo "Activating conda environment 'sccvi'..."
conda activate sccvi

# Go to project directory
cd "$PROJECT_ROOT"
echo "Current directory: $(pwd)"

# Check for requirements.txt
if [ ! -f requirements.txt ]; then
    echo "Error: requirements.txt not found in $(pwd)"
    exit 1
fi

# Install requirements (if not already installed)
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Run the benchmarking pipeline
echo "Running benchmarking pipeline..."
python -m src.sccvi_impl.benchmarking.model_benchmark

# Check if benchmarking was successful
if [ $? -eq 0 ]; then
    echo "Benchmarking complete. Results are in data/benchmarking_results/"
else
    echo "Error: Benchmarking failed. Check the logs for details."
    exit 1
fi
