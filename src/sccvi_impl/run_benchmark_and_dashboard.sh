#!/bin/bash
# This script runs the benchmarking pipeline and launches the Streamlit dashboard.
# Usage: bash run_benchmark_and_dashboard.sh [results_dir] [port] [--browser]
#   results_dir: Optional path to store benchmark results (default: ./benchmark_results)
#   port: Optional port for Streamlit dashboard (default: 8501)
#   --browser: Optional flag to open dashboard in browser

set -e

# Function for error handling
handle_error() {
    echo "Error: $1"
    exit 1
}

# Set up trap to handle interruptions
trap 'echo "Script interrupted"; exit 1' INT TERM

# Check if Python is installed
if ! command -v python &> /dev/null; then
    handle_error "Python is not installed or not in PATH"
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go to project root (two directories up from script)
cd "$SCRIPT_DIR/../.." || handle_error "Failed to navigate to project root"
echo "Working directory: $(pwd)"

# Print usage if --help or -h is provided
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: bash run_benchmark_and_dashboard.sh [results_dir] [port] [--browser] [--skip-benchmark]"
    echo "  results_dir: Optional path to store benchmark results (default: ./benchmark_results)"
    echo "  port: Optional port for Streamlit dashboard (default: 8501)"
    echo "  --browser: Optional flag to open dashboard in browser"
    echo "  --skip-benchmark: Optional flag to skip benchmarking and only launch dashboard"
    exit 0
fi

# Parse command line arguments
RESULTS_DIR="${1:-./benchmark_results}"
PORT="${2:-8501}"
BROWSER_FLAG=""
SKIP_BENCHMARK=""

# Process flags
for arg in "$@"; do
    if [[ "$arg" == "--browser" ]]; then
        BROWSER_FLAG="--browser"
    elif [[ "$arg" == "--skip-benchmark" ]]; then
        SKIP_BENCHMARK="true"
    fi
done

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR" || handle_error "Failed to create results directory: $RESULTS_DIR"

# Check for required Python packages
required_packages=("streamlit" "pandas" "numpy" "torch" "plotly" "sklearn" "anndata" "scanpy")
missing_packages=()

for package in "${required_packages[@]}"; do
    if ! python -c "import $package" &> /dev/null; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo "Installing missing required packages: ${missing_packages[*]}"
    pip install "${missing_packages[@]}" || handle_error "Failed to install required packages"
fi

# Run benchmarking pipeline if not skipped
if [[ -z "$SKIP_BENCHMARK" ]]; then
    echo "Running benchmarking pipeline..."
    # Set environment variable for results directory
    export BENCHMARK_RESULTS_DIR="$RESULTS_DIR"
    python ./src/sccvi_impl/benchmarking/model_benchmark.py --output_dir "$RESULTS_DIR" || handle_error "Benchmarking failed"
else
    echo "Skipping benchmarking as requested..."
fi

# Check if dashboard script exists
DASHBOARD_SCRIPT="./src/sccvi_impl/benchmarking/run_dashboard.py"
if [ ! -f "$DASHBOARD_SCRIPT" ]; then
    handle_error "Dashboard script not found at $DASHBOARD_SCRIPT"
fi

# Check if results were generated
BENCHMARK_RESULTS="$RESULTS_DIR/benchmarking_results"
if [ ! -f "$BENCHMARK_RESULTS/all_benchmarks.json" ] && [ ! -f "$BENCHMARK_RESULTS/all_benchmarks.csv" ]; then
    echo "Warning: No benchmark results found in $BENCHMARK_RESULTS"
    # Continue anyway, the dashboard will show an appropriate message
fi

# Launch Streamlit dashboard
echo "Launching Streamlit dashboard..."
# Pass the results directory and port to the dashboard script
python "$DASHBOARD_SCRIPT" --results-dir "$BENCHMARK_RESULTS" --port "$PORT" $BROWSER_FLAG --force || handle_error "Failed to launch dashboard"

echo "Benchmark and dashboard execution completed successfully."
