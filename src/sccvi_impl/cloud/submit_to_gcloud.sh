#!/bin/bash
# Usage: bash submit_to_gcloud.sh <VM_NAME> <ZONE>
# Example: bash submit_to_gcloud.sh my-vm-name us-central1-a

set -e

# Check for gcloud
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed or not in PATH"
    echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get script directory for more reliable paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

VM_NAME=$1
ZONE=$2

if [ -z "$VM_NAME" ] || [ -z "$ZONE" ]; then
  echo "Usage: bash submit_to_gcloud.sh <VM_NAME> <ZONE>"
  exit 1
fi

# Get the remote username by running whoami on the VM
echo "Checking remote username..."
REMOTE_USER=$(gcloud compute ssh "$VM_NAME" --zone "$ZONE" --command "whoami" 2>/dev/null || echo "$USER")
REMOTE_HOME="/home/$REMOTE_USER"  # More reliable than tilde
echo "Remote user: $REMOTE_USER"
echo "Remote home: $REMOTE_HOME"
PROJECT_DIR="sccvi_impl"
CLOUD_DIR="src/sccvi_impl/cloud"
RESULTS_DIR="data/benchmarking_results"

echo "[1/3] Syncing code to VM..."
cd "$PROJECT_ROOT"
if ! gcloud compute scp --recurse . "$VM_NAME":"$REMOTE_HOME/$PROJECT_DIR" --zone "$ZONE"; then
  echo "Error: Failed to sync code to VM"
  exit 1
fi

# Make the remote script executable
echo "[2/3] Running remote benchmarking script..."
# Check if conda is installed on the remote machine
if ! gcloud compute ssh "$VM_NAME" --zone "$ZONE" --command "command -v conda &>/dev/null"; then
  echo "Warning: conda is not installed on the remote machine. The benchmark may fail."
fi

# Check if sccvi conda environment exists on the remote machine
if ! gcloud compute ssh "$VM_NAME" --zone "$ZONE" --command "conda env list | grep -q sccvi"; then
  echo "Warning: sccvi conda environment not found on remote machine. Will attempt to create it."
fi

if ! gcloud compute ssh "$VM_NAME" --zone "$ZONE" --command "cd $REMOTE_HOME/$PROJECT_DIR/$CLOUD_DIR && chmod +x run_remote_benchmark.sh && ./run_remote_benchmark.sh"; then
  echo "Error: Failed to run remote benchmark script"
  exit 1
fi

echo "[3/3] Fetching results back to local machine..."
mkdir -p "$PROJECT_ROOT/$RESULTS_DIR"
# Check if remote directory exists and has files before trying to copy
if ! gcloud compute ssh "$VM_NAME" --zone "$ZONE" --command "test -d $REMOTE_HOME/$PROJECT_DIR/$RESULTS_DIR && test -n \"\$(ls -A $REMOTE_HOME/$PROJECT_DIR/$RESULTS_DIR 2>/dev/null)\""; then
  echo "Warning: Remote results directory doesn't exist or is empty. No results to fetch."
else
  # Using rsync-like approach to avoid wildcard expansion issues
  if ! gcloud compute scp --recurse "$VM_NAME":"$REMOTE_HOME/$PROJECT_DIR/$RESULTS_DIR/" "$PROJECT_ROOT/$RESULTS_DIR" --zone "$ZONE"; then
    echo "Error: Failed to fetch results from VM"
    exit 1
  fi
fi

echo "All done! Results are in $PROJECT_ROOT/$RESULTS_DIR/"
echo "To view the dashboard, run: cd $PROJECT_ROOT && bash src/sccvi_impl/run_benchmark_and_dashboard.sh"
