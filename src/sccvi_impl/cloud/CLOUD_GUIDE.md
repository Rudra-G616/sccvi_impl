# Cloud Guide for Running Benchmarks

This guide provides instructions for running the benchmarking scripts on a cloud GPU instance using `cloud_script.py`. The script handles uploading the code, setting up the environment, running benchmarks, and downloading results.

## Prerequisites

1. A cloud instance with:
   - SSH access
   - Python 3.x installed
   - GPU support (optional but recommended)
   - Sufficient disk space (~2GB)

2. SSH connection string in the format:
   ```bash
   ssh -o StrictHostKeyChecking=no -p <port> root@<host>
   ```

## Usage

### Basic Command

```bash
python -m sccvi_impl.cloud.cloud_script --ssh-string "ssh -o StrictHostKeyChecking=no -p <port> root@<host>"
```

### Command Line Arguments

- `--ssh-string` (required): SSH connection string for your cloud instance
- `--n_epochs` (optional): Number of training epochs (default: 100)
- `--batch_size` (optional): Batch size for training (default: 128)
- `--no-gpu` (optional): Disable GPU usage even if available

### Example Commands

Basic usage with default parameters:
```bash
python -m sccvi_impl.cloud.cloud_script \
  --ssh-string "ssh -o StrictHostKeyChecking=no -p 12014 root@ssha.jarvislabs.ai"
```

Custom training parameters:
```bash
python -m sccvi_impl.cloud.cloud_script \
  --ssh-string "ssh -o StrictHostKeyChecking=no -p 12014 root@ssha.jarvislabs.ai" \
  --n_epochs 200 \
  --batch_size 256
```

Force CPU usage:
```bash
python -m sccvi_impl.cloud.cloud_script \
  --ssh-string "ssh -o StrictHostKeyChecking=no -p 12014 root@ssha.jarvislabs.ai" \
  --no-gpu
```

## What the Script Does

The script performs the following steps automatically:

1. **Cleanup**: Removes any existing files from previous runs on the remote server
2. **Upload**: Transfers the entire project directory to the cloud instance
3. **Setup**: Installs the package and its dependencies in the cloud environment
4. **Benchmarking**: Runs the simulated benchmarking script with specified parameters
5. **Download**: Retrieves the results file from the cloud instance

## Results

After successful execution:

- Results are saved locally in `src/results/simulated/benchmarking_results.csv`
- The CSV file contains performance metrics for both Model1 and scCausalVI:
  - L2 norm of reconstruction
  - Covariance metrics for batch effects
  - Average Silhouette Width (ASW)
  - Treatment effect metrics


