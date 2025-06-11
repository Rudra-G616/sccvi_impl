# Cloud Benchmarking Workflow

This module contains scripts to help you offload model benchmarking to a Google Cloud GPU VM for faster training and evaluation.

## Overview

Running benchmarks on single-cell data can be computationally intensive, especially with multiple datasets and model configurations. This cloud workflow allows you to:
- Run benchmarks on powerful GPU-accelerated cloud instances
- Synchronize your code to the cloud instance
- Execute benchmarking in an optimized environment
- Retrieve and analyze results locally

## Files

- `run_remote_benchmark.sh`: Execute this on your VM to set up the environment and run the benchmarking pipeline
- `submit_to_gcloud.sh`: Run this locally to sync your code, trigger the remote script, and fetch results

## Detailed Setup

### 1. Google Cloud VM Setup

1. Create a GPU-enabled VM instance in Google Cloud:
   ```bash
   gcloud compute instances create sccvi-benchmark \
     --machine-type=n1-standard-8 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=50GB \
     --boot-disk-type=pd-ssd \
     --zone=us-central1-a
   ```

2. Install NVIDIA drivers and CUDA:
   ```bash
   # SSH into your VM
   gcloud compute ssh sccvi-benchmark --zone=us-central1-a

   # Install drivers (on the VM)
   sudo apt-get update
   sudo apt-get install -y build-essential
   curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
   sudo apt-get update
   sudo apt-get install -y cuda-11-8
   ```

3. Install Miniconda and create environment:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b
   ~/miniconda3/bin/conda init bash
   source ~/.bashrc
   conda create -n sccvi python=3.9
   conda activate sccvi
   ```

### 2. Usage

1. **Configure the scripts** if your setup differs from default:
   - Edit `run_remote_benchmark.sh` to match your conda environment name or paths
   - Edit `submit_to_gcloud.sh` if you need to customize synchronization options

2. **From your local machine**, run:
   ```bash
   bash src/sccvi_impl/cloud/submit_to_gcloud.sh <VM_NAME> <ZONE> [OPTIONS]
   ```
   - Replace `<VM_NAME>` with your VM instance name (e.g., sccvi-benchmark)
   - Replace `<ZONE>` with your VM's zone (e.g., us-central1-a)
   - Optional: Add additional arguments to pass to the benchmarking script

   Example:
   ```bash
   bash src/sccvi_impl/cloud/submit_to_gcloud.sh sccvi-benchmark us-central1-a --dataset simulated --max_epochs 100
   ```

3. **Monitor progress**:
   - The script will show real-time output from the cloud VM
   - You can also SSH into the VM to check progress:
     ```bash
     gcloud compute ssh <VM_NAME> --zone=<ZONE>
     tail -f /tmp/benchmark_log.txt
     ```

4. **Retrieve results**:
   - Results will be automatically downloaded to your local `data/benchmarking_results/` directory
   - You can run the dashboard locally to visualize the cloud-generated results:
     ```bash
     python -m src.sccvi_impl.benchmarking.run_dashboard
     ```

## Requirements

- **Local Machine**:
  - Google Cloud SDK (`gcloud`) installed and authenticated
  - SSH access configured for Google Cloud
  - Git repository with your project code

- **Cloud VM**:
  - Ubuntu 20.04 LTS or similar Linux distribution
  - CUDA 11.x with compatible GPU drivers
  - Anaconda/Miniconda installation
  - A conda environment named `sccvi` (or customize the script)
  - Sufficient disk space for datasets and results (~20GB recommended)

## Troubleshooting

- **Permission denied errors**: Ensure you have proper permissions to execute the scripts:
  ```bash
  chmod +x src/sccvi_impl/cloud/run_remote_benchmark.sh
  chmod +x src/sccvi_impl/cloud/submit_to_gcloud.sh
  ```

- **CUDA errors**: Check if your PyTorch version is compatible with the installed CUDA version:
  ```bash
  # On the VM
  nvidia-smi  # Check CUDA version
  python -c "import torch; print(torch.version.cuda)"  # Check PyTorch CUDA version
  ```

- **Sync issues**: If code synchronization fails, try using alternative methods:
  ```bash
  # Manual sync example
  gcloud compute scp --recurse ./sccvi_impl <VM_NAME>:~/ --zone=<ZONE>
  ```
