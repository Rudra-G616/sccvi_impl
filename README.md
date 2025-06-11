# Benchmarking custom model against scCausalVI model

This repository contains an implementation of scCausalVI and a comparative model (Model1) for single-cell causal inference. It includes a benchmarking pipeline to evaluate both models across multiple datasets and visualize their performance.

## Overview

scCausalVI is a variational inference model designed for causal inference in single-cell RNA sequencing data. This implementation:

- Provides two model implementations: scCausalVI and Model1
- Includes benchmarking tools to compare model performance
- Streamlit dashboard for visualization
- Cloud-based benchmarking for GPU acceleration

## Features

- Train and evaluate causal inference models on single-cell data
- Compare disentanglement of background factors and treatment effects
- Evaluate reconstruction quality and latent space structure
- Visualize performance metrics through an interactive dashboard
- Run benchmarks on local or cloud GPU infrastructure

## Installation

```bash
# Clone the repository
git clone https://github.com/Rudra-G616/sccvi_impl.git
cd sccvi_impl

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the complete benchmarking pipeline (benchmark all datasets and launch dashboard):

```bash
python run_benchmarking_pipeline.py
```

Run benchmarking for a specific dataset:

```bash
python run_benchmarking_pipeline.py --run_benchmark --dataset simulated
```

Launch only the dashboard:

```bash
python run_benchmarking_pipeline.py --run_dashboard
```

## Usage Options

```
usage: run_benchmarking_pipeline.py [-h] [--data_dir DATA_DIR] [--install_requirements]
                                    [--run_benchmark] [--dataset DATASET]
                                    [--max_epochs MAX_EPOCHS] [--run_dashboard]
                                    [--port PORT] [--browser]

Run the benchmarking pipeline

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory to store datasets and results
  --install_requirements
                        Install required packages before running
  --run_benchmark       Run the benchmarking process
  --dataset DATASET     Specific dataset to benchmark (default: all)
  --max_epochs MAX_EPOCHS
                        Maximum number of training epochs
  --run_dashboard       Launch the Streamlit dashboard
  --port PORT           Port to run the dashboard on
  --browser             Open dashboard in browser
```

## Documentation

For more detailed information, see:
- [Benchmarking Documentation](src/sccvi_impl/benchmarking/README.md)
- [Cloud Deployment Guide](src/sccvi_impl/cloud/README_cloud.md)

## Requirements

See `requirements.txt` for the complete list of dependencies.