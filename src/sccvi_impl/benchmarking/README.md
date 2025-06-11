# scCausalVI vs Model1 Benchmarking

This module provides tools to benchmark `scCausalVI` and `Model1` on various single-cell datasets and visualize the results through a Streamlit dashboard.

## Features

- Download and prepare single-cell datasets for causal inference evaluation
- Train both `scCausalVI` and `Model1` on each dataset with configurable parameters
- Calculate and compare key metrics:
  - Correlation: E[z_bg*e_tilda] - E[z_bg]*E[e_tilda]
  - Average silhouette score on z_bg
  - L2 norm reconstruction error
- Visualize results through an interactive Streamlit dashboard with comparative charts
- Support for both local and cloud-based benchmarking

## Installation

Ensure you have all required dependencies:

```bash
pip install -r requirements.txt
```

For GPU acceleration (recommended), ensure you have the appropriate CUDA drivers installed.

## Usage

### Running the Benchmarking Pipeline

To run the benchmarking on all datasets:

```bash
python -m sccvi_impl.benchmarking.run_benchmark
```

To benchmark a specific dataset:

```bash
python -m sccvi_impl.benchmarking.run_benchmark --dataset simulated
```

Additional options:
```bash
python -m sccvi_impl.benchmarking.run_benchmark --help
```

Command-line arguments:
```
--output_dir OUTPUT_DIR   Directory to save results (default: data)
--dataset DATASET         Dataset to benchmark (default: all)
--max_epochs MAX_EPOCHS   Maximum training epochs (default: 50)
--batch_size BATCH_SIZE   Training batch size (default: 256)
--learning_rate LR        Learning rate (default: 0.001)
--use_gpu                 Use GPU for training if available
```

### Running the Dashboard

Once you have generated benchmark results, you can view them in the Streamlit dashboard:

```bash
python -m sccvi_impl.benchmarking.run_dashboard
```

The dashboard will be available at http://localhost:8501 by default.

You can customize the dashboard with these options:
```
--port PORT               Port to run the dashboard on (default: 8501)
--browser                 Open dashboard in browser automatically
```

## Metrics Explanation

1. **Correlation between z_bg and e_tilda**
   - Measures correlation/dependence between background and treatment effect latent variables
   - Computed as: E[z_bg*e_tilda] - E[z_bg]*E[e_tilda] (covariance)
   - Values closer to zero indicate better disentanglement (independence)
   - This metric evaluates how well the model separates background and treatment effects

2. **Average Silhouette Score on z_bg**
   - Measures how well cells from different conditions are separated in the background latent space
   - Range: [-1, 1], higher values indicate better condition-invariant representation
   - Good background representations should show similar values regardless of treatment condition

3. **L2 Reconstruction Error**
   - Measures how well the model reconstructs the original data
   - Computed as the mean L2 norm of the difference between original and reconstructed data
   - Lower values indicate better reconstruction ability

## Dataset Information

The benchmarking uses the following datasets:
- **simulated**: Simulated dataset with known ground truth effects
- **ifn_beta**: Interferon beta treatment on immune cells
- **covid_epithelial**: COVID-19 epithelial cells with infected and uninfected conditions
- **covid_pbmc**: COVID-19 peripheral blood mononuclear cells (PBMCs) from patients and controls
- **pbmc_batch_effect**: PBMCs with batch effects to test batch correction capabilities
- **pbmc_negative_control**: PBMCs with negative control for method validation

## Model Comparison

The benchmarking compares two models:

1. **scCausalVI**: A variational autoencoder designed specifically for causal inference in single-cell data
   - Disentangles background cell state from treatment effects
   - Uses a specialized latent space structure
   - Implements MMD regularization for proper disentanglement

2. **Model1**: An alternative implementation with different architectural choices
   - Uses a mutual information neural estimator approach
   - Implements different regularization strategies
   - Provides a comparative baseline for evaluation

## Visualization Features

The dashboard provides:
- Side-by-side model comparison across all datasets
- Radar charts for multi-metric evaluation
- Detailed performance breakdowns by dataset
- Interactive filtering and selection options
