# scCausalVI vs Model1 Benchmarking

This module provides tools to benchmark `scCausalVI` and `Model1` on various single-cell datasets and visualize the results through a Streamlit dashboard.

## Features

- Download and prepare single-cell datasets for causal inference evaluation
- Train both `scCausalVI` and `Model1` on each dataset with configurable parameters
- Calculate and compare key metrics:
  - Correlation: $E[z_{bg} \times \tilde{e}] - (E[z_{bg}] \times E[\tilde{e}])$
  - Average silhouette score on $z_{bg}$
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
--n_latent N_LATENT       Dimension of latent space (default: 15)
--max_epochs MAX_EPOCHS   Maximum training epochs (default: 100)
--batch_size BATCH_SIZE   Training batch size (default: 128)
--use-gpu                 Use GPU for training if available
```

Note: The code now supports GPU acceleration through the `--use-gpu` command-line argument. When enabled, training will use GPU if available.

### Running the Dashboard

Once you have generated benchmark results, you can view them in the Streamlit dashboard:

```bash
python -m sccvi_impl.benchmarking.run_dashboard
```

The dashboard will be available at http://localhost:8501 by default.

You can customize the dashboard with these options:
```
--results-dir RESULTS_DIR  Directory containing benchmark results (default: data/benchmarking_results)
--port PORT                Port to run the dashboard on (default: 8501)
--browser                  Open dashboard in browser automatically
--force                    Start the dashboard even if no results exist
```

## Metrics Explanation

1. **Correlation between $z_{bg}$ and $\tilde{e}$**
   - Measures correlation/dependence between background and treatment effect latent variables
   - Computed as: $E[z_{bg}\tilde{e}] - E[z_{bg}]E[\tilde{e}]$ (covariance)
   - Values closer to zero indicate better disentanglement (orthogonality)
   - This metric evaluates how well the model separates background and treatment effects

2. **Average Silhouette Score on $z_{bg}$**
   - Measures how well cells from different conditions are separated in the background latent space
   - Good background representations should show similar values regardless of treatment condition (i.e. SUTVA (Stable Unit Treatment Value Assumption) is valid. Hence we expect the ASW to be close to zero)

3. **L2 Reconstruction Error**
   - Measures how well the model reconstructs the original data
   - Computed as the mean L2 norm of the difference between original and reconstructed data
   - Lower values indicate better reconstruction ability

## Dataset Information

The benchmarking uses the following datasets:
- **simulated** 
- **ifn_beta**
- **covid_epithelial**
- **covid_pbmc**
- **pbmc_batch_effect**
- **pbmc_negative_control**

## Model Comparison

The benchmarking compares two models:

1. **scCausalVI**: A variational autoencoder designed specifically for causal inference in single-cell data
   - Disentangles background cell state from treatment effects
   - Uses a specialized latent space structure
   - Implements MMD regularization to ensure disentanglement

2. **Model1**: An alternative implementation with different architectural choices
   - Uses mutual information minimization along with mmd to ensure disentanglement
   - Uses MINE (Mutual Information Neural Estimation), a neural network based approach to estimate mutual information

## Visualization Features

The dashboard provides:
- Side-by-side model comparison across all datasets
- Radar charts for multi-metric evaluation
- Detailed performance breakdowns by dataset
- Interactive filtering and selection options
