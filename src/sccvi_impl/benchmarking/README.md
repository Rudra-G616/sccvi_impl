# scCausalVI vs Model1 Benchmarking

This module provides tools to benchmark `scCausalVI` and `Model1` on various single-cell datasets and visualize the results through a Streamlit dashboard.

## Features

- Download and prepare five single-cell datasets from the paper
- Train both `scCausalVI` and `Model1` on each dataset
- Calculate and compare key metrics:
  - Correlation: E[z_bg*e_tilda] - E[z_bg]*E[e_tilda]
  - Average silhouette score on z_bg
  - L2 norm reconstruction error
- Visualize results through an interactive Streamlit dashboard

## Installation

Ensure you have all required dependencies:

```bash
pip install -r requirements.txt
```

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

### Running the Dashboard

Once you have generated benchmark results, you can view them in the Streamlit dashboard:

```bash
python -m sccvi_impl.benchmarking.run_dashboard
```

The dashboard will be available at http://localhost:8501 by default.

## Metrics Explanation

1. **Correlation between z_bg and e_tilda**
   - Measures independence between background and treatment effect latent variables
   - Computed as: E[z_bg*e_tilda] - E[z_bg]*E[e_tilda]
   - Lower values indicate better disentanglement

2. **Average Silhouette Score on z_bg**
   - Measures how well cells from different conditions are separated in the background latent space
   - Higher values indicate better condition-invariant representation

3. **L2 Reconstruction Error**
   - Measures how well the model reconstructs the original data
   - Computed as the L2 norm of the difference between original and reconstructed data
   - Lower values indicate better reconstruction ability

## Dataset Information

The benchmarking uses the following datasets:
- simulated: Simulated dataset
- ifn_beta: Interferon beta dataset
- covid_epithelial: COVID-19 epithelial cells
- covid_pbmc: COVID-19 peripheral blood mononuclear cells (PBMCs)
- pbmc_batch_effect: PBMCs with batch effects
- pbmc_negative_control: PBMCs with negative control
