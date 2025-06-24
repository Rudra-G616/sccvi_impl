# Benchmarking Guide for scCausalVI

This guide provides instructions for running benchmarking comparisons between the `Model1` and `scCausalVI` models on various datasets. The benchmarking scripts allow you to quantitatively compare the performance of both models using multiple evaluation metrics.

## Available Benchmarking Scripts

The following benchmarking scripts are available in this directory:

1. `simulated_benchmarking.py` - Benchmarking on simulated data
2. `ifn_beta_benchmarking.py` - Benchmarking on IFN-β stimulated PBMCs
3. `covid_epithelial_benchmarking.py` - Benchmarking on COVID-19 epithelial cells
4. `covid_pbmc_benchmarking.py` - Benchmarking on COVID-19 PBMCs

## Running Benchmarking Scripts

### Basic Usage

To run a benchmarking script with default parameters:

```bash
python simulated_benchmarking.py
```

### Command Line Arguments

All benchmarking scripts support the following command line arguments:

- `--output_dir`: Directory to save results (default: `results/[dataset_name]`)
- `--n_epochs`: Number of training epochs (default: 100)
- `--use_gpu`: Flag to use GPU for training if available
- `--batch_size`: Batch size for training (default: 128)
- `--data_dir`: Directory to store downloaded datasets (default: `data/datasets`)

Example with custom parameters:

```bash
python simulated_benchmarking.py --output_dir custom_results/simulated --n_epochs 200 --batch_size 256 --use_gpu
```

## Benchmarking Process

Each benchmarking script follows a similar workflow:

1. **Data Acquisition**: Downloads the dataset if not already available
2. **Preprocessing**: Applies standard single-cell preprocessing steps
3. **Model Setup**: Sets up both models with identical hyperparameters
4. **Training**: Trains both models with the same configuration
5. **Evaluation**: Evaluates models on multiple metrics
6. **Results**: Saves numerical results and visualizations

## Evaluation Metrics

The benchmarking scripts compare models on the following metrics:

1. **L2 norm of (x_hat-x)**: Measures reconstruction accuracy (lower is better)
   - Quantifies how well the model reconstructs the original data

2. **Covariance between z_bg and c** (cov_z_bg_c): Measures batch effect (lower is better)
   - z_bg: Background latent representation
   - c: Batch (cell type) information
   - Lower values indicate better batch correction

3. **Covariance between z_bg and e_tilda** (cov_z_bg_e_tilda): Measures disentanglement (lower is better)
   - e_tilda: Treatment effect representation
   - Lower values indicate better disentanglement of biological and treatment effects

4. **Covariance between e_tilda and c** (cov_e_tilda_c): Measures treatment effect specificity (lower is better)
   - Lower values indicate that treatment effects are not confounded by batch

5. **Average Silhouette Width on z_bg** (asw_z_bg): Measures batch mixing (lower is better)
   - Lower values indicate better mixing of batches in the latent space

## Understanding Results

After running a benchmarking script, results are saved in two formats:

1. **CSV File**: `benchmark_results.csv` in the specified output directory
   - Contains numerical values for all metrics for both models

2. **Visualization Plots**: In the `plots` subdirectory
   - `all_metrics_comparison.png`: Bar plot comparing all metrics
   - Individual plots for each metric (e.g., `l2_norm_comparison.png`)

### Interpreting Metrics

When comparing Model1 and scCausalVI:

- **L2 norm**: Lower values indicate better reconstruction accuracy
- **Covariance metrics**: Lower values indicate better disentanglement of factors
- **ASW**: Lower values indicate better batch correction

## Dataset-Specific Notes

### Simulated Data

The simulated dataset contains artificial data with known ground truth factors. This makes it ideal for benchmarking as the true effects are known.

```bash
python simulated_benchmarking.py
```

### IFN-β Dataset

The IFN-β dataset contains PBMCs with and without IFN-β stimulation. This dataset is useful for evaluating how well the models capture real biological responses to treatment.

```bash
python ifn_beta_benchmarking.py
```

### COVID-19 Datasets

The COVID-19 datasets (epithelial and PBMC) allow comparison of model performance in understanding disease effects across different cell types.

```bash
python covid_epithelial_benchmarking.py
python covid_pbmc_benchmarking.py
```

## Advanced Usage

### Saving and Loading Models

To save trained models for later use:

```python
# Save models
model1.save("path/to/save/model1")
model_scvi.save("path/to/save/scvi_model")

# Load models
model1 = Model1.load("path/to/save/model1")
model_scvi = scCausalVI.load("path/to/save/scvi_model")
```

### Custom Evaluation

You can extend the benchmarking scripts to include additional evaluation metrics:

```python
def evaluate_models(model1, model_scvi, adata, control_key, args):
    # Existing code...
    
    # Add your custom metric
    results['custom_metric'] = []
    results['custom_metric'].append(compute_custom_metric(model1, adata))
    results['custom_metric'].append(compute_custom_metric(model_scvi, adata))
    
    # Existing code...
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size: `--batch_size 64`
   - Subsample the dataset in the script

2. **Training Taking Too Long**:
   - Reduce number of epochs: `--n_epochs 50`
   - Use GPU if available: `--use_gpu`

3. **Dataset Download Failures**:
   - Check internet connection
   - Manually download datasets using `python -m sccvi_impl.data.Download_Dataset`
