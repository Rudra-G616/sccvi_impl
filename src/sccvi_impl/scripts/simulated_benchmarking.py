"""
Simulated benchmarking script for comparing model_1 and scCausalVI models on simulated data.

This script:
1. Downloads simulated data
2. Performs preprocessing steps
3. Trains both models
4. Evaluates models on various metrics
5. Stores the evaluation results
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.metrics.cluster import silhouette_score

# Import project modules using relative imports
from ..data.Download_Dataset import download_dataset
from ..model.model_1 import Model1
from ..model.scCausalVI import scCausalVIModel as scCausalVI
from ..data.utils import gram_matrix

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
sc.settings.verbosity = 0  
sc.settings.set_figure_params(dpi=100, figsize=(10, 8))

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking of Model1 and scCausalVI on simulated data')
    parser.add_argument('--output_dir', type=str, default='sccvi_impl/src/sccvi_impl/results/simulated',
                        help='Directory to save results')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Use GPU for training if available')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--data_dir', type=str, default='sccvi_impl/src/sccvi_impl/data/datasets',
                        help='Directory to store downloaded datasets')
    parser.add_argument('--skip_visualization', action='store_true', default=False,
                        help='Skip UMAP calculation and visualization steps (useful for remote execution)')
    return parser.parse_args()

def download_data(data_dir):
    """Download the simulated dataset."""
    os.makedirs(data_dir, exist_ok=True)
    data_path = download_dataset('simulated', data_dir)
    return data_path

def load_and_preprocess_data(data_path, skip_visualization=False):
    """Load and preprocess the AnnData object."""
    print(f"Loading data from {data_path}")
    adata = sc.read_h5ad(data_path)
    
    # Basic information about the dataset
    print(f"AnnData object: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # As per requirements, cell_type is the batch key and condition is the treatment key
    print(f"Available batch categories: {adata.obs['cell_type'].unique()}")
    print(f"Available condition categories: {adata.obs['condition'].unique()}")
    
    # Preprocessing steps
    print("Performing preprocessing...")
    
    # Filter cells with too few genes
    sc.pp.filter_cells(adata, min_genes=200)
    
    # Filter genes that are expressed in too few cells
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Store raw counts for model training
    adata.layers['counts'] = adata.X.copy()
    
    # Normalize data
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Log-transform
    sc.pp.log1p(adata)
    
    # Store normalized data in .raw attribute
    adata.raw = adata
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(
        adata,
        flavor='seurat_v3',
        n_top_genes=2000,
        layer='counts',
        subset=True
    )
    
    print(f"After preprocessing: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Visualization of the data (skip if requested)
    if not skip_visualization:
        # Calculate UMAP first
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        
        # Then plot visualizations
        sc.pl.highest_expr_genes(adata, n_top=20, save=".png")
        sc.pl.umap(adata, color=['cell_type', 'condition'], save='.png')
    
    return adata

def setup_models(adata, batch_key='cell_type', condition_key='condition'):
    """Set up the models for training."""
    # Setup AnnData with common parameters for both models
    Model1.setup_anndata(
        adata, 
        layer='counts',
        batch_key=batch_key, 
        condition_key=condition_key
    )
    
    scCausalVI.setup_anndata(
        adata, 
        layer='counts',
        batch_key=batch_key, 
        condition_key=condition_key
    )
    
    # Create condition to integer mapping
    condition2int = adata.obs.groupby(condition_key, observed=False)['_scvi_condition'].first().to_dict()
    print(f"Condition to integer mapping: {condition2int}")
    
    # Determine control condition (assuming the first condition is control)
    control_key = list(condition2int.keys())[0]
    print(f"Using '{control_key}' as control condition")
    
    # Initialize models with the same hyperparameters for fair comparison
    model1 = Model1(
        adata,
        condition2int=condition2int,
        control=control_key,
        n_latent=10,
        n_layers=2,
        n_hidden=128,
        dropout_rate=0.1,
        use_observed_lib_size=True,
        use_mmd=True,
        mmd_weight=1.5,
        norm_weight=0.5,
        mi_weight=2
    )
    
    model_scvi = scCausalVI(
        adata,
        condition2int=condition2int,
        control=control_key,
        n_background_latent=10,
        n_te_latent=10,
        n_layers=2,
        n_hidden=128,
        dropout_rate=0.1,
        use_observed_lib_size=True,
        use_mmd=True,
        mmd_weight=1.5,
        norm_weight=0.5
    )
    
    return model1, model_scvi, condition2int, control_key

def prepare_data_for_training(adata, condition_key='condition'):
    """Prepare condition-specific indices for training."""
    conditions = adata.obs[condition_key].unique().tolist()
    group_indices_list = [np.where(adata.obs[condition_key] == condition)[0] for condition in conditions]
    return group_indices_list

def train_models(model1, model_scvi, group_indices_list, args):
    """Train both models with the same parameters."""
    print("Training Model1...")
    train_size = min(len(indices) for indices in group_indices_list)
    # Ensure batch size is smaller than training size
    batch_size = min(args.batch_size, train_size // 2)
    print(f"Using batch size: {batch_size}")
    
    # Disable epoch logging and progress bars
    trainer_kwargs = {
        'enable_progress_bar': False,
        'enable_model_summary': False,
        'logger': False,
        'enable_checkpointing': False
    }
    
    model1.train(
        group_indices_list,
        max_epochs=args.n_epochs,
        use_gpu=args.use_gpu,
        batch_size=batch_size,
        **trainer_kwargs
    )
    
    print("Training scCausalVI...")
    model_scvi.train(
        group_indices_list,
        max_epochs=args.n_epochs,
        use_gpu=args.use_gpu,
        batch_size=batch_size,
        **trainer_kwargs
    )
    
    return model1, model_scvi

def compute_l2_norm(x_hat, x):
    """Compute L2 norm of (x_hat-x)."""
    return np.linalg.norm(x_hat - x, axis=1).mean()

def compute_covariance_metric(z, factor):
    """Compute covariance between latent representation z and factor."""
    # Check if any of the inputs are empty
    if z.size == 0 or factor.size == 0:
        return 0.0
        
    # If factor has more dimensions than z, we might need to reshape or sample
    if isinstance(factor, np.ndarray) and factor.ndim > 1 and factor.shape[0] != z.shape[0]:
        print(f"Warning: Incompatible shapes in covariance calculation: {z.shape} vs {factor.shape}")
        return 0.0
        
    # Normalize z and factor
    z_norm = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)  # Add small epsilon to avoid division by zero
    
    # One-hot encode if factor is categorical
    if hasattr(factor, 'dtype') and factor.dtype.kind in 'OSU':  # Object, String, or Unicode
        factor_dummies = pd.get_dummies(factor).values
    elif isinstance(factor, np.ndarray) and factor.ndim > 1:
        # Already a numeric array with potentially multiple features
        factor_dummies = factor
    else:
        # Single numeric feature, reshape to column vector
        factor_dummies = np.asarray(factor).reshape(-1, 1)
    
    # Normalize factor
    factor_mean = factor_dummies.mean(axis=0)
    factor_std = factor_dummies.std(axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    factor_norm = (factor_dummies - factor_mean) / factor_std
    
    try:
        # Compute covariance
        cov = np.abs(np.cov(z_norm.T, factor_norm.T)[:z.shape[1], z.shape[1]:])
        
        # Return mean absolute covariance
        return np.mean(cov)
    except Exception as e:
        print(f"Error in covariance calculation: {e}")
        return 0.0

def compute_asw(z, batch_labels):
    """Compute Average Silhouette Width (ASW) for batch correction assessment."""
    # Check if input is empty or too small
    if z.size == 0 or z.shape[0] < 2:
        return 0.0
        
    # Convert batch labels to integers if they are strings
    if batch_labels.dtype.kind in 'OSU':
        unique_batches = list(np.unique(batch_labels))
        batch_indices = np.array([unique_batches.index(b) for b in batch_labels])
    else:
        batch_indices = batch_labels
        
    # Check if there are at least 2 batches
    if len(np.unique(batch_indices)) < 2:
        return 0.0
    
    # Compute silhouette score
    try:
        asw = silhouette_score(z, batch_indices)
    except Exception as e:
        print(f"Warning: Error computing ASW: {e}")
        asw = 0.0
    
    return asw

def evaluate_models(model1, model_scvi, adata, control_key, args):
    """Evaluate both models on multiple metrics."""
    results = {
        'model': [],
        'l2_norm': [],
        'cov_z_bg_c': [],
        'cov_z_bg_e_tilda': [],
        'cov_e_tilda_c': [],
        'asw_z_bg': []
    }
    
    # Get latent representations
    print("Getting latent representations...")
    model1_latent_tuple = model1.get_latent_representation(adata)
    model_scvi_latent_bg, model_scvi_latent_te = model_scvi.get_latent_representation(adata)
    
    # Unpack Model1 latent representations (now returns a tuple)
    model1_latent_bg, model1_latent_te = model1_latent_tuple
    
    # Get reconstructions for L2 norm computation
    print("Getting reconstructions...")
    model1_recon = model1.get_reconstructions(adata)
    model_scvi_recon_adata = model_scvi.get_count_expression(adata)
    model_scvi_recon = model_scvi_recon_adata.X
    
    # Extract original data for comparison
    original_data = adata.layers['counts']
    
    # Get batch and condition information
    batch_labels = adata.obs['cell_type'].values
    condition_labels = adata.obs['condition'].values
    
    # One-hot encode batch for covariance calculation
    batch_one_hot = pd.get_dummies(batch_labels).values
    
    # Compute metrics for Model1
    print("Computing metrics for Model1...")
    results['model'].append('Model1')
    results['l2_norm'].append(compute_l2_norm(model1_recon, original_data))
    results['cov_z_bg_c'].append(compute_covariance_metric(model1_latent_bg, batch_labels))
    
    # Now that we have separate background and treatment effect vectors, use them directly
    results['cov_z_bg_e_tilda'].append(compute_covariance_metric(model1_latent_bg, model1_latent_te))
    results['cov_e_tilda_c'].append(compute_covariance_metric(model1_latent_te, condition_labels))
    results['asw_z_bg'].append(compute_asw(model1_latent_bg, batch_labels))
    
    # Compute metrics for scCausalVI
    print("Computing metrics for scCausalVI...")
    results['model'].append('scCausalVI')
    results['l2_norm'].append(compute_l2_norm(model_scvi_recon, original_data))
    results['cov_z_bg_c'].append(compute_covariance_metric(model_scvi_latent_bg, batch_labels))
    
    # Check if the latent representations have compatible shapes
    if model_scvi_latent_bg.shape[0] == model_scvi_latent_te.shape[0]:
        results['cov_z_bg_e_tilda'].append(compute_covariance_metric(model_scvi_latent_bg, model_scvi_latent_te))
        results['cov_e_tilda_c'].append(compute_covariance_metric(model_scvi_latent_te, condition_labels))
        results['asw_z_bg'].append(compute_asw(model_scvi_latent_bg, batch_labels))
    else:
        print(f"Warning: scCausalVI latent representations have incompatible shapes: {model_scvi_latent_bg.shape} vs {model_scvi_latent_te.shape}")
        results['cov_z_bg_e_tilda'].append(0.0)
        results['cov_e_tilda_c'].append(0.0)
        results['asw_z_bg'].append(compute_asw(model_scvi_latent_bg, batch_labels))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'benchmarking_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Print results
    print("\nBenchmarking Results:")
    print(results_df)
    
    # Create visualizations of results
    if not args.skip_visualization:
        plot_results(results_df, args.output_dir)
    
    return results_df

def plot_results(results_df, output_dir):
    """Create and save visualizations of benchmarking results."""
    # Melt the dataframe for easier plotting
    melted_df = pd.melt(results_df, id_vars=['model'], var_name='metric', value_name='value')
    
    # Create directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create bar plot for all metrics
    plt.figure(figsize=(12, 8))
    sns.barplot(data=melted_df, x='metric', y='value', hue='model')
    plt.title('Benchmarking Results')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_metrics_comparison.png'))
    
    # Create individual plots for each metric
    for metric in results_df.columns[1:]:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=results_df, x='model', y=metric)
        plt.title(f'Comparison of {metric}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{metric}_comparison.png'))
    
    plt.close('all')

def main():
    # Parse command line arguments
    args = parse_args()
    
    try:

        data_path = download_data(args.data_dir)
            
        adata = load_and_preprocess_data(data_path, skip_visualization=args.skip_visualization)
    
        model1, model_scvi, condition2int, control_key = setup_models(adata)
        
        group_indices_list = prepare_data_for_training(adata)
        
        model1, model_scvi = train_models(model1, model_scvi, group_indices_list, args)
        
        results = evaluate_models(model1, model_scvi, adata, control_key, args)
        
        print("Benchmark completed successfully!")
        return True
    except Exception as e:
        import traceback
        print(f"Error during benchmark: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)

