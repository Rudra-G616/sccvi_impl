import os
import numpy as np
import pandas as pd
import torch
import anndata as ad
import scanpy as sc
from sklearn.metrics import silhouette_score
from typing import Dict, List, Optional, Tuple

from ..data.Download_Dataset import download_dataset, DATASET_URLS
from ..model.scCausalVI import scCausalVIModel
from ..model.model_1 import Model1


class ModelBenchmarker:
    """
    Class to benchmark scCausalVI and Model1 on various datasets.
    
    This class provides methods to:
    1. Train both models on a given dataset
    2. Compute various metrics for comparison:
       - Correlation between z_bg and e_tilda: E[z_bg*e_tilda] - E[z_bg]*E[e_tilda]
       - Average silhouette score on z_bg 
       - L2 norm of difference between x and x_reconstructed
       
    Note: For scCausalVI, the n_latent parameter is used for both n_background_latent and n_te_latent.
    For Model1, n_latent is used as is.
    """
    
    def __init__(
        self, 
        output_dir: str = "data",
        n_latent: int = 15,
        batch_size: int = 128,
        max_epochs: int = 100,
        use_gpu: bool = False
    ):
        """
        Initialize the benchmarker.
        
        Args:
            output_dir: Directory to save/load datasets and results
            n_latent: Dimensionality of latent space for both models
                      (used for n_latent in Model1 and for both n_background_latent and n_te_latent in scCausalVI)
            batch_size: Batch size for training
            max_epochs: Maximum number of epochs for training
            use_gpu: Whether to use GPU for training if available
        """
        self.output_dir = output_dir
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.use_gpu = use_gpu
        self.results = {}
        
        # Create results directory
        self.results_dir = os.path.join(output_dir, "benchmarking_results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _prepare_data(self, dataset_name: str) -> Tuple[ad.AnnData, List[np.ndarray]]:
        """
        Load and prepare a dataset for benchmarking.
        
        Args:
            dataset_name: Name of the dataset to load
        
        Returns:
            Tuple of (AnnData object, list of group indices)
        """
        # Download the dataset if needed
        file_path = download_dataset(dataset_name, self.output_dir)
        if file_path is None:
            raise ValueError(f"Failed to download dataset: {dataset_name}")
        
        # Load the dataset
        adata = ad.read_h5ad(file_path)
        print(f"Loaded dataset {dataset_name} with shape {adata.shape}")
        
        # Perform basic preprocessing if not already done
        if "highly_variable" not in adata.var:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        
        if "X_norm" not in adata.layers:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.layers["X_norm"] = adata.X.copy()
        
        # Get condition info - assuming 'condition' is in obs
        if "condition" not in adata.obs:
            raise ValueError(f"Dataset {dataset_name} does not have a 'condition' column in obs")
        
        # Create group indices
        conditions = adata.obs.condition.unique()
        group_indices = [np.where(adata.obs.condition == cond)[0] for cond in conditions]
        
        return adata, group_indices
    
    def train_models(self, dataset_name: str) -> Dict:
        """
        Train both scCausalVI and Model1 on a dataset and compute metrics.
        
        Args:
            dataset_name: Name of the dataset to benchmark on
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"Training models on dataset: {dataset_name}")
        adata, group_indices = self._prepare_data(dataset_name)
        
        # Setup condition mapping
        conditions = adata.obs.condition.unique()
        condition2int = {c: i for i, c in enumerate(conditions)}
        control = conditions[0]  # Assume first condition is control
        
        # Initialize and train scCausalVI model
        # Note: scCausalVI requires both n_background_latent and n_te_latent parameters
        # We use the same value (self.n_latent) for both to ensure equal dimensionality
        print("Training scCausalVI model...")
        sccausalvi = scCausalVIModel(
            adata,
            condition2int=condition2int,
            control=control,
            n_background_latent=self.n_latent,
            n_te_latent=self.n_latent,
        )
        sccausalvi.train(
            group_indices_list=group_indices,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            early_stopping=True,
            use_gpu=self.use_gpu
        )
        
        # Initialize and train Model1
        # Model1 only requires a single n_latent parameter
        print("Training Model1...")
        model1 = Model1(
            adata,
            condition2int=condition2int,
            control=control,
            n_latent=self.n_latent,
        )
        model1.train(
            group_indices_list=group_indices,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            early_stopping=True,
            use_gpu=self.use_gpu
        )
        
        # Validate models are properly trained
        if not hasattr(sccausalvi, 'module') or sccausalvi.module is None:
            raise ValueError("scCausalVI model not properly trained")
        if not hasattr(model1, 'module') or model1.module is None:
            raise ValueError("Model1 not properly trained")
        
        # Compute metrics
        print("Computing metrics...")
        metrics = {
            "dataset": dataset_name,
            "sccausalvi": self._compute_metrics(sccausalvi, adata),
            "model1": self._compute_metrics_model1(model1, adata)
        }
        
        # Save results
        self.results[dataset_name] = metrics
        self._save_results(dataset_name, metrics)
        
        return metrics
    
    def _compute_metrics(self, model: scCausalVIModel, adata: ad.AnnData) -> Dict:
        """Compute metrics for scCausalVI model"""
        try:
            # Get latent representations - scCausalVI returns tuple (bg, te)
            latent_bg, latent_te = model.get_latent_representation(adata)
            
            # Get reconstructions using get_count_expression method
            adata_recon = model.get_count_expression(adata)
            x_recon = adata_recon.X
            
            # Compute E[z_bg*e_tilda] - E[z_bg]*E[e_tilda]
            z_bg = torch.tensor(latent_bg, dtype=torch.float32)
            e_tilda = torch.tensor(latent_te, dtype=torch.float32)
            
            # Correct correlation calculation using element-wise multiplication
            correlation = torch.mean(z_bg * e_tilda).item() - torch.mean(z_bg).item() * torch.mean(e_tilda).item()
        except Exception as e:
            print(f"Error computing scCausalVI latent representations or reconstructions: {e}")
            return {
                "correlation_bg_te": float('nan'),
                "silhouette_bg": float('nan'),
                "reconstruction_error": float('nan')
            }
        
        # Compute silhouette score on z_bg
        # Get condition labels for silhouette score
        conditions = adata.obs.condition.values
        try:
            if len(np.unique(conditions)) > 1:  # Need at least 2 conditions for silhouette score
                silhouette = silhouette_score(latent_bg, conditions)
            else:
                silhouette = float('nan')  # Only one condition
        except Exception as e:
            print(f"Warning: Could not compute silhouette score: {e}")
            silhouette = float('nan')
        
        # Compute L2 norm of difference between x and x_reconstructed
        x_orig = adata.layers["X_norm"] if "X_norm" in adata.layers else adata.X
        x_orig_dense = x_orig.toarray() if hasattr(x_orig, "toarray") else x_orig
        x_recon_dense = x_recon.toarray() if hasattr(x_recon, "toarray") else x_recon
        reconstruction_error = np.mean(np.sqrt(np.sum((x_orig_dense - x_recon_dense)**2, axis=1)))
        
        return {
            "correlation_bg_te": correlation,
            "silhouette_bg": silhouette,
            "reconstruction_error": reconstruction_error
        }
    
    def _compute_metrics_model1(self, model: Model1, adata: ad.AnnData) -> Dict:
        """Compute metrics for Model1 model"""
        try:
            # For Model1, we need to interpret the unified latent space differently
            # Since Model1 doesn't have explicit bg/te separation like scCausalVI,
            # we'll extract the treatment effect representation from the model's inference
            
            from ..model.base import SCCAUSALVI_REGISTRY_KEYS
            from scvi.dataloaders import AnnDataLoader
            
            # Get unified latent representation as background
            latent_bg = model.get_latent_representation(adata)
            
            # Extract treatment effect representation by running inference
            data_loader = model._make_data_loader(
                adata=adata, 
                batch_size=128, 
                shuffle=False,
                data_loader_class=AnnDataLoader
            )
            
            e_t_list = []
            z_bg_list = []
            
            for tensors in data_loader:
                x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
                batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY]
                condition_label = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]
                
                inference_outputs = model.module.inference(
                    x=x, batch_index=batch_index, condition_label=condition_label
                )
                
                # Get background and treatment effect representations
                ctrl_mask = (condition_label == model.module.condition2int[model.module.control]).squeeze(dim=-1)
                
                # For all cells, collect z_bg (background representation)
                if torch.any(ctrl_mask):
                    z_bg_list.append(inference_outputs["control"]["z_bg"].detach().cpu())
                if torch.any(~ctrl_mask):
                    z_bg_list.append(inference_outputs["treatment"]["z_bg"].detach().cpu())
                    # For treatment cells, collect e_t (treatment effect)
                    e_t_list.append(inference_outputs["treatment"]["e_t"].detach().cpu())
                
            # Combine all representations
            if len(z_bg_list) > 0:
                latent_bg_tensor = torch.cat(z_bg_list, dim=0)
            else:
                latent_bg_tensor = torch.zeros((adata.n_obs, model.module.n_latent))
                
            if len(e_t_list) > 0:
                latent_te_tensor = torch.cat(e_t_list, dim=0)
                # Pad with zeros for control cells
                n_treatment = latent_te_tensor.shape[0]
                n_total = adata.n_obs
                if n_treatment < n_total:
                    padding = torch.zeros((n_total - n_treatment, latent_te_tensor.shape[1]))
                    latent_te_tensor = torch.cat([padding, latent_te_tensor], dim=0)
            else:
                # No treatment cells, create zero tensor
                latent_te_tensor = torch.zeros((adata.n_obs, model.module.n_latent))
            
            # Get reconstructions
            x_recon = model.get_normalized_expression(adata, library_size=1)
            
            # Compute correlation: E[z_bg*e_tilda] - E[z_bg]*E[e_tilda]
            # Use proper tensors with matching dimensions
            if latent_bg_tensor.shape[1] != latent_te_tensor.shape[1]:
                print(f"Warning: Latent space dimensions do not match: bg={latent_bg_tensor.shape[1]}, te={latent_te_tensor.shape[1]}")
                correlation = float('nan')
            else:
                min_samples = min(latent_bg_tensor.shape[0], latent_te_tensor.shape[0])
                z_bg = latent_bg_tensor[:min_samples]
                e_tilda = latent_te_tensor[:min_samples]
                
                # Correct correlation calculation using element-wise multiplication
                correlation = torch.mean(z_bg * e_tilda).item() - torch.mean(z_bg).item() * torch.mean(e_tilda).item()
        except Exception as e:
            print(f"Error computing Model1 metrics: {e}")
            return {
                "correlation_bg_te": float('nan'),
                "silhouette_bg": float('nan'),
                "reconstruction_error": float('nan')
            }
        
        # Compute silhouette score on z_bg (use numpy array)
        conditions = adata.obs.condition.values
        latent_bg_np = latent_bg_tensor.numpy() if hasattr(latent_bg_tensor, 'numpy') else latent_bg_tensor
        try:
            if len(np.unique(conditions)) > 1:  # Need at least 2 conditions for silhouette score
                silhouette = silhouette_score(latent_bg_np, conditions)
            else:
                silhouette = float('nan')  # Only one condition
        except Exception as e:
            print(f"Warning: Could not compute silhouette score: {e}")
            silhouette = float('nan')
        
        # Compute L2 norm of difference between x and x_reconstructed
        x_orig = adata.layers["X_norm"] if "X_norm" in adata.layers else adata.X
        x_orig_dense = x_orig.toarray() if hasattr(x_orig, "toarray") else x_orig
        x_recon_dense = x_recon.toarray() if hasattr(x_recon, "toarray") else x_recon
        reconstruction_error = np.mean(np.sqrt(np.sum((x_orig_dense - x_recon_dense)**2, axis=1)))
        
        return {
            "correlation_bg_te": correlation,
            "silhouette_bg": silhouette,
            "reconstruction_error": reconstruction_error
        }
    
    def _save_results(self, dataset_name: str, results: Dict) -> None:
        """Save benchmark results to file"""
        # Create a pandas DataFrame for easier saving
        df_data = []
        
        for model_name in ["sccausalvi", "model1"]:
            model_results = results[model_name]
            df_data.append({
                "dataset": dataset_name,
                "model": model_name,
                "correlation_bg_te": model_results["correlation_bg_te"],
                "silhouette_bg": model_results["silhouette_bg"],
                "reconstruction_error": model_results["reconstruction_error"]
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(self.results_dir, f"{dataset_name}_benchmark.csv"), index=False)
        
        # Also save as JSON for easier loading in the dashboard
        results_json = {
            "dataset": dataset_name,
            "sccausalvi": results["sccausalvi"],
            "model1": results["model1"]
        }
        
        import json
        with open(os.path.join(self.results_dir, f"{dataset_name}_benchmark.json"), "w") as f:
            json.dump(results_json, f)
    
    def benchmark_all_datasets(self) -> Dict:
        """Benchmark both models on all available datasets"""
        all_results = {}
        
        for dataset_name in DATASET_URLS.keys():
            try:
                results = self.train_models(dataset_name)
                all_results[dataset_name] = results
                print(f"Completed benchmarking for {dataset_name}")
            except Exception as e:
                print(f"Error benchmarking {dataset_name}: {str(e)}")
        
        # Combine all results into a single DataFrame
        self._save_combined_results(all_results)
        
        return all_results
    
    def _save_combined_results(self, all_results: Dict) -> None:
        """Save combined results from all datasets"""
        df_data = []
        
        for dataset_name, results in all_results.items():
            for model_name in ["sccausalvi", "model1"]:
                model_results = results[model_name]
                df_data.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "correlation_bg_te": model_results["correlation_bg_te"],
                    "silhouette_bg": model_results["silhouette_bg"],
                    "reconstruction_error": model_results["reconstruction_error"]
                })
        
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(self.results_dir, "all_benchmarks.csv"), index=False)
        
        # Also save as JSON
        import json
        with open(os.path.join(self.results_dir, "all_benchmarks.json"), "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model benchmarks")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", 
                        help="Directory to store benchmark results")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum number of training epochs")
    args = parser.parse_args()
    
    # Get output_dir from environment variable if set
    output_dir = os.environ.get("BENCHMARK_RESULTS_DIR", args.output_dir)
    
    benchmarker = ModelBenchmarker(output_dir=output_dir, max_epochs=args.max_epochs)
    benchmarker.benchmark_all_datasets()
