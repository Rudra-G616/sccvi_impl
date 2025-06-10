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
    """
    
    def __init__(
        self, 
        output_dir: str = "data",
        n_background_latent: int = 10,
        n_te_latent: int = 5,
        n_latent: int = 15,
        batch_size: int = 128,
        max_epochs: int = 100
    ):
        """
        Initialize the benchmarker.
        
        Args:
            output_dir: Directory to save/load datasets and results
            n_background_latent: Dimensionality of background latent space for scCausalVI
            n_te_latent: Dimensionality of treatment effect latent space for scCausalVI
            n_latent: Dimensionality of latent space for Model1
            batch_size: Batch size for training
            max_epochs: Maximum number of epochs for training
        """
        self.output_dir = output_dir
        self.n_background_latent = n_background_latent
        self.n_te_latent = n_te_latent
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.max_epochs = max_epochs
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
        print("Training scCausalVI model...")
        sccausalvi = scCausalVIModel(
            adata,
            condition2int=condition2int,
            control=control,
            n_background_latent=self.n_background_latent,
            n_te_latent=self.n_te_latent,
        )
        sccausalvi.train(
            group_indices_list=group_indices,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            early_stopping=True
        )
        
        # Initialize and train Model1
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
            early_stopping=True
        )
        
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
        # Get latent representations
        latent_bg = model.get_latent_representation(adata, give_z="background")
        latent_te = model.get_latent_representation(adata, give_z="effect")
        
        # Get reconstructions
        x_recon = model.get_normalized_expression(adata, library_size=1)
        
        # Compute E[z_bg*e_tilda] - E[z_bg]*E[e_tilda]
        z_bg = torch.tensor(latent_bg, dtype=torch.float32)
        e_tilda = torch.tensor(latent_te, dtype=torch.float32)
        
        # Mean of the product - product of the means
        correlation = torch.mean(torch.bmm(
            z_bg.unsqueeze(2), 
            e_tilda.unsqueeze(1)
        )).item() - torch.mean(z_bg).item() * torch.mean(e_tilda).item()
        
        # Compute silhouette score on z_bg
        # Get condition labels for silhouette score
        conditions = adata.obs.condition.values
        try:
            silhouette = silhouette_score(latent_bg, conditions)
        except:
            silhouette = float('nan')  # In case of only one condition or other issues
        
        # Compute L2 norm of difference between x and x_reconstructed
        x_orig = adata.layers["X_norm"] if "X_norm" in adata.layers else adata.X
        x_orig_dense = x_orig.toarray() if hasattr(x_orig, "toarray") else x_orig
        reconstruction_error = np.mean(np.sqrt(np.sum((x_orig_dense - x_recon)**2, axis=1)))
        
        return {
            "correlation_bg_te": correlation,
            "silhouette_bg": silhouette,
            "reconstruction_error": reconstruction_error
        }
    
    def _compute_metrics_model1(self, model: Model1, adata: ad.AnnData) -> Dict:
        """Compute metrics for Model1 model"""
        # For Model1, we need to extract background and treatment effect parts
        # from the unified latent space
        latent = model.get_latent_representation(adata)
        
        # Assuming first n_background_latent dimensions are background
        # and the rest are treatment effect
        latent_bg = latent[:, :self.n_background_latent]
        latent_te = latent[:, self.n_background_latent:]
        
        # Get reconstructions
        x_recon = model.get_normalized_expression(adata, library_size=1)
        
        # Compute E[z_bg*e_tilda] - E[z_bg]*E[e_tilda]
        z_bg = torch.tensor(latent_bg, dtype=torch.float32)
        e_tilda = torch.tensor(latent_te, dtype=torch.float32)
        
        # Mean of the product - product of the means
        correlation = torch.mean(torch.bmm(
            z_bg.unsqueeze(2), 
            e_tilda.unsqueeze(1)
        )).item() - torch.mean(z_bg).item() * torch.mean(e_tilda).item()
        
        # Compute silhouette score on z_bg
        conditions = adata.obs.condition.values
        try:
            silhouette = silhouette_score(latent_bg, conditions)
        except:
            silhouette = float('nan')
        
        # Compute L2 norm of difference between x and x_reconstructed
        x_orig = adata.layers["X_norm"] if "X_norm" in adata.layers else adata.X
        x_orig_dense = x_orig.toarray() if hasattr(x_orig, "toarray") else x_orig
        reconstruction_error = np.mean(np.sqrt(np.sum((x_orig_dense - x_recon)**2, axis=1)))
        
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
    benchmarker = ModelBenchmarker(output_dir="data", max_epochs=50)
    benchmarker.benchmark_all_datasets()
