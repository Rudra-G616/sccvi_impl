import logging
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import scanpy as sc
import anndata as ad
from anndata import AnnData
from statsmodels.stats.multitest import multipletests

from numpy import ndarray
import pandas as pd
from .base import SCCAUSALVI_REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass
from scvi.distributions import ZeroInflatedNegativeBinomial

from .base.model_1_training_mixin import Model1TrainingMixin
from ..module.model_1 import Model1Module
from .base._utils import _invert_dict

logger = logging.getLogger(__name__)


class Model1(Model1TrainingMixin, BaseModelClass):
    """
    Model class for Model1.
    Args:
    -----
        adata: AnnData object with count data.
        condition2int: Dict mapping condition name (str) -> index (int)
        control: Control condition in case-control study, containing cells in unperturbed states
        n_latent: Dimensionality of latent space.
        n_layers: Number of hidden layers of each sub-networks.
        n_hidden: Number of hidden nodes in each layer of neural network.
        dropout_rate: Dropout rate for the network.
        use_observed_lib_size: Whether to use the observed library size.
        use_mmd: Whether to use Maximum Mean Discrepancy (MMD) to align latent representations
        across conditions.
        mmd_weight: Weight of MMD in loss function.
        norm_weight: Normalization weight in loss function.
        mi_weight: Weight for total correlation loss to promote better disentanglement.
            When > 0, a direct upper bound on total correlation is used to minimize
            dependence between background latent space, treatment effect, and batch.
        gammas: Kernel bandwidths for calculating MMD.
    """

    def __init__(
            self,
            adata: AnnData,
            condition2int: dict,
            control: str,
            n_latent: int = 10,
            n_layers: int = 2,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            use_observed_lib_size: bool = True,
            use_mmd: bool = True,
            mmd_weight: float = 1.0,
            norm_weight: float = 0.3,
            mi_weight: float = 0.0,
            gammas: Optional[np.ndarray] = None,
    ) -> None:

        super(Model1, self).__init__(adata)

        # Determine number of batches from summary stats
        n_batch = self.summary_stats.n_batch

        # Initialize library size parameters if not using observed library size
        if use_observed_lib_size:
            library_log_means, library_log_vars = None, None
        else:
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        # Set default gamma values for MMD if not provided
        if use_mmd and gammas is None:
            gammas = torch.FloatTensor([10 ** x for x in range(-6, 7, 1)])

        # Initialize the module with the specified parameters
        self.module = Model1Module(
            n_input=self.summary_stats["n_vars"],
            control=control,
            condition2int=condition2int,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            use_mmd=use_mmd,
            mmd_weight=mmd_weight,
            norm_weight=norm_weight,
            mi_weight=mi_weight,
            gammas=gammas,
        )

        # Summary string for the model
        self._model_summary_string = "Model1"

        # Capture initialization parameters for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized.")

    @classmethod
    def setup_anndata(
            cls,
            adata: AnnData,
            layer: Optional[str] = None,
            batch_key: Optional[str] = None,
            condition_key: Optional[str] = None,
            size_factor_key: Optional[str] = None,
            categorical_covariate_keys: Optional[List[str]] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        Set up AnnData instance for Model1

        Args:
            adata: AnnData object with .layers[layer] attribute containing count data.
            layer: Key for `.layers` or `.raw` where count data are stored.
            batch_key: Key for batch information in `adata.obs`.
            condition_key: Key for condition information in `adata.obs`.
            size_factor_key: Key for size factor information in `adata.obs`.
            categorical_covariate_keys: Keys for categorical covariates in `adata.obs`.
            continuous_covariate_keys: Keys for continuous covariates in `adata.obs`.
        """

        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(SCCAUSALVI_REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY, condition_key),
            NumericalObsField(
                SCCAUSALVI_REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                SCCAUSALVI_REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                SCCAUSALVI_REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            batch_size: Optional[int] = None,
    ) -> Union[ndarray, tuple[ndarray, ndarray]]:
        """
        Compute latent representation for each cell based on their condition labels.

        Args:
        ----
        adata: AnnData object. If `None`, defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Bool. If True, give mean of distribution instead of sampling from it.
        batch_size: Mini-batch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        
        Returns
        -------
            By default, returns a tuple of two numpy arrays with shape `(n_cells, n_latent)` for 
            background and treatment effect latent representations. For backwards compatibility,
            if the parameter 'return_tuple' is set to False, returns only the background latent representation.
        """
        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        latent_bg = []
        latent_te = []
        
        control_label_idx = self.module.condition2int[self.module.control]
        
        for tensors in data_loader:
            x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
            batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY].to(x.device)
            condition_label = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY].to(x.device)
            
            # The inference method returns a dictionary with 'control' and 'treatment' keys
            inference_outputs = self.module.inference(
                x=x, batch_index=batch_index, condition_label=condition_label
            )
            
            # Need to process control and treatment cells separately
            ctrl_mask = (condition_label == self.module.condition2int[self.module.control]).squeeze(dim=-1)
            
            # Initialize containers for all cells
            batch_size = x.shape[0]
            z_bg = torch.zeros((batch_size, self.module.n_latent), device=x.device)
            z_te = torch.zeros((batch_size, self.module.n_latent), device=x.device)
            
            if give_mean:
                # Use the mean of q(z|x)
                if torch.any(ctrl_mask):
                    z_bg[ctrl_mask] = inference_outputs["control"]["qbg_m"].to(x.device)
                    # Treatment effect is zero for control cells
                if torch.any(~ctrl_mask):
                    z_bg[~ctrl_mask] = inference_outputs["treatment"]["qbg_m"].to(x.device)
                    # For treatment cells, get treatment effect if available
                    if "qt_m" in inference_outputs["treatment"]:
                        z_te[~ctrl_mask] = inference_outputs["treatment"]["qt_m"].to(x.device)
                    elif "e_t" in inference_outputs["treatment"]:
                        z_te[~ctrl_mask] = inference_outputs["treatment"]["e_t"].to(x.device)
            else:
                # Use samples from q(z|x)
                if torch.any(ctrl_mask):
                    z_bg[ctrl_mask] = inference_outputs["control"]["z_bg"].to(x.device)
                    # Treatment effect is zero for control cells
                if torch.any(~ctrl_mask):
                    z_bg[~ctrl_mask] = inference_outputs["treatment"]["z_bg"].to(x.device)
                    # For treatment cells, get treatment effect if available
                    if "z_t" in inference_outputs["treatment"]:
                        z_te[~ctrl_mask] = inference_outputs["treatment"]["z_t"].to(x.device)
                    elif "e_t" in inference_outputs["treatment"]:
                        z_te[~ctrl_mask] = inference_outputs["treatment"]["e_t"].to(x.device)
            
            latent_bg.append(z_bg.detach().cpu())
            latent_te.append(z_te.detach().cpu())
                
        latent_bg = torch.cat(latent_bg, dim=0).numpy()
        latent_te = torch.cat(latent_te, dim=0).numpy()
        
        # Return a tuple of both latent representations
        return latent_bg, latent_te

    @torch.no_grad()
    def get_reconstructions(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return reconstructed expression values for all cells.
        
        Args:
        ----
            adata: AnnData object to use. If None, uses the model's AnnData object.
            indices: Indices of cells in adata to use. If None, uses all cells.
            batch_size: Minibatch size for data loading. If None, uses default.
            
        Returns:
        -------
            Array of reconstructed expression values for each cell.
        """
        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        
        reconstructions = []
        for tensors in data_loader:
            x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
            batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY]
            condition_label = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]
            
            # The forward method internally calls inference and generative
            inference_outputs = self.module.inference(
                x=x, batch_index=batch_index, condition_label=condition_label
            )
            
            # Use _get_generative_input to format the inference outputs for the generative model
            generative_inputs = self.module._get_generative_input(
                tensors={
                    SCCAUSALVI_REGISTRY_KEYS.X_KEY: x, 
                    SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY: batch_index,
                    SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY: condition_label
                },
                inference_outputs=inference_outputs
            )
            
            # Call the generative model with the prepared inputs
            generative_outputs = self.module.generative(
                z_bg=generative_inputs['z_bg'],
                e_t=generative_inputs['e_t'],
                library=generative_inputs['library'],
                batch_index=generative_inputs['batch_index']
            )
            
            # Get decoded values (reconstructions)
            px_rate = generative_outputs["px_rate"]
            reconstructions.append(px_rate.detach().cpu().numpy())
            
        return np.concatenate(reconstructions)
    
    @torch.no_grad()
    def get_normalized_expression(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            transform_batch: Optional[str] = None,
            batch_size: Optional[int] = None,
            gene_list: Optional[Sequence[str]] = None,
            library_size: Optional[Union[float, str]] = 1.0,
            n_samples: int = 1,
            return_mean: bool = True,
            return_numpy: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Get normalized expression values for cells
        
        Args:
        ----
            adata: AnnData object with expression data
            indices: Indices of cells to use
            transform_batch: Batch to condition on
            batch_size: Minibatch size for data loading
            gene_list: List of genes for which to get expression
            library_size: Scale factor for library size normalization
            n_samples: Number of posterior samples
            return_mean: Return mean of posterior samples
            return_numpy: Return numpy array if True, else pandas DataFrame
            
        Returns:
        -------
            Normalized expression values
        """
        adata = self._validate_anndata(adata)
        
        if indices is None:
            indices = np.arange(adata.n_obs)
        
        if n_samples > 1 and return_mean is False:
            raise ValueError("return_mean must be True when n_samples > 1")
            
        if gene_list is None:
            gene_indices = slice(None)
        else:
            all_genes = adata.var_names
            gene_indices = [np.where(all_genes == gene)[0][0] for gene in gene_list]
            
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        
        transform_batch_idx = None
        if transform_batch is not None:
            transform_batch_idx = adata.obs[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY].cat.categories.get_loc(transform_batch)
            
        if library_size == "latent":
            library_size = torch.ones(1)
        elif library_size == "observed":
            library_size = None
        else:
            library_size = torch.tensor(library_size)
            
        exprs = []
        for tensors in data_loader:
            x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
            batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY]
            condition_label = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]
            
            if transform_batch_idx is not None:
                batch_index = torch.ones_like(batch_index) * transform_batch_idx
            
            # Run inference
            inference_outputs = self.module.inference(
                x=x, batch_index=batch_index, condition_label=condition_label
            )
            
            # Prepare generative inputs
            generative_inputs = self.module._get_generative_input(
                tensors={
                    SCCAUSALVI_REGISTRY_KEYS.X_KEY: x, 
                    SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY: batch_index,
                    SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY: condition_label
                },
                inference_outputs=inference_outputs
            )
            
            # Generate expression
            generative_outputs = self.module.generative(
                z_bg=generative_inputs['z_bg'],
                e_t=generative_inputs['e_t'],
                library=generative_inputs['library'],
                batch_index=generative_inputs['batch_index']
            )
            
            # Get normalized expression values
            px_scale = generative_outputs["px_scale"]
            exprs.append(px_scale[:, gene_indices].detach().cpu())
                
        if n_samples > 1:
            # Average multiple samples if needed
            exprs = torch.cat(exprs, dim=0)
            exprs = exprs.reshape(-1, n_samples, exprs.size(-1)).mean(1)
        else:
            exprs = torch.cat(exprs, dim=0)
            
        if return_numpy:
            return exprs.numpy()
        else:
            return pd.DataFrame(
                exprs.numpy(),
                columns=adata.var_names[gene_indices],
                index=adata.obs_names[indices],
            )
    
    @torch.no_grad()
    def get_current_tc_estimate(self) -> float:
        """
        Get the current total correlation estimate from the model.
        
        This value represents the estimated dependency between the background latent space,
        treatment effect vector, and batch representation. A lower value indicates better
        disentanglement between these components.
        
        This method is only meaningful when mi_weight > 0 in the model initialization.
        
        Returns
        -------
            float: Current total correlation estimate. Returns 0.0 if total correlation
                   loss is not being used (mi_weight = 0).
        """
        return self.module.get_current_tc_estimate().item()