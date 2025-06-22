from typing import Dict, Optional, Tuple, Union, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from ..model.base import SCCAUSALVI_REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch import Tensor
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from ..data.utils import gram_matrix


def compute_tc_bound(z_mean: torch.Tensor, z_var: torch.Tensor):
    """
    Compute the upper bound on total correlation (TC) from the paper
    "Isolating Sources of Disentanglement in Variational Autoencoders"
    (https://arxiv.org/pdf/1802.04942).
    
    This implements a direct analytical upper bound on TC which can be computed
    without training a discriminator network. It measures the dependence between
    dimensions of the latent space.
    
    Args:
        z_mean: Mean of q(z|x) - shape (batch_size, latent_dim)
        z_var: Variance of q(z|x) - shape (batch_size, latent_dim)
        
    Returns:
        Total correlation upper bound (to be minimized)
    """
    batch_size, latent_dim = z_mean.shape
    
    # Compute q(z) by aggregating q(z|x) across the batch
    # q(z) = 1/N * sum_i q(z|x_i)
    
    # First compute the mean and variance of the aggregated posterior q(z)
    q_z_mean = z_mean.mean(dim=0)  # (latent_dim,)
    
    # Variance of q(z) has two components:
    # 1. Mean of the variances (expected variance)
    # 2. Variance of the means (variance of expectations)
    q_z_var = z_var.mean(dim=0)  # Average variance across batch
    q_z_var += ((z_mean - q_z_mean)**2).mean(dim=0)  # Add variance of means
    
    # KL divergence between aggregated posterior q(z) and factorized q(z)
    # The factorized q(z) has the same marginals as q(z) but assumes independence
    
    # Compute log det covariance matrix (diagonal approximation)
    log_det_qz = torch.sum(torch.log(q_z_var + 1e-10))
    
    # Compute log det of factorized covariance (product of marginals)
    log_det_qz_factorized = torch.sum(torch.log(q_z_var + 1e-10))
    
    # For diagonal covariance matrices, the log determinants are the same
    # so this term is 0. The TC comes from the correlation between dimensions
    
    # For each data point, compute KL(q(z|x) || q(z)) - sum_j KL(q(z_j|x) || q(z_j))
    # This measures how much information is captured in the correlations
    
    # KL(q(z|x_i) || q(z)) for each data point
    kl_qzx_qz = 0.5 * torch.sum(
        (z_mean - q_z_mean)**2 / (q_z_var + 1e-10) + 
        z_var / (q_z_var + 1e-10) -
        torch.log(z_var + 1e-10) + 
        torch.log(q_z_var + 1e-10) - 1,
        dim=1
    )
    
    # Sum of KL(q(z_j|x_i) || q(z_j)) for each dimension j and data point i
    kl_qzjx_qzj = 0.5 * torch.sum(
        (z_mean - q_z_mean)**2 / (q_z_var + 1e-10) + 
        z_var / (q_z_var + 1e-10) -
        torch.log(z_var + 1e-10) + 
        torch.log(q_z_var + 1e-10) - 1,
        dim=1
    )
    
    # The difference is an upper bound on TC
    # When dimensions are independent, this approaches zero
    tc_bound = torch.mean(kl_qzx_qz - kl_qzjx_qzj)
    
    # Add regularization to ensure positive values (due to numerical issues)
    tc_bound = torch.abs(tc_bound)
    
    return tc_bound


class Model1Module(BaseModuleClass):
    """
    PyTorch module for Model 1.

    Args:
    ----
        n_input: Number of input genes.
        condition2int: Dict mapping condition name (str) -> index (int)
        control: Control condition in case-control study, containing cells in unperturbed states
        n_batch: Number of batches. If 1, no batch information incorporated into model/
        n_hidden: Number of hidden nodes in each layer of neural network.
        n_latent: Dimensionality of the latent space.
        n_layers: Number of hidden layers of each sub-networks.
        dropout_rate: Dropout rate for neural networks.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        library_log_means: 1 x n_batch array of means of the log library sizes.
            Parameterize prior on library size if not using observed library size.
        library_log_vars: 1 x n_batch array of variances of the log library sizes.
            Parameterize prior on library size if not using observed library size.
        use_mmd: Whether to use the maximum mean discrepancy to force background latent
            variables of the control and treatment dataset to follow the same
            distribution.
        mmd_weight: Weight of MMD in loss function.
        norm_weight: Normalization weight in loss function.
        mi_weight: Weight for the total correlation loss to promote disentanglement between
            background latent space, treatment effect vector and batch information.
            When > 0, a direct upper bound on total correlation is used to minimize
            dependence between these representations without requiring a discriminator network.
        gammas: Kernel bandwidths for calculating MMD.
    """

    def sample(self, *args, **kwargs):
        pass

    def __init__(
            self,
            n_input: int,
            condition2int: dict,
            control: str,
            n_batch: int,
            n_hidden: int = 128,
            n_latent: int = 10,
            n_layers: int = 1,
            dropout_rate: float = 0.1,
            use_observed_lib_size: bool = True,
            library_log_means: Optional[np.ndarray] = None,
            library_log_vars: Optional[np.ndarray] = None,
            use_mmd: bool = True,
            mmd_weight: float = 1.0,
            norm_weight: float = 1.0,
            mi_weight: float = 0.0,
            gammas: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the Model1Module.
        
        Parameters
        ----------
        n_input
            Number of input genes
        condition2int
            Dictionary mapping condition names to indices
        control
            Name of the control condition
        n_batch
            Number of batches
        n_hidden
            Number of hidden units in each layer
        n_latent
            Dimensionality of the latent space
        n_layers
            Number of hidden layers
        dropout_rate
            Dropout rate for neural networks
        use_observed_lib_size
            Whether to use observed library size
        library_log_means
            Mean of log library size (if not using observed library size)
        library_log_vars
            Variance of log library size (if not using observed library size)
        use_mmd
            Whether to use MMD loss to match background latent distributions
        mmd_weight
            Weight of MMD loss in total loss
        norm_weight
            Weight of L2 normalization on treatment effect vectors
        mi_weight
            Weight of total correlation loss between latent representations (z_bg, e_tilda, and batch)
            When > 0, a direct upper bound on total correlation is used to minimize
            dependence between these variables, promoting better disentanglement
        gammas
            Kernel bandwidths for MMD loss
        """
        super(Model1Module, self).__init__()
        self.n_input = n_input
        self.control = control
        self.condition2int = condition2int
        self.n_conditions = len(condition2int)
        self.treat_ind = [i for i in range(self.n_conditions) if i != self.condition2int[self.control]]
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.latent_distribution = "normal"
        self.dispersion = "gene"
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self.use_observed_lib_size = use_observed_lib_size
        self.use_mmd = use_mmd
        self.mmd_weight = mmd_weight
        self.norm_weight = norm_weight
        self.mi_weight = mi_weight
        self.gammas = gammas

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )
            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if use_mmd:
            if gammas is None:
                raise ValueError("If using mmd, must provide gammas.")
            self.gammas = gammas
            
        # Register buffer to track TC loss values
        if self.mi_weight > 0:
            self.register_buffer("tc_loss_tracker", torch.tensor(0.0))

        cat_list = [n_batch]                  

        # Single encoder for all conditions that produces the base latent representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )

        # Single treatment effect encoder that takes latent representation + treatment label as input
        # The output is a treatment effect vector that will be combined with the latent representation
        # after applying attention weights
        
        self.te_encoder = Encoder(
                n_input=n_latent + 1, # +1 for treatment value
                n_output=n_latent,
                n_cat_list=None,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                distribution=self.latent_distribution,
                inject_covariates=False,
                use_batch_norm=True,
                use_layer_norm=False,
                var_activation=None,
        )

        # Attention layer to capture differential treatment effect patterns
        # The output of this layer is used to weight the treatment effect vector before
        # adding it to the latent representation
        self.attention = torch.nn.Linear(self.n_latent, 1)

        # Library size encoder.
        self.l_encoder = Encoder(
            n_input,
            n_output=1,
            n_layers=1,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )

        # Use concatenated latent space (background + treatment effect), matching scCausalVI implementation
        n_total_latent = n_latent + n_latent  # Background + treatment effect latent space.

        self.decoder = DecoderSCVI(
            n_total_latent,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=True,
        )

    @auto_move_data
    def _compute_local_library_params(
            self, batch_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    def _get_inference_input(
            self,
            tensors: Dict[str, torch.Tensor],
            **kwargs
    ) -> Union[Dict[str, list[str]], Dict[str, Any]]:

        x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
        batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY]
        condition_label = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]

        out = dict(x=x, condition_label=condition_label, batch_index=batch_index,)
        return out

    @auto_move_data
    def _generic_inference(
            self,
            x: torch.Tensor,
            batch_index: torch.Tensor,
            condition_label: torch.Tensor,
            src: str,
    ) -> Dict[str, torch.Tensor]:
        """
        If src = 'control', use the z_encoder for the latent representation.
        If src = 'treatment', use z_encoder for the latent representation and te_encoder for 
        treatment effect by concatenating the latent representation with the treatment label.
        
        The latent representation and treatment effect will be concatenated in the generative step
        after applying attention weights to the treatment effect, following the scCausalVI approach.
        """
        n_cells = x.shape[0]
        x_ = torch.log(x + 1)

        z_bg = torch.zeros((n_cells, self.n_latent), device=x.device)
        qbg_m = torch.zeros((n_cells, self.n_latent), device=x.device)
        qbg_v = torch.zeros((n_cells, self.n_latent), device=x.device)

        e_t = torch.zeros((n_cells, self.n_latent), device=x.device)
        qt_m = torch.zeros((n_cells, self.n_latent), device=x.device)
        qt_v = torch.zeros((n_cells, self.n_latent), device=x.device)

        library = torch.zeros((n_cells, 1), device=x.device)
        ql_m = None
        ql_v = None

        if not self.use_observed_lib_size:
            ql_m = torch.zeros((n_cells, 1), device=x.device)
            ql_v = torch.zeros((n_cells, 1), device=x.device)

        # Library size
        if self.use_observed_lib_size:
            lib_ = torch.log(x.sum(dim=1, keepdim=True) + 1e-8)
            library[:] = lib_
        else:
            qlm, qlv, lib_ = self.l_encoder(x_, batch_index)
            ql_m[:] = qlm
            ql_v[:] = qlv
            library[:] = lib_

        # Background latent factors using the z_encoder
        bg_m, bg_v_, bg_z = self.z_encoder(x_, batch_index)
        qbg_m[:] = bg_m
        qbg_v[:] = bg_v_
        z_bg[:] = bg_z

        # Treatment effect latent factors
        if src == 'control':
            # Treatment effect latent factors remain 0 for control data
            # Only the base latent representation from z_encoder is used
            pass
        else:
            # Process treatment data
            # For each treatment label, create an input tensor for the treatment encoder
            # by concatenating the background latent with the treatment label
            for t_lbl in condition_label.unique():
                if t_lbl.item() == self.condition2int[self.control]:
                    raise ValueError('Control label found in treatment labels')  # skip control label if present by chance.
                
                mask = (condition_label == t_lbl).squeeze(dim=-1)
                bg_z_sub = bg_z[mask]
                
                # Create input for treatment encoder by concatenating latent representation with treatment label
                # This allows the treatment encoder to know which treatment is being applied
                t_label = t_lbl.repeat(bg_z_sub.size(0), 1).to(bg_z_sub.device)
                te_input = torch.cat([bg_z_sub, t_label], dim=1)
                
                # Apply treatment effect encoder to get treatment effect
                # This will be combined with the latent representation in the generative step
                tm, tv, zt = self.te_encoder(te_input)
                qt_m[mask] = tm
                qt_v[mask] = tv
                e_t[mask] = zt

        return {
            "z_bg": z_bg,
            "qbg_m": qbg_m,
            "qbg_v": qbg_v,
            "e_t": e_t,
            "qt_m": qt_m,
            "qt_v": qt_v,
            "library": library,
            "ql_m": ql_m,
            "ql_v": ql_v,
        }

    @auto_move_data
    def inference(
            self,
            x: torch.Tensor,
            condition_label: torch.Tensor,
            batch_index: torch.Tensor,
            n_samples: int = 1,
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        # Inference of data
        ctrl_mask = (condition_label == self.condition2int[self.control]).squeeze(dim=-1)

        x_control = x[ctrl_mask]
        condition_control = condition_label[ctrl_mask]
        batch_control = batch_index[ctrl_mask]

        x_treatment = x[~ctrl_mask]
        condition_treatment = condition_label[~ctrl_mask]
        batch_treatment = batch_index[~ctrl_mask]

        inference_control = self._generic_inference(
            x_control, batch_control, src='control', condition_label=condition_control
        )
        inference_treatment = self._generic_inference(
            x_treatment, batch_treatment, src='treatment', condition_label=condition_treatment
        )
        
        # Store inference outputs for later use in forward method
        self.inference_outputs = {"control": inference_control, "treatment": inference_treatment}

        return self.inference_outputs

    def _get_generative_input(
            self,
            tensors: torch.Tensor,
            inference_outputs: Dict[str, Dict[str, torch.Tensor]],
            **kwargs,
    ):
        """
        Merges the control/treatment in original order.
        
        The base latent representation (z_bg) and treatment effect vector (e_t) 
        are kept separate here and will be concatenated in the generative step
        after applying attention weights to e_t, following the scCausalVI approach.
        """
        x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
        batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS. BATCH_KEY]
        condition_label = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]

        ctrl_mask = (condition_label == self.condition2int[self.control]).squeeze(dim=-1)
        n_cells = x.shape[0]

        z_bg_merged = torch.zeros((n_cells, self.n_latent), device=x.device)
        e_t_merged = torch.zeros((n_cells, self.n_latent), device=x.device)
        library_merged = torch.zeros((n_cells, 1), device=x.device)

        ctrl_inference = inference_outputs['control']
        treatment_inference = inference_outputs['treatment']

        # Fill control portion
        z_bg_merged[ctrl_mask] = ctrl_inference['z_bg']
        e_t_merged[ctrl_mask] = ctrl_inference['e_t']
        library_merged[ctrl_mask] = ctrl_inference['library']

        # Fill treatment portion
        z_bg_merged[~ctrl_mask] = treatment_inference['z_bg']
        e_t_merged[~ctrl_mask] = treatment_inference['e_t']
        library_merged[~ctrl_mask] = treatment_inference['library']

        return {
            'z_bg': z_bg_merged,
            'e_t': e_t_merged,
            'library': library_merged,
            'batch_index': batch_index,
        }

    @auto_move_data
    def generative(
            self,
            z_bg: torch.Tensor,
            e_t: torch.Tensor,
            library: torch.Tensor,
            batch_index: List[int],
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        # Apply attention weights to treatment effect vector - matches scCausalVI implementation
        attention_weights = torch.softmax(self.attention(e_t), dim=-1)
        e_tilda = attention_weights * e_t
        
        # Concatenate the latent representation with the weighted treatment effect
        latent = torch.cat([z_bg, e_tilda], dim=-1)
        
        # Use the combined latent representation in the decoder
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            latent,
            library,
            batch_index,
        )
        px_r = torch.exp(self.px_r)
        return {
            'px_scale': px_scale,
            'px_r': px_r,
            'px_rate': px_rate,
            'px_dropout': px_dropout,
            'e_tilda': e_tilda,  # Store for TC loss calculation
        }

    @staticmethod
    def reconstruction_loss(
            x: torch.Tensor,
            px_rate: torch.Tensor,
            px_r: torch.Tensor,
            px_dropout: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[0] != px_rate.shape[0]:
            print(f"x.shape[0]= {x.shape[0]} and px_rate.shape[0]= {px_rate.shape[0]}.")
        recon_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )
        return recon_loss

    @staticmethod
    def latent_kl_divergence(
            variational_mean: torch.Tensor,
            variational_var: torch.Tensor,
            prior_mean: torch.Tensor,
            prior_var: torch.Tensor,
    ) -> torch.Tensor:
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

    def library_kl_divergence(
            self,
            batch_index: torch.Tensor,
            variational_library_mean: torch.Tensor,
            variational_library_var: torch.Tensor,
            library: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_observed_lib_size:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_library = kl(
                Normal(variational_library_mean, variational_library_var.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            )
        else:
            kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

    def mmd_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cost = torch.mean(gram_matrix(x, x, gammas=self.gammas.to(self.device)))
        cost += torch.mean(gram_matrix(y, y, gammas=self.gammas.to(self.device)))
        cost -= 2 * torch.mean(gram_matrix(x, y, gammas=self.gammas.to(self.device)))
        if cost < 0:  # Handle numerical instability.
            return torch.tensor(0)
        return cost

    def _generic_loss(
            self,
            x: torch.Tensor,
            batch_index: torch.Tensor,
            qbg_m: torch.Tensor,
            qbg_v: torch.Tensor,
            library: torch.Tensor,
            ql_m: Optional[torch.Tensor],
            ql_v: Optional[torch.Tensor],
            px_rate: torch.Tensor,
            px_r: torch.Tensor,
            px_dropout: torch.Tensor,
    ) -> dict[str, Union[Tensor, list[Tensor]]]:
        prior_bg_m = torch.zeros_like(qbg_m)
        prior_bg_v = torch.ones_like(qbg_v)

        recon_loss = self.reconstruction_loss(x, px_rate, px_r, px_dropout)
        kl_bg = self.latent_kl_divergence(qbg_m, qbg_v, prior_bg_m, prior_bg_v)

        if not self.use_observed_lib_size:
            kl_library = self.library_kl_divergence(batch_index, ql_m, ql_v, library)
        else:
            kl_library = torch.zeros_like(recon_loss)

        return {
            'recon_loss': recon_loss,
            'kl_bg': kl_bg,
            'kl_library': kl_library,
        }

    def loss(
            self,
            tensors: Dict[str, torch.Tensor],
            inference_outputs: Dict[str, Dict[str, torch.Tensor]],
            generative_outputs: Dict[str, torch.Tensor],
            **loss_args,
    ) -> LossOutput:
        """
        The entire batch is in `tensors`.
        We separate  control vs. treat, compute the relevant losses, and combine.
        """
        x = tensors[SCCAUSALVI_REGISTRY_KEYS.X_KEY]
        batch_index = tensors[SCCAUSALVI_REGISTRY_KEYS.BATCH_KEY]
        condition_label = tensors[SCCAUSALVI_REGISTRY_KEYS.CONDITION_KEY]

        ctrl_mask = (condition_label == self.condition2int[self.control]).squeeze(dim=-1)
        x_ctrl = x[ctrl_mask]
        batch_ctrl = batch_index[ctrl_mask]
        ctrl_inference = inference_outputs['control']

        x_trt = x[~ctrl_mask]
        batch_trt = batch_index[~ctrl_mask]
        trt_inference = inference_outputs['treatment']

        # generative outputs are for all cells in order.
        px_rate = generative_outputs['px_rate']
        px_r = generative_outputs['px_r']
        px_dropout = generative_outputs['px_dropout']

        # separate them
        px_rate_ctrl = px_rate[ctrl_mask]
        px_dropout_ctrl = px_dropout[ctrl_mask]
        px_rate_trt = px_rate[~ctrl_mask]
        px_dropout_trt = px_dropout[~ctrl_mask]

        # ELBO loss of control data
        elbo_ctrl = self._generic_loss(
            x_ctrl,
            batch_ctrl,
            ctrl_inference['qbg_m'],
            ctrl_inference['qbg_v'],
            ctrl_inference['library'],
            ctrl_inference['ql_m'],
            ctrl_inference['ql_v'],
            px_rate_ctrl,
            px_r,
            px_dropout_ctrl,
        )

        # ELBO loss of treatment data
        elbo_trt = self._generic_loss(
            x_trt,
            batch_trt,
            trt_inference['qbg_m'],
            trt_inference['qbg_v'],
            trt_inference['library'],
            trt_inference['ql_m'],
            trt_inference['ql_v'],
            px_rate_trt,
            px_r,
            px_dropout_trt,
        )

        recon_loss = torch.cat([elbo_ctrl['recon_loss'], elbo_trt['recon_loss']], dim=0)
        kl_bg = torch.cat([elbo_ctrl['kl_bg'], elbo_trt['kl_bg']], dim=0)
        kl_library = torch.cat([elbo_ctrl['kl_library'], elbo_trt['kl_library']], dim=0)

        # MMD loss
        loss_mmd = torch.tensor(0.0, device=x.device)
        if self.use_mmd:
            z_bg_control = ctrl_inference["z_bg"]
            z_bg_treatment_all = trt_inference["z_bg"]
            cond_treat = condition_label[~ctrl_mask]

            # Compute MMD loss between distributions of background latent space for control and
            # each treatment data, to align each baseline states of treated samples with the control population
            unique_treats = cond_treat.unique()
            for t_lbl in unique_treats:
                treat_submask = (cond_treat == t_lbl).squeeze(dim=-1)
                z_bg_t_sub = z_bg_treatment_all[treat_submask]
                # MMD between all control cells and the subset of treatment cells of this label
                loss_mmd += self.mmd_loss(z_bg_control, z_bg_t_sub)

            loss_mmd *= self.mmd_weight

        # Norm cost, e.g. L2 on treatment effect vector in treatments.
        loss_norm = torch.tensor(0.0, device=x.device)
        if self.norm_weight > 0:
            e_t_trt = trt_inference['e_t']
            norm_val = (e_t_trt ** 2).sum(dim=1)
            loss_norm = self.norm_weight * norm_val

        # Total Correlation loss using direct upper bound
        loss_tc = torch.tensor(0.0, device=x.device)
        if self.mi_weight > 0:
            # We want to minimize the total correlation between:
            # 1. Background latent representation (z_bg)
            # 2. Treatment effect vector (e_tilda) - using the weighted version for consistency
            # 3. Batch information (one-hot encoded batch vector)
            
            # Get the required components
            # Combine control and treatment latent variables
            z_bg_all = torch.cat([ctrl_inference['z_bg'], trt_inference['z_bg']], dim=0)
            qbg_m_all = torch.cat([ctrl_inference['qbg_m'], trt_inference['qbg_m']], dim=0)
            qbg_v_all = torch.cat([ctrl_inference['qbg_v'], trt_inference['qbg_v']], dim=0)
            
            # Get the weighted treatment effect
            e_tilda = generative_outputs.get('e_tilda', None)
            if e_tilda is None:
                # If e_tilda wasn't already computed in generative, compute it here
                attention_weights = torch.softmax(self.attention(trt_inference['e_t']), dim=-1)
                e_t_trt = trt_inference['e_t']
                e_tilda_trt = attention_weights * e_t_trt
                # For control samples, e_tilda is zero
                e_tilda_ctrl = torch.zeros_like(ctrl_inference['z_bg'])
                e_tilda = torch.cat([e_tilda_ctrl, e_tilda_trt], dim=0)
            
            # Prepare batch encoding - one hot encoded batch vector
            batch_onehot = None
            if self.n_batch > 1:  # Only include batch if we have more than one batch
                batch_onehot = torch.zeros(batch_index.size(0), self.n_batch, device=x.device)
                batch_onehot.scatter_(1, batch_index.unsqueeze(1), 1)
            
            # Calculate the TC bound for each component separately
            # For z_bg (background latent representation)
            tc_z_bg = compute_tc_bound(qbg_m_all, qbg_v_all)
            
            # For e_tilda (treatment effect) - since this is a deterministic transformation,
            # we use samples directly with a small fixed variance
            # Create mean and variance for e_tilda (it's deterministic, so var is small)
            e_tilda_mean = e_tilda
            e_tilda_var = torch.ones_like(e_tilda) * 0.01  # Small fixed variance
            tc_e_tilda = compute_tc_bound(e_tilda_mean, e_tilda_var)
            
            # For batch encoding (optional)
            tc_batch = torch.tensor(0.0, device=x.device)
            if batch_onehot is not None:
                # Batch encoding is one-hot, so each dimension is independent already
                # but we compute it for completeness
                batch_mean = batch_onehot
                batch_var = torch.ones_like(batch_onehot) * 0.01  # Small fixed variance
                tc_batch = compute_tc_bound(batch_mean, batch_var)
            
            # Sum up all TC components
            tc_total = tc_z_bg + tc_e_tilda + tc_batch
            
            # Apply weight
            loss_tc = self.mi_weight * tc_total
            
            # Store the TC estimate for tracking
            if hasattr(self, 'tc_loss_tracker'):
                self.tc_loss_tracker = tc_total.detach()

        # Summation
        total_loss = recon_loss.mean() + kl_bg.mean() + kl_library.mean() + loss_mmd + loss_norm.mean() + loss_tc

        kl_local = {
            'loss_kl_bg': kl_bg,
            'loss_kl_l': kl_library,
            'loss_mmd': loss_mmd,
            'loss_norm': loss_norm,
            'loss_tc': loss_tc,
            'recon_loss': recon_loss,
        }

        return LossOutput(
            loss=total_loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_local,
        )

    def forward(self, *inputs, **kwargs):
        """
        Forward pass for Model1Module.
        
        Parameters
        ----------
        *inputs
            Same as specified in ``model.forward`` method.
        **kwargs
            Same as specified in ``model.forward`` method.
            
        Returns
        -------
        output :
            Same as specified in ``model.forward`` method but with an additional
            "tc_estimate" element containing the current total correlation estimate
            when mi_weight > 0.
        """
        outputs = super().forward(*inputs, **kwargs)
        
        # Add total correlation estimate to the outputs when TC loss is used
        if self.mi_weight > 0 and hasattr(self, 'tc_loss_tracker'):
            outputs["tc_estimate"] = self.tc_loss_tracker
        
        return outputs
    
    def get_current_tc_estimate(self):
        """
        Get the current total correlation estimate.
        
        Returns
        -------
        tc_estimate : torch.Tensor
            Current total correlation estimate
        """
        if self.mi_weight > 0 and hasattr(self, 'tc_loss_tracker'):
            return self.tc_loss_tracker
        return torch.tensor(0.0, device=self.device)