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


class MINEEstimator(torch.nn.Module):
    """
    Mutual Information Neural Estimator (MINE).
    
    Implements a neural network to estimate mutual information between two random variables
    based on the paper "Mutual Information Neural Estimation" by Belghazi et al.
    
    This implementation includes moving average updates for more stable training.
    """
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128, ma_rate: float = 0.01):
        """
        Args:
            x_dim: Dimension of the first random variable
            y_dim: Dimension of the second random variable
            hidden_dim: Dimension of the hidden layer
            ma_rate: Moving average rate for the exponential moving average
        """
        super(MINEEstimator, self).__init__()
        
        # Neural network for joint distribution estimation
        self.net = torch.nn.Sequential(
            torch.nn.Linear(x_dim + y_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        
        # Moving average for stable training
        self.ma_rate = ma_rate
        self.register_buffer("ma_et", torch.tensor(1.0))  # Use register_buffer for device management
        
    def forward(self, x, y, ma_update: bool = True):
        """
        Estimate mutual information between x and y.
        
        Args:
            x: First random variable (batch_size, x_dim)
            y: Second random variable (batch_size, y_dim)
            ma_update: Whether to update the moving average
            
        Returns:
            Mutual information estimate
        """
        batch_size = x.shape[0]
        
        # Ensure batch size is at least 2 for proper shuffling
        if batch_size < 2:
            # Return a small positive value for very small batches
            # to prevent numerical issues
            return torch.tensor(1e-8, device=x.device)
        
        # Joint distribution samples (x_i, y_i)
        joint_input = torch.cat([x, y], dim=1)
        t_joint = self.net(joint_input)
        
        # Marginal distribution samples (x_i, y_j) where i != j
        # Shuffle y to get samples from marginal distribution
        y_shuffled = y[torch.randperm(batch_size)]
        marginal_input = torch.cat([x, y_shuffled], dim=1)
        t_marginal = self.net(marginal_input)
        
        # Compute exp(t_marginal) with numerical stability
        max_t_marginal = torch.max(t_marginal)
        exp_t_marginal = torch.exp(t_marginal - max_t_marginal)
        exp_t_mean = torch.mean(exp_t_marginal)
        
        # Update moving average if in training mode
        if ma_update and self.training:
            self.ma_et = (1 - self.ma_rate) * self.ma_et + self.ma_rate * exp_t_mean.detach()
        
        # Use moving average for more stable training
        # MI = E_P[T] - log(E_Q[e^T])
        # Add the max value back for correct scaling
        mi_estimate = torch.mean(t_joint) - torch.log(self.ma_et + 1e-8) - max_t_marginal
        
        return mi_estimate
    
    def mi_loss(self, x, y):
        """
        Calculate the negative MI loss (for minimization)
        
        Args:
            x: First random variable (batch_size, x_dim)
            y: Second random variable (batch_size, y_dim)
            
        Returns:
            Negative mutual information estimate (for minimization)
        """
        return -self.forward(x, y)


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
        mi_weight: Weight for the mutual information loss between latent representation and treatment effect vector.
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
            Weight of mutual information loss between latent representation and treatment effect
            When > 0, a Mutual Information Neural Estimator (MINE) is used to estimate
            and minimize the mutual information, promoting better disentanglement
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
            
        # Initialize the Mutual Information Neural Estimator
        if self.mi_weight > 0:
            self.mine = MINEEstimator(x_dim=n_latent, y_dim=n_latent, hidden_dim=n_hidden, ma_rate=0.01)
            # Create a persistent optimizer for MINE
            self.mine_optimizer = torch.optim.Adam(self.mine.parameters(), lr=1e-4)
            # Register buffer to track MINE training statistics
            self.register_buffer("mine_loss_tracker", torch.tensor(0.0))

        cat_list = [n_batch]                    # Don't know what is the purpose of this, but it is in the original code.



        #######--------------------------- Encoder ---------------------------- ########
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

        ##### ----------------------------------------------------------------- #####

        #####--------------------- Treatment effect encoder ------------------------ #####
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

        ###### ----------------------------------------------------------------- #####

        ######----------------------- Attention layer------------------------------ ######

        # Attention layer to capture differential treatment effect patterns
        # The output of this layer is used to weight the treatment effect vector before
        # adding it to the latent representation
        self.attention = torch.nn.Linear(self.n_latent, 1)

        ######------------------------------------------------------------------ #####

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

        ######-----------------------Decoder---------------------------------- ######

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
            'e_tilda': e_tilda,  # Store for MI loss calculation
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

        # Mutual Information loss using MINE
        loss_mi = torch.tensor(0.0, device=x.device)
        if self.mi_weight > 0:
            # Calculate mutual information between latent representation (z_bg) and 
            # treatment effect vector (e_t) in treatment samples
            z_bg_trt = trt_inference['z_bg']
            e_t_trt = trt_inference['e_t']
            
            # Only compute if we have enough treatment samples (at least 2 for shuffling)
            if z_bg_trt.size(0) >= 2:  
                # First update the MINE network to better estimate MI
                # Detach z_bg and e_t to avoid influencing their gradients during MINE training
                # We're training MINE here to maximize the MI estimation accuracy
                if self.training:
                    # Use a persistent optimizer instead of creating a new one each time
                    if not hasattr(self, 'mine_optimizer'):
                        self.mine_optimizer = torch.optim.Adam(self.mine.parameters(), lr=1e-4)
                    
                    # Train MINE for a few steps to get better MI estimates
                    prev_loss = float('inf')
                    for i in range(5):  # Train MINE for up to 5 steps for each VAE update
                        self.mine_optimizer.zero_grad()
                        # Forward is MI estimate, negative for minimization
                        mine_train_loss = self.mine.mi_loss(z_bg_trt.detach(), e_t_trt.detach())
                        
                        # Early stopping if loss starts increasing
                        if i > 0 and mine_train_loss.item() > prev_loss * 1.2:  # 20% increase threshold
                            break
                            
                        prev_loss = mine_train_loss.item()
                        mine_train_loss.backward()
                        # Add gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.mine.parameters(), max_norm=1.0)
                        self.mine_optimizer.step()
                
                # Now use the trained MINE to estimate MI for the actual loss
                # Higher values of MI mean more dependence, so we negate for loss
                # Don't detach here so gradients flow to z_bg and e_t
                mine_loss = self.mine.mi_loss(z_bg_trt, e_t_trt)
                loss_mi = self.mi_weight * mine_loss

        # Summation
        total_loss = recon_loss.mean() + kl_bg.mean() + kl_library.mean() + loss_mmd + loss_norm.mean() + loss_mi

        kl_local = {
            'loss_kl_bg': kl_bg,
            'loss_kl_l': kl_library,
            'loss_mmd': loss_mmd,
            'loss_norm': loss_norm,
            'loss_mi': loss_mi,
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
            "mi_estimate" element containing the current mutual information estimate
            when mi_weight > 0.
        """
        outputs = super().forward(*inputs, **kwargs)
        
        # Add mutual information estimate to the outputs when MI loss is used
        if self.mi_weight > 0 and hasattr(self, 'mine'):
            # Get latest MI estimate from the MINE module (positive value)
            if hasattr(self, 'inference_outputs'):
                inference_outputs = self.inference_outputs
                if 'treatment' in inference_outputs and inference_outputs['treatment']['z_bg'].size(0) > 0:
                    z_bg_trt = inference_outputs['treatment']['z_bg']
                    e_t_trt = inference_outputs['treatment']['e_t']
                    with torch.no_grad():
                        mi_estimate = -self.mine.mi_loss(z_bg_trt.detach(), e_t_trt.detach())
                        # Update the tracker
                        self.mine_loss_tracker = mi_estimate.detach()
                    
                    # Add to outputs
                    outputs["mi_estimate"] = mi_estimate
        
        return outputs

    def estimate_mutual_information(self, z_bg, e_t):
        """
        Estimate mutual information between latent representation and treatment effect.
        
        Parameters
        ----------
        z_bg : torch.Tensor
            Latent representation tensor
        e_t : torch.Tensor
            Treatment effect tensor
            
        Returns
        -------
        mi_estimate : torch.Tensor
            Mutual information estimate (positive value, higher means more dependence)
        """
        # Check tensor dimensions and batch size
        if z_bg.size(0) != e_t.size(0) or z_bg.size(0) < 2:
            return torch.tensor(0.0, device=z_bg.device)
            
        if self.mi_weight > 0 and hasattr(self, 'mine'):
            try:
                with torch.no_grad():
                    # Use negative of mi_loss since mi_loss returns negative MI for minimization
                    mi_estimate = -self.mine.mi_loss(z_bg.detach(), e_t.detach())
                    return mi_estimate
            except Exception as e:
                print(f"Error in MI estimation: {e}")
                # Fallback to correlation-based estimate on error
                pass
                
        # Fallback to correlation-based MI estimate if MINE is not available or fails
        z_bg_centered = z_bg - z_bg.mean(dim=0, keepdim=True)
        e_t_centered = e_t - e_t.mean(dim=0, keepdim=True)
        corr_matrix = torch.mm(z_bg_centered.t(), e_t_centered) / (z_bg.size(0) - 1)
        return torch.norm(corr_matrix, p='fro')
    
    def get_current_mi_estimate(self):
        """
        Get the current mutual information estimate.
        
        Returns
        -------
        mi_estimate : torch.Tensor
            Current mutual information estimate
        """
        if self.mi_weight > 0 and hasattr(self, 'mine_loss_tracker'):
            return self.mine_loss_tracker
        return torch.tensor(0.0, device=self.device)