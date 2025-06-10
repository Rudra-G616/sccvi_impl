from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import torch
from scvi.train import TrainingPlan, TrainRunner
# Use relative imports instead of absolute imports starting with 'src'
from ...data.dataloaders.data_splitting import scCausalVIDataSplitter


class Model1TrainingMixin:
    def train(
            self,
            group_indices_list: List[np.array],
            max_epochs: Optional[int] = None,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = 0.1,
            batch_size: int = 128,
            early_stopping: bool = False,
            plan_kwargs: Optional[dict] = None,
            **trainer_kwargs,
    ) -> TrainRunner:
        """
        Train a Model1 model.

        Args:
        ----
            group_indices_list: List of index arrays for different groups or conditions in `adata`.
            max_epochs: Number of passes through the dataset. If `None`, default to
                `np.min([round((20000 / n_cells) * 400), 400])`.
            use_gpu: Use default GPU if available (if `None` or `True`), or index of
                GPU to use (if `int`), or name of GPU (if `str`, e.g., `"cuda:0"`),
                or use CPU (if `False`).
            train_size: Size of training set in the range [0.0, 1.0].
            validation_size: Size of the validation set. If `None`, default to
                `1 - train_size`. If `train_size + validation_size < 1`, the remaining
                cells belong to the test set.
            batch_size: Mini-batch size to use during training.
            early_stopping: Perform early stopping. Additional arguments can be passed
                in `**kwargs`. See :class:`~scvi.train.Trainer` for further options.
            plan_kwargs: Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword
                arguments passed to `train()` will overwrite values present
                in `plan_kwargs`, when appropriate.
            **trainer_kwargs: Other keyword args for :class:`~scvi.train.Trainer`.

        Returns
        -------
            TrainRunner: The training runner object used for training.
        """
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = scCausalVIDataSplitter(
            self.adata_manager,
            group_indices_list,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            accelerator="gpu" if use_gpu else "cpu",  # Match the accelerator parameter format with TrainRunner
        )

        training_plan = TrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator="gpu" if use_gpu else "cpu",
            **trainer_kwargs,
        )
        return runner()
        
    def train_with_multiple_datasets(
            self,
            adatas_list: List,
            max_epochs: Optional[int] = None,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = 0.1,
            batch_size: int = 128,
            early_stopping: bool = False,
            plan_kwargs: Optional[dict] = None,
            **trainer_kwargs,
    ) -> TrainRunner:
        """
        Train a Model1 model with multiple datasets.

        Args:
        ----
            adatas_list: List of AnnData objects for training.
            max_epochs: Number of passes through the dataset. If `None`, default to
                `np.min([round((20000 / n_cells) * 400), 400])`.
            use_gpu: Use default GPU if available (if `None` or `True`), or index of
                GPU to use (if `int`), or name of GPU (if `str`, e.g., `"cuda:0"`),
                or use CPU (if `False`).
            train_size: Size of training set in the range [0.0, 1.0].
            validation_size: Size of the validation set. If `None`, default to
                `1 - train_size`. If `train_size + validation_size < 1`, the remaining
                cells belong to the test set.
            batch_size: Mini-batch size to use during training.
            early_stopping: Perform early stopping. Additional arguments can be passed
                in `**kwargs`. See :class:`~scvi.train.Trainer` for further options.
            plan_kwargs: Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword
                arguments passed to `train()` will overwrite values present
                in `plan_kwargs`, when appropriate.
            **trainer_kwargs: Other keyword args for :class:`~scvi.train.Trainer`.

        Returns
        -------
            TrainRunner: The training runner object used for training.
        """
        # Calculate total number of cells across all datasets
        total_cells = sum(adata.n_obs for adata in adatas_list)
        
        if max_epochs is None:
            max_epochs = np.min([round((20000 / total_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        
        # Implement custom logic for handling multiple datasets
        # This would require creating a custom data splitter or adapting the existing one
        import scanpy as sc
        
        # Check that all AnnData objects have compatible features
        var_names = adatas_list[0].var_names
        for i, adata in enumerate(adatas_list[1:], 1):
            if not var_names.equals(adata.var_names):
                raise ValueError(f"AnnData object at index {i} has different features compared to the first dataset. "
                               "All datasets must have the same features.")
        
        # Concatenate all datasets with proper batch annotation
        combined_adata = sc.concat(
            adatas_list, 
            label="dataset_batch",  # Create a column indicating the source dataset
            keys=[f"dataset_{i}" for i in range(len(adatas_list))]
        )
        
        # Create group indices for the combined dataset
        group_indices_list = []
        cumulative_index = 0
        for adata in adatas_list:
            n_cells = adata.n_obs
            group_indices_list.append(np.arange(cumulative_index, cumulative_index + n_cells))
            cumulative_index += n_cells
            
        # Use the standard training method with the combined dataset
        return self.train(
            group_indices_list=group_indices_list,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            early_stopping=early_stopping,
            plan_kwargs=plan_kwargs,
            **trainer_kwargs
        )