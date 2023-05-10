import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset

from ..utils import make_splits

from typing import List, Union

def make_subsample_splits(
    L: int, 
    dataset_prob: torch.Tensor,
    val_ratio: float,
    test_ratio: float,
    splits_fn: str,
) -> List[np.ndarray]:
    
    ## Make array of length L that keeps track of all indices
    all_indices = np.arange(L)
    
    ## Make array of decision probabilities, drawn from a uniform distribution
    prob_sample_inclusion = np.random.uniform(
        size=all_indices.shape
        )
    
    ## Compare sample_inclusion probablities to dataset probabilities array
    ## If p_sample <= p_dataset, include sample
    ## E.g., p_sample = 0.3, p_dataset = 0.4, include sample since sample
    ## has 40% chance of being included
    chosen_ind = all_indices[ prob_sample_inclusion <= dataset_prob ]
    
    ## Make splits of length of chosen_ind
    idx_train, idx_val, idx_test = make_splits(
            len(chosen_ind),
            val_ratio,
            test_ratio,
            splits=splits_fn,
        )
    
    ## Return train, val, test indices, which are all part of chosen_ind
    return chosen_ind[idx_train], chosen_ind[idx_val], chosen_ind[idx_test]
    
    
class SubsampleDataModule(pl.LightningDataModule):

    def __init__(
            self,
            dataset: InMemoryDataset,
            log_dir: str,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            splits: str = None,
            batch_size: int = 512,
            inference_batch_size: int = 64,
            num_workers: int = 1,
            
            ):

        super(SubsampleDataModule, self).__init__()
        # self.save_hyperparameters()

        # assume dataset is similart to torch_geometric.dataset.Dataset
        self.dataset_init_kwargs = {
            "root": dataset.root,
            "transform": dataset.transform,
            "pre_transform": dataset.pre_transform,
            "pre_filter": dataset.pre_filter,
        }
        self.dataset_cls = type(dataset)
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.log_dir = log_dir
        self.splits_fn = os.path.join(self.log_dir, "splits.npz")


    def prepare_data(self):
        """No need to prepare data -- assume dataset has been loaded already
        """
        pass

    def setup(self, stage=None):
        dataset = self.load_dataset()

        ### TO WRITE THIS FUNCTION
        self.idx_train, self.idx_val, self.idx_test = make_subsample_splits(
            len(dataset),
            dataset.weights,
            self.val_ratio,
            self.test_ratio,
            splits=self.splits_fn,
        )

        self.train_dataset = [dataset[ii] for ii in self.idx_train]
        self.val_dataset = [dataset[ii] for ii in self.idx_val]
        self.test_dataset = [dataset[ii] for ii in self.idx_test]
    
    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    def _get_dataloader(self, dataset, stage):
        if stage == "train":
            batch_size = self.batch_size
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.inference_batch_size
            shuffle = False

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )