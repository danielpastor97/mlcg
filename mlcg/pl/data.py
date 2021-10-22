import os
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset

from ..utils import make_splits


class DataModule(pl.LightningDataModule):
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
        train_stride: int = 1,
        save_local_copy: bool = True,
    ) -> None:

        super(DataModule, self).__init__()
        # assume dataset is similart to torch_geometric.dataset.Dataset
        print(dataset)
        self.dataset_init_kwargs = {
            "root": dataset.root,
            "transform": dataset.transform,
            "pre_transform": dataset.pre_transform,
            "pre_filter": dataset.pre_filter,
        }
        self.dataset_cls = type(dataset)
        self.save_local_copy = save_local_copy

        if self.save_local_copy:
            self.dataset_root = os.path.join(log_dir, "dataset.pt")
            torch.save(dataset, self.dataset_root)

        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.log_dir = log_dir
        self.splits = splits
        self.train_stride = train_stride
        self.splits_fn = os.path.join(self.log_dir, "splits.npz")

    def load_dataset(self):
        if self.save_local_copy:
            dataset = torch.load(self.dataset_root)
        else:
            dataset = self.dataset_cls(**self.dataset_init_kwargs)
        return dataset

    def prepare_data(self):
        # make sure the dataset is downloaded
        dataset = self.load_dataset()

        # setup the train/test/val split
        idx_train, idx_val, idx_test = make_splits(
            len(dataset),
            self.val_ratio,
            self.test_ratio,
            filename=self.splits_fn,
            splits=self.splits,
        )

    def setup(self, stage=None):
        dataset = self.load_dataset()

        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(dataset),
            self.val_ratio,
            self.test_ratio,
            splits=self.splits_fn,
        )

        self.train_dataset = tuple(
            [dataset[ii] for ii in self.idx_train[:: self.train_stride]]
        )
        self.val_dataset = tuple([dataset[ii] for ii in self.idx_val])
        self.test_dataset = tuple([dataset[ii] for ii in self.idx_test])

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
