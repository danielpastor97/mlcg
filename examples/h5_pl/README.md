# Example for HDF5-PyTorch-Lightning-Parallel dataset construction and training-validation

## Features
- Parallelized training on multiple GPUs with distributed data parallel (DDP) and low memory footprint
- Balanced batch sized with accurate data loading proportions
- Bundled datasets in single (or several) HDF5 files

## Additional dependencies
- h5py (install can be done with `conda install -c conda-forge h5py`)

## Guidelines
1. Run `save_h5.py` to create the dataset (Please check the CLI helper. The full-sized combined datasets require around 60GB disk space.)
2. Check out `train_h5_1_10.yaml` and `partition_settings.yaml` and adjust the settings according to the actual needs, e.g., the training set composition and number of GPUs.
3. Run `python ../../scripts/mlcg-train_h5.py fit --config train_h5_1_10.yaml` for model training.

## Note
1. The detailed data structure readme can be found at the header of `mlcg/datasets/h5_dataset.py`.
2. The pytorch-lightning script `scripts/mlcg-train_h5.py` may serve as an example for constructing training pipelines without pytorch-lightning.
