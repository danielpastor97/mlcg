# mlcg-tools

Some special installs that are one well handled by requirements.txt:

```
conda install pyg=2.0.1 -c pyg -c conda-forge

pip install git+https://github.com/PyTorchLightning/pytorch-lightning.git

```

# Doc

## doc dependencies

```
pip install sphinx sphinx_rtd_theme
```

## build doc

```
cd docs
sphinx-build -b html source build
```