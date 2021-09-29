FROM conda/miniconda3

RUN apt update -y; \
    apt install build-essential -y

RUN conda install python=3.8

RUN TORCH=1.8.1; \
    CUDA=cpu; \
    pip install torch==$TORCH+$CUDA -f https://download.pytorch.org/whl/torch_stable.html; \
    pip install scipy; \
    pip install --upgrade --force-reinstall torch-scatter -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html; \
    pip install --upgrade --force-reinstall torch-sparse -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html; \
    pip install --upgrade --force-reinstall torch-cluster -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html; \
    pip install --upgrade --force-reinstall torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html; \
    pip install torch_geometric
