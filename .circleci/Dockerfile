FROM python:3.9.7-bullseye

RUN pip3 install scipy

ARG TORCH=1.10.0
ARG CUDA=cpu

RUN pip3 install torch==${TORCH} torchvision==0.11.1 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN pip3 install --upgrade --force-reinstall torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html; \
    pip3 install --upgrade --force-reinstall torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html; \
    pip3 install --upgrade --force-reinstall torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html; \
    pip3 install --upgrade --force-reinstall torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html; \
    pip3 install torch_geometric

RUN apt update -y; \
    apt install cmake -y

RUN pip3 install ipython pytorch-lightning[extra]
