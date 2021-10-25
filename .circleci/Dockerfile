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


# SHELL ["/bin/bash", "-c"]

# ARG username=dev
# ARG condaenv=mlcg

# RUN conda init bash

# RUN conda install python=3.8
# # RUN conda activate ${condaenv}
# RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
# RUN conda install pyg -c pyg -c conda-forge

# RUN apt update -y  && \
#     apt install build-essential  -y && \
#     apt install bash  -y

# ARG username=dev
# ARG condaenv=mlcg

# RUN useradd --create-home --uid 1000 --shell /bin/bash ${username}
# WORKDIR /home/${username}


# RUN mkdir /opt/conda/envs/${condaenv} /opt/conda/pkgs && \
#     chgrp ${username} /opt/conda/pkgs && \
#     chmod g+w /opt/conda/pkgs && \
#     touch /opt/conda/pkgs/urls.txt && \
#     chown -R ${username} /opt/conda/

# RUN chown 1000:1000 /opt/conda/pkgs/urls.txt

# SHELL ["/bin/bash", "-c"]

# USER ${username}

# RUN conda init bash

# RUN conda config --add channels conda-forge
# RUN conda config --add channels pytorch

# RUN conda create -y --name ${condaenv} python=3.8

# RUN source activate ${condaenv}

# RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch


# RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cpu.html