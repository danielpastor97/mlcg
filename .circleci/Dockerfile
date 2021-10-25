FROM continuumio/miniconda3

RUN apt update -y; \
    apt install build-essential -y

RUN conda create -n mlcg python=3.8

RUN conda install -n mlcg pyg=2.0.1 -c pyg -c conda-forge

