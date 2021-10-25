FROM continuumio/miniconda3

RUN apt update -y; \
    apt install build-essential -y

RUN conda install python=3.8

RUN conda install pyg=2.0.1 -c pyg -c conda-forge

