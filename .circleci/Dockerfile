FROM anibali/pytorch:1.11.0-nocuda

RUN conda install pyg -c pyg
RUN pip3 install pytorch-lightning==1.6.1
RUN pip install jsonargparse[signatures]==4.7.2
# RUN conda install cmake

RUN pip3 install git+https://github.com/ACEsuit/mace.git@develop