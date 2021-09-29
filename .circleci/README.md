# Docker image and CI
---
Here, we include a docker image based on `conda/miniconda3`
with additional PyTorch Geometric tools installed. If you update the docker file, be sure
to change the tag so that an image version change can be tracked. There is an additional,
currently unused docker file, `Dockerfile_CUDA`, that is based on the
`pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime` image. Because it includes additional
CUDA libraries for GPU usage, it is larger and is currently not used for CI.
