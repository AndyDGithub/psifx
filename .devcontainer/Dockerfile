FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ARG USERNAME=psifx
ARG USER_UID=1000
ARG USER_GID=1000

# System dependencies: ffmpeg for audio/video I/O, the OpenFace build chain
# (cmake/boost/dlib/openblas/opencv-dev -- Ubuntu 22.04 ships OpenCV 4.5.x,
# which satisfies OpenFace's CMakeLists requirement of OpenCV 4.x), and the
# runtime libs opencv-python needs headlessly (libgl1, libsm6, ...).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        libboost-all-dev \
        libdlib-dev \
        libgl1 \
        libglib2.0-0 \
        libopenblas-dev \
        libopencv-dev \
        libsm6 \
        libsqlite3-dev \
        libxext6 \
        libxrender1 \
        nano \
        ninja-build \
        sudo \
        unzip \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

ENV CONDA_DIR=/home/$USERNAME/conda
ENV OPENFACE_DIR=/home/$USERNAME/openface
# The psifx-env bin dir goes first so `python`/`pip`/`psifx` resolve to it
# directly via plain PATH lookup -- no `conda activate` required. Ubuntu's
# default .bashrc bails out early for non-interactive shells (its "case $-
# in *i*)" guard), so relying on activation there breaks `docker run ...
# bash -c`, devcontainer.json's postCreateCommand, and CI, which are all
# non-interactive.
ENV PATH="$CONDA_DIR/envs/psifx-env/bin:$CONDA_DIR/bin:$CONDA_DIR/condabin:$OPENFACE_DIR/build/bin:$PATH"

# Conda environment (mirrors the README's local setup instructions).
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR" && \
    rm /tmp/miniconda.sh && \
    conda config --set always_yes yes && \
    conda config --set channel_priority strict && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n psifx-env python=$PYTHON_VERSION pip && \
    conda clean -afy

# OpenFace, non-interactively (same flags as the project's root Dockerfile).
RUN wget -q https://raw.githubusercontent.com/GuillaumeRochette/OpenFace/master/install.py && \
    python3 install.py \
        --license_accepted \
        --install_path "$OPENFACE_DIR" \
        --overwrite_install \
        --minimal_install \
        --no-add_to_login_shell && \
    rm install.py

# Build-time copy so `pip install -e .` has a package to resolve; at runtime
# devcontainer.json bind-mounts the live workspace over this same path, so
# edits on the host are picked up immediately without rebuilding the image.
COPY --chown=$USERNAME:$USERNAME . /home/$USERNAME/psifx
WORKDIR /home/$USERNAME/psifx

# `source` is a bash builtin, not available under Docker's default /bin/sh.
SHELL ["/bin/bash", "-c"]

RUN source "$CONDA_DIR/etc/profile.d/conda.sh" && \
    conda activate psifx-env && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e '.[dev]'

# Ollama CLI, for the 'ollama' text/llm provider. The daemon isn't started
# here (containers don't run background services at build time) -- run
# `ollama serve &` then `ollama pull <model>` once inside the container.
RUN curl -fsSL https://ollama.com/install.sh | sh

# `conda activate` needs the shell hook sourced first (this is what
# `conda init bash` would otherwise add) -- plain PATH isn't enough.
RUN echo "source \"$CONDA_DIR/etc/profile.d/conda.sh\"" >> /home/$USERNAME/.bashrc && \
    echo "conda activate psifx-env" >> /home/$USERNAME/.bashrc

CMD ["/bin/bash"]
