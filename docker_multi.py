# ================================
# Stage 1: Build Steve + Sofa
# ================================
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS steve-builder

ENV DEBIAN_FRONTEND=noninteractive

# Base setup
RUN apt-get update && apt-get install -y \
    git wget curl zip unzip net-tools patchelf \
    build-essential cmake software-properties-common \
    python3.12 python3.12-dev python3-tk tk \
    pybind11-dev \
    qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
    libboost-all-dev libpng-dev libjpeg-dev libtiff-dev \
    libglew-dev zlib1g-dev libeigen3-dev libtinyxml2-dev \
    tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/London /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# Install pip for Python 3.12
RUN curl -L https://bootstrap.pypa.io/pip/get-pip.py -o /tmp/get-pip3.py && \
    python3.12 /tmp/get-pip3.py --break-system-packages

# Python packages
RUN python3.12 -m pip install --break-system-packages \
    numpy scipy pybind11 torch torchvision torchaudio \
    scikit-image pyvista PyOpenGL PyOpenGL_accelerate pygame \
    matplotlib pillow opencv-python meshio pyyaml optuna \
    gymnasium transforms3d attrdict ujson omegaconf \
    hydra-core termcolor tensordict torchrl stable-baselines3

# NVIDIA CUDA libs
RUN python3.12 -m pip install --break-system-packages \
    nvidia-cuda-cupti-cu12==12.4.127 \
    nvidia-cublas-cu12==12.4.5.8 \
    nvidia-cudnn-cu12==9.10.1.4 \
    nvidia-cufft-cu12==11.2.1.3 \
    nvidia-curand-cu12==10.3.5.147 \
    nvidia-cusolver-cu12==11.6.1.9 \
    nvidia-cusparse-cu12==12.3.1.170 \
    nvidia-nccl-cu12==2.21.5

# Build SOFA
RUN git clone --depth 1 -b v24.12 https://github.com/sofa-framework/sofa.git /opt/sofa/src && \
    mkdir /opt/sofa/build && \
    cmake -D SOFA_FETCH_SOFAPYTHON3=True \
          -D SOFA_FETCH_BEAMADAPTER=True \
          -D SOFA_FETCH_MULTITHREADING=True \
          -D PYTHON_VERSION=3.12 \
          -D pybind11_DIR=/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11/ \
          -D PYTHON_EXECUTABLE=/usr/bin/python3.12 \
          -S /opt/sofa/src -B /opt/sofa/build && \
    cmake -D PLUGIN_SOFAPYTHON3=True \
          -D PLUGIN_BEAMADAPTER=True \
          -D PLUGIN_MULTITHREADING=True \
          -S /opt/sofa/src -B /opt/sofa/build && \
    make -j4 -C /opt/sofa/build && \
    make install -C /opt/sofa/build

# Copy your eve_training code
COPY ./ /opt/eve_training

# Install eve packages
RUN python3.12 -m pip install --break-system-packages \
    /opt/eve_training/eve \
    /opt/eve_training/eve_bench

# Copy STL asset
RUN cp /opt/eve_training/eve/eve/intervention/simulation/util/unit_sphere.stl /usr/local/lib/python3.12/dist-packages/eve/intervention/simulation/util/


# ================================
# Stage 2: Final Runtime Image
# ================================
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS final

# Install system packages (lightweight)
RUN apt-get update && apt-get install -y \
    libgl1 libxrender1 libsm6 libxext6 libglib2.0-0 python3.12 python3.12-dev python3-tk tk tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/London /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# Copy SOFA and eve environment from builder
COPY --from=steve-builder /opt/sofa /opt/sofa
COPY --from=steve-builder /opt/eve_training /opt/eve_training
COPY --from=steve-builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=steve-builder /usr/local/bin/python3.12 /usr/local/bin/python3.12
COPY --from=steve-builder /usr/local/bin/pip3 /usr/local/bin/pip3
COPY --from=steve-builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages

# Set environment variables
ENV PYTHONPATH="/opt/sofa/build/lib/python3/site-packages:/opt/eve_training:$PYTHONPATH"
ENV SOFA_ROOT="/opt/sofa/build/install"
ENV PATH="/usr/local/bin:$PATH"

# (Optional) Install additional RL dependencies or data loaders
# RUN pip install -r /opt/eve_training/requirements.txt

WORKDIR /opt/eve_training
CMD ["python3.12", "main.py"]
