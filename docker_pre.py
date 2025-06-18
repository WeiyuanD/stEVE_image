# FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04
# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install APT tools and repos
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    apt-utils \
    software-properties-common

RUN apt-get update && apt-get upgrade -y

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install system tools
RUN apt-get install -y \
    git \
    net-tools \
    wget \
    curl \
    zip \
    unzip \
    patchelf \
    pybind11-dev \
    python3-pip

# #Install SOFA Dependencies
RUN apt install -y \
    gcc-10 \
    cmake \
    # qt5-default \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    libboost-all-dev \
    python3-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libglew-dev \
    zlib1g-dev \
    libeigen3-dev \
    libtinyxml2-dev

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy scipy pybind11

RUN mkdir /opt/sofa
RUN git clone --depth 1 -b v24.06 https://github.com/sofa-framework/sofa.git /opt/sofa/src

RUN mkdir /opt/sofa/build


RUN cmake -D SOFA_FETCH_SOFAPYTHON3=True -D SOFA_FETCH_BEAMADAPTER=True -D SOFA_FETCH_MULTITHREADING=True -D PYTHON_VERSION=3.10 -D pybind11_DIR=/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11/ -D PYTHON_EXECUTABLE=/usr/bin/python3.10 -G "CodeBlocks - Unix Makefiles" -S /opt/sofa/src -B /opt/sofa/build
RUN cmake -D PLUGIN_SOFAPYTHON3=True -D PLUGIN_BEAMADAPTER=True -D PLUGIN_MULTITHREADING=True -S /opt/sofa/src -B /opt/sofa/build

RUN make -j 4 --directory /opt/sofa/build
RUN make install --directory /opt/sofa/build

ENV PYTHONPATH="/opt/sofa/build/install/plugins/SofaPython3/lib/python3/site-packages/:/opt/eve_training/:$PYTHONPATH"
ENV SOFA_ROOT="/opt/sofa/build/install"

# RUN python3 -m pip install nvidia-cuda-cupti-cu11==11.7.101
# RUN python3 -m pip install nvidia-cublas-cu11==11.10.3.66
# RUN python3 -m pip install nvidia-cudnn-cu11==8.5.0.96
# RUN python3 -m pip install nvidia-cufft-cu11==10.9.0.58
# RUN python3 -m pip install nvidia-curand-cu11==10.2.10.91
# RUN python3 -m pip install nvidia-cusolver-cu11==11.4.0.1
# RUN python3 -m pip install nvidia-cusparse-cu11==11.7.4.91
# RUN python3 -m pip install nvidia-nccl-cu11==2.14.3

RUN python3 -m pip install nvidia-cuda-cupti-cu12==12.3.52
RUN python3 -m pip install nvidia-cublas-cu12==12.1.0.26
RUN python3 -m pip install nvidia-cudnn-cu12==8.9.6.50
RUN python3 -m pip install nvidia-cufft-cu12==11.0.2.4
RUN python3 -m pip install nvidia-curand-cu12==10.3.2.56
RUN python3 -m pip install nvidia-cusolver-cu12==11.4.4.55
RUN python3 -m pip install nvidia-cusparse-cu12==12.0.2.55
RUN python3 -m pip install nvidia-nccl-cu12==2.18.3


RUN python3 -m pip install torch
RUN python3 -m pip install torchvision torchaudio
RUN python3 -m pip install scipy scikit-image pyvista PyOpenGL PyOpenGL_accelerate pygame matplotlib pillow opencv-python meshio pyyaml optuna gymnasium transforms3d attrdict ujson omegaconf hydra-core termcolor tensordict torchrl
RUN python3 -m pip install hydra-core termcolor tensordict torchrl wandb pandas

RUN apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Europe/London /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata


COPY . /opt/eve_training
RUN python3 -m pip install /opt/eve_training/eve
# RUN python3 -m pip install /opt/eve_training/eve_bench
# RUN python3 -m pip install /opt/eve_training/eve_rl
# RUN python3 -m pip install /opt/eve_training

WORKDIR /opt/eve_training

# docker buildx build --platform=linux/amd64 -t 10.15.17.136:5555/lnk/eve_training -f ./dockerfile .
# docker push 10.15.17.136:5555/lnk/eve_training
# docker pull 10.15.17.136:5555/lnk/eve_training
 
# docker buildx build --platform=linux/amd64 -t 10.15.17.136:5555/lnk/eve_training -f ./dockerfile . && docker push 10.15.17.136:5555/lnk/eve_training
 
# docker container stop $(docker container ls --filter label=lnk_training --quiet) ; docker pull 10.15.17.136:5555/lnk/eve_training
 
# docker image rm $(docker image ls --filter reference="registry.gitlab.cc-asp.fraunhofer.de/stacie/ma_projects/lnk_training" --filter "dangling=true" --quiet)
 
# .73
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/eve_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/neurovascular/aorta/gw_only/train_vmr_94.py -d cuda -nw 55 -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1 -n arch_vmr_94_lstm
 
# .197
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/eve_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/neurovascular/aorta/gw_only/train_vmr_94.py -d cuda -nw 29 -lr 0.00021989352630306626 --hidden 500 900 900 900 900 -en 0 -el 0 -n arch_vmr_94_ff
# -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1
 
# .223
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/eve_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/neurovascular/aorta/gw_only/train_vmr_94.py -d cuda -nw 20 -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1 -n arch_vmr_94_lstm
 
# .238
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/eve_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/neurovascular/aorta/gw_only/train_vmr_94.py -d cuda -nw 20 -lr 0.00021989352630306626 --hidden 500 900 900 900 900 -en 0 -el 0 -n arch_vmr_94_ff
 
# .142
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/eve_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/neurovascular/aorta/gw_only/train_1669028594.py -d cuda -nw 20 -lr 0.00021989352630306626 --hidden 500 900 900 900 900 -en 0 -el 0 -n arch_1669028594_ff
 
#.17.164
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/eve_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/neurovascular/aorta/gw_only/optuna_hyperparam.py -d cuda -nw 14 -n archgen_optuna
 
# pamb-dlp
# docker run --label=lnk_training --gpus all --rm --mount type=bind,source=$PWD/results,target=/opt/lnk_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/neurovascular/aorta/gw_only/typeI_optuna_hyerparam.py -d cuda -nw 35 -n typeI_archgen_optuna
# docker run --label=lnk_training --gpus all --rm --mount type=bind,source=$PWD/results,target=/opt/lnk_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/neurovascular/aorta/gw_only/optuna_hyperparam.py -d cuda:1 -nw 35 -n typeI_archgen_optuna
 
 
# runai submit --name eve-training1 --image aicregistry:5000/hrobertshaw:sofa-docker --run-as-user --gpu 2 --large-shm --project hrobertshaw -v /nfs:/nfs --command -- python3 /nfs/home/hrobertshaw/eve_training/eve_training/eve_paper/neurovascular/full/train_mesh_ben_two_device.py -nw 20 -d cuda -n test -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1

