# FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04
# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
# FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install APT tools and repos
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    apt-utils \
    software-properties-common

RUN apt-get update && apt-get upgrade -y

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install system tools (might want to check python and pybind version)
RUN apt-get install -y \
    git \
    net-tools \
    wget \
    curl \
    zip \
    unzip \
    patchelf \
    pybind11-dev \
    # python3-pip \
    python3-tk \
    tk \
    build-essential \
    software-properties-common

# #Install SOFA Dependencies
RUN apt install -y \
    gcc-11 \
    g++-11 \ 
    cmake \
    # qt5-default \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    libboost-all-dev \
    python3.12-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libglew-dev \
    zlib1g-dev \
    libeigen3-dev \
    libtinyxml2-dev

# # Set python3 and pip to Python 3.12 for consistency
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
#     update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.12 1

# Install pip for Python 3.12
RUN curl -L https://bootstrap.pypa.io/pip/get-pip.py -o /tmp/get-pip3.py

RUN python3.12 /tmp/get-pip3.py --break-system-packages

RUN python3.12 -m pip install --upgrade pip --break-system-packages
RUN python3.12 -m pip install numpy scipy pybind11 --break-system-packages
# RUN python3 -m pip install pybind11==2.9.1 --break-system-packages



RUN mkdir /opt/sofa
RUN git clone --depth 1 -b v24.12 https://github.com/sofa-framework/sofa.git /opt/sofa/src

RUN mkdir /opt/sofa/build


RUN cmake -D SOFA_FETCH_SOFAPYTHON3=True -D SOFA_FETCH_BEAMADAPTER=True -D SOFA_FETCH_MULTITHREADING=True -D PYTHON_VERSION=3.12 -D pybind11_DIR=/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11/ -D PYTHON_EXECUTABLE=/usr/bin/python3.12 -G "CodeBlocks - Unix Makefiles" -S /opt/sofa/src -B /opt/sofa/build
RUN cmake -D PLUGIN_SOFAPYTHON3=True -D PLUGIN_BEAMADAPTER=True -D PLUGIN_MULTITHREADING=True -S /opt/sofa/src -B /opt/sofa/build

RUN make -j 4 --directory /opt/sofa/build
RUN make install --directory /opt/sofa/build

# ENV PYTHONPATH="/opt/sofa/build/install/plugins/SofaPython3/lib/python3/site-packages"
ENV PYTHONPATH="/opt/sofa/build/lib/python3/site-packages"
ENV SOFA_ROOT="/opt/sofa/build/install"
# /:/opt/eve_training/:$PYTHONPATH
# RUN python3 -m pip install nvidia-cuda-cupti-cu11==11.8.87
# RUN python3 -m pip install nvidia-cublas-cu11==11.11.3.6
# RUN python3 -m pip install nvidia_cudnn_cu12==9.1.0.70
# RUN python3 -m pip install nvidia-cufft-cu11==10.9.0.58
# RUN python3 -m pip install nvidia-curand-cu11==10.3.0.86
# RUN python3 -m pip install nvidia-cusolver-cu11==11.4.1.48
# RUN python3 -m pip install nvidia-cusparse-cu11==11.7.5.86
# RUN python3 -m pip install nvidia-nccl-cu11==2.21.5

# RUN python3 -m pip  install nvidia-cuda-nvrtc-cu11==11.8.89
# RUN python3 -m pip  install nvidia-cuda-runtime-cu11==11.8.89
# RUN python3 -m pip  install nvidia-nvtx-cu11==11.8.86

RUN python3 -m pip install nvidia-cuda-cupti-cu12==12.4.127 --break-system-packages
RUN python3 -m pip install nvidia-cublas-cu12==12.4.5.8 --break-system-packages
# RUN python3 -m pip install nvidia-cuda-nvcc-cu12==12.1.105
# RUN python3 -m pip install nvidia-cuda-nvrtc-cu12==12.4.127
# RUN python3 -m pip install nvidia-cuda-runtime-cu12==12.4.127
RUN python3 -m pip install nvidia-cudnn-cu12==9.10.1.4 --break-system-packages
RUN python3 -m pip install nvidia-cufft-cu12==11.2.1.3 --break-system-packages
RUN python3 -m pip install nvidia-curand-cu12==10.3.5.147 --break-system-packages
RUN python3 -m pip install nvidia-cusolver-cu12==11.6.1.9 --break-system-packages
RUN python3 -m pip install nvidia-cusparse-cu12==12.3.1.170 --break-system-packages
RUN python3 -m pip install nvidia-nccl-cu12==2.21.5 --break-system-packages
# RUN python3 -m pip install nvidia-nvjitlink-cu12==12.4.127
# RUN python3 -m pip install nvidia-nvtx-cu12==12.4.127


RUN python3 -m pip install torch --break-system-packages
RUN python3 -m pip install torchvision torchaudio --break-system-packages
RUN python3 -m pip install scipy scikit-image pyvista PyOpenGL PyOpenGL_accelerate pygame matplotlib pillow opencv-python meshio pyyaml optuna gymnasium transforms3d attrdict ujson omegaconf hydra-core termcolor tensordict torchrl stable-baselines3 --break-system-packages 

# RUN python3 -m pip install hydra-core termcolor tensordict torchrl wandb pandas

RUN apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Europe/London /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata


COPY . /opt/eve_training
RUN python3 -m pip install /opt/eve_training/eve --break-system-packages
RUN python3 -m pip install /opt/eve_training/eve_bench --break-system-packages
# RUN python3 -m pip install /opt/eve_training/eve_rl
# RUN python3 -m pip install /opt/eve_training

COPY ./eve/eve/intervention/simulation/util/unit_sphere.stl /usr/local/lib/python3.12/dist-packages/eve/intervention/simulation/util/

# If use linux and run docker on local machine, add the below two lines and use the command
# sudo docker run --rm -it --env DISPLAY=$DISPLAY --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -v /tmp/.X11-unix:/tmp/.X11-unix steve_test:v0 python3 eve/examples/function_check.py
# ENV XDG_RUNTIME_DIR=/tmp/runtime
# RUN mkdir -p /tmp/runtime

# If don't want default buffered output for testing, uncomment below line
# ENV PYTHONUNBUFFERED=1

WORKDIR /opt/eve_training
# WORKDIR /opt/RL
# WORKDIR /opt/RL-image

# build
# sudo docker build -t steve_test:v0 .
# test
# sudo docker run --rm -it steve_test:v0 python3 eve/examples/function_check.py

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
 
# bash build.sh 

# runai submit --name eve-training1 --image aicregistry:5000/hrobertshaw:sofa-docker --run-as-user --gpu 2 --large-shm --project hrobertshaw -v /nfs:/nfs --command -- python3 /nfs/home/hrobertshaw/eve_training/eve_training/eve_paper/neurovascular/full/train_mesh_ben_two_device.py -nw 20 -d cuda -n test -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1

# runai submit --name ppo-big --image aicregistry:5000/agranados:sofa-docker --run-as-user --gpu 1 --cpu 6 --large-shm --project agranados -v /nfs:/nfs --command -- python3 /nfs/home/agranados/projects/RL/Scripts/batch_eleven/evaluate_PPO_model11_archvariety.py

# runai submit --name a2c-image --image aicregistry:5000/agranados:steve-img-docker --run-as-user --gpu 1 --cpu 6 --large-shm --project agranados -v /nfs:/nfs --command -- python /nfs/home/agranados/projects/RL-Image/stEVE_image/eve_bench/run/train_A2C_archvariety_image_v2.py

# runai describe job a2c-image -p agranados

# runai delete job a2c-image

# runai logs a2c-image



