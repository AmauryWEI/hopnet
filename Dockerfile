FROM python:3.11.12-bookworm

# Environment variables
ENV TZ="Europe/Zurich"
ENV PIP_DEFAULT_TIMEOUT=120
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Perform any task requiring root privileges
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

RUN apt update && \
    apt install -y htop tzdata wget ssh git libssl-dev tmux vim ffmpeg libsm6 libxext6 libgl1-mesa-glx libosmesa6 && \
    apt autoremove -y && \
    apt autoclean -y && \
    rm -rf /var/lib/apt/lists/*

# Update Python package builder dependencies
RUN pip install -U pip wheel setuptools

# Install the TopoModelX and TopoNetX packages (included in TopoModelX)
RUN git clone https://github.com/pyt-team/TopoModelX
RUN cd TopoModelX && \
    git checkout 226776822925b0984b1c5dbd097234d7fcbc274e && \
    pip install -e '.[all]'

# Install common dependencies for HOPNet and FIGNet
RUN pip install numpy==1.26.4
RUN pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu124
RUN pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
RUN pip install matplotlib wandb trimesh==4.6.10 scipy tqdm tensorboard

# Install dependencies of FIGNet
RUN pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
RUN pip install eigenpy==3.10.3 coal==3.0.1
RUN pip install robosuite==1.4.1 dacite==1.8.1 pyyaml moviepy pillow==9.5.0 pyrender

# Install dependencies of HOPNet
RUN pip install rtree plotly kaleido==0.1.0

CMD sleep infinity
