# DeepCAM README
For more detailed installation parameters, please refer to DeepCAM install guide

Perlmutter Repository: [hpc_results_v3.0](https://github.com/hpcgroup/hpc_results_v3.0)  
Frontier Repository: [hpc](https://github.com/hpcgroup/hpc)


## Perlmutter Setup

### Setup steps

#### 1. Environment Setup
- Load necessary modules:
    ```bash
    module load PrgEnv-nvidia cray-mpich cudatoolkit craype-accel-nvidia80 python/3.10 nccl cudnn cray-hdf5
    ```
- Create and activate a Python virtual environment:
    ```bash
    # Choose a suitable location for your environment, e.g., within your project directory
    # python -m venv /path/to/your/env/torch2.5-py3.10 --system-site-packages
    # source /path/to/your/env/torch2.5-py3.10/bin/activate

    # Example using a relative path:
    python -m venv torch2.5-py3.10 --system-site-packages
    source torch2.5-py3.10/bin/activate
    ```
- Upgrade pip:
    ```bash
    pip install --upgrade pip
    ```

#### 2. Install Core Dependencies
- Install PyTorch, torchvision, and torchaudio for CUDA 12.1:
    ```bash
    pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
- Install other required Python packages:
    ```bash
    pip install mpi4py pillow pandas pyyaml scipy h5py
    pip install cupy-cuda12x
    pip install pynvml
    pip install nvidia-dali-cuda120
    ```

#### 3. Install MLPerf and Logging Libraries
- Install MLPerf common utilities:
    ```bash
    pip install "git+https://github.com/NVIDIA/mlperf-common.git"
    ```
- Install MLPerf logging library:
    ```bash
    pip install "git+https://github.com/mlperf/logging.git"
    ```

#### 4. Install NVIDIA Apex
- Clone the Apex repository and checkout a specific commit:
    ```bash
    # Choose a location for repositories, e.g., ./repos
    # mkdir -p ./repos && cd ./repos
    git clone https://github.com/NVIDIA/apex
    cd apex
    git checkout 89cc215a
    ```
- Install Apex dependencies:
    ```bash
    pip install -r requirements.txt
    ```
- **Important:** Manually comment out line 32 in `setup.py` before proceeding.

- Set the C++ compiler and install Apex with CUDA extensions:
    ```bash
    export CXX=/usr/bin/g++
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    cd ..
    ```

#### 5. Install DeepCAM IO Helpers
- Navigate to the DeepCAM IO helpers directory within your cloned DeepCAM source code:
    ```bash
    # Adjust the path based on where you cloned the DeepCAM source code
    # cd /pscratch/sd/c/cunyang/deepcam/optimized_pm/HPE+LBNL/benchmarks/deepcam/implementations/deepcam-pytorch/io_helpers
    cd path/to/your/deepcam-pytorch/io_helpers
    ```
- Install the IO helpers package in editable mode:
    ```bash
    pip install -e .
    ```

## Frontier Setup

### Setup steps

#### 1. Pytorch Install
- Load modules
    ```bash
    module reset
    module load PrgEnv-gnu/8.5.0
    module load rocm/6.1.3
    module load craype-accel-amd-gfx90a
    module load cray-python/3.9.13.1
    module load cray-hdf5-parallel/1.12.2.9
    ```
- Create env variables
    ```bash
    DEEPCAM_ROOT=/lustre/orion/csc547/scratch/keshprad/deepcam/
    PYVENV_ROOT=${DEEPCAM_ROOT}/.venv
    PYVENV_SITEPKGS=${PYVENV_ROOT}/lib/python3.9/site-packages

    cd ${DEEPCAM_ROOT}
    ```
- Create python virtual env
    ```bash
    python -m venv ${PYVENV_ROOT}
    source ${PYVENV_ROOT}/bin/activate
    ```
- Install torch and mpi4py
    ```bash
    # torch==2.5.0
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.1

    MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
    ```
- Install AWS-OCI-RCCL plugin
    ```bash
    mkdir -p ${DEEPCAM_ROOT}/repos
    cd ${DEEPCAM_ROOT}/repos

    rocm_version=6.1.3
    # Load modules
    module load PrgEnv-gnu/8.5.0
    module load rocm/$rocm_version
    module load craype-accel-amd-gfx90a
    module load gcc-native/12.3
    module load cray-mpich/8.1.30
    #module load libtool
    libfabric_path=/opt/cray/libfabric/1.15.2.0

    # Download the plugin repo
    git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
    cd aws-ofi-rccl

    # Build the plugin
    ./autogen.sh
    export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/hip/lib:$LD_LIBRARY_PATH
    PLUG_PREFIX=$PWD

    CC=hipcc CFLAGS=-I/opt/rocm-$rocm_version/rccl/include ./configure \
    --with-libfabric=$libfabric_path --with-rccl=/opt/rocm-$rocm_version --enable-trace \
    --prefix=$PLUG_PREFIX --with-hip=/opt/rocm-$rocm_version/hip --with-mpi=$MPICH_DIR

    make
    make install

    # Reminder to export the plugin to your path
    echo $PLUG_PREFIX
    echo "Add the following line in the environment to use the AWS OFI RCCL plugin"
    echo "export LD_LIBRARY_PATH="$PLUG_PREFIX"/lib:$""LD_LIBRARY_PATH"
    ```
- Install supporting dependencies
    ```bash
    cd ${DEEPCAM_ROOT}

    pip install wandb
    pip install gym
    pip install pyspark
    pip install scikit-learn
    pip install scikit-image
    pip install opencv-python
    pip install wheel
    pip install tomli
    pip install h5py

    # tensorboard
    pip install tensorboard
    pip install tensorboard_plugin_profile
    pip install tensorboard-plugin-wit
    pip install tensorboard-pytorch

    pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
    ```
- Install mlperf-logging
    ```bash
    mkdir -p ${DEEPCAM_ROOT}/repos
    cd ${DEEPCAM_ROOT}/repos

    git clone -b hpc-1.0-branch https://github.com/mlcommons/logging mlperf-logging
    # NOTE: you may need to manually change mlperf-logging/VERSION to a valid version number (e.g. 1.0.0.rc2)
    pip install -e mlperf-logging

    rm ${PYVENV_SITEPKGS}/mlperf-logging.egg-link
    cp -r ./mlperf-logging/mlperf_logging ${PYVENV_SITEPKGS}/mlperf_logging
    cp -r ./mlperf-logging/mlperf_logging.egg-info ${PYVENV_SITEPKGS}/mlperf_logging.egg-info
    ```

#### 2. Download src code
- Download DeepCAM source (linked at top of README)
    ```bash
    # REPLACE WITH YOUR PATH
    PRFX=/lustre/orion/csc569/scratch/keshprad
    DEEPCAM_ROOT=${PRFX}/deepcam

    mkdir -p ${DEEPCAM_ROOT}
    cd ${DEEPCAM_ROOT}

    git clone https://github.com/hpcgroup/hpc.git hpc
    cd ./hpc/deepcam/src/deepCam
    ```

# Download dataset with globus

## Perlmutter
- Refer to [Link](https://gitlab.com/NERSC/N10-benchmarks/deepcam/-/blob/main/data/globus.md)

## Frontier
- [Globus Link](https://app.globus.org/file-manager?origin_id=0b226e2c-4de0-11ea-971a-021304b0cca7&origin_path=%2F)
    - Download to `$DEEPCAM_ROOT/data`