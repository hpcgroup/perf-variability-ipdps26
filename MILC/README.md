# MILC Installation Guide

## Perlmutter Setup
### Prerequisites

Load the necessary modules:
```sh
module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80
module load cmake/3.22.0
module unload xalt
export MPICH_GPU_SUPPORT_ENABLED=1
```

### Setup Download and Installation Paths

Define the paths for the build and installation:
```sh
BUILD_DIR=$(pwd)
MILC_QCD_DIR="${BUILD_DIR}/milc_qcd"
QUDA_DIR="${BUILD_DIR}/quda"
QUDA_BUILD="${BUILD_DIR}/quda/build"
QUDA_INSTALL_PREFIX="${BUILD_DIR}/quda_install"
PATH_TO_CUDA=$CUDA_HOME
PATH_TO_QUDA=$QUDA_INSTALL_PREFIX
```

### Clone and Build QUDA

Clone the QUDA repository and build it:
```sh
git clone --branch develop https://github.com/lattice/quda.git ${QUDA_DIR}
cd ${QUDA_DIR}

mkdir -p ${QUDA_BUILD}
cd ${QUDA_BUILD}

cmake \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DQUDA_GPU_ARCH=sm_80 \
    -DQUDA_DIRAC_DEFAULT_OFF=ON \
    -DQUDA_DIRAC_STAGGERED=ON \
    -DQUDA_FORCE_HISQ=ON \
    -DQUDA_FORCE_GAUGE=ON \
    -DQUDA_MPI=ON \
    -DCMAKE_INSTALL_PREFIX=${QUDA_INSTALL_PREFIX} \
    ../

cmake --build . -- -j$(nproc)
cmake --build . -- install
```

### Clone and Build MILC

Clone the MILC repository and build it:
```sh
git clone --branch develop https://github.com/milc-qcd/milc_qcd.git ${MILC_QCD_DIR}
cd ${MILC_QCD_DIR}

cp Makefile ks_imp_rhmc
cd ks_imp_rhmc

rm -f ../libraries/*.[ao]
make clean

MY_CC=cc \
MY_CXX=CC \
MPP=true \
OMP=true \
PRECISION=1 \
WANT_MIXED_PRECISION_GPU=0 \
CGEOM="-DFIX_NODE_GEOM -DFIX_IONODE_GEOM" \
CTIME="-DCGTIME -DFFTIME -DFLTIME -DGFTIME -DIOTIME" \
KSCGMULTI="-DKS_MULTICG=HYBRID -DMULTISOURCE" \
CUDA_HOME=${PATH_TO_CUDA} \
QUDA_HOME=${PATH_TO_QUDA} \
WANTQUDA=true \
WANT_FN_CG_GPU=true \
WANT_GF_GPU=true \
WANT_GA_GPU=true \
WANT_FL_GPU=true \
WANT_FF_GPU=true \
make -j 1 su3_rhmd_hisq
```

---

For downloading data, refer to [NERSC Lattice QCD Workflow](https://gitlab.com/NERSC/N10-benchmarks/lattice-qcd-workflow/-/tree/main/lattices).

## Frontier Setup
1. Download source code
    ````bash
    wget https://code.ornl.gov/olcf-6_benchmarks/header/-/raw/release-1.0.0/OLCF-6_MILC_benchmark.tar.gz
    tar -xzvf OLCF-6_MILC_benchmark.tar.gz

    cd OLCF-6_MILC_benchmark
    ````
2. Setup paths
    ```bash
    export BENCH_TOPDIR=/ccs/home/keshprad/MILC/OLCF-6_MILC_benchmark
    cd ${BENCH_TOPDIR}
    mkdir build install
    export BUILD_DIR=${BENCH_TOPDIR}/build
    cd ${BUILD_DIR}

    cp -r ${BENCH_TOPDIR}/milc_qcd .
    cp -r ${BENCH_TOPDIR}/quda .
    export QUDA_DIR="${BUILD_DIR}/quda"
    ```
3. Create ${BENCH_TOPDIR}/build/env.sh
    ```bash
    # Set up ROCm-5.7.1
    # Account for the shortcomings of the wrappers
    module reset
    module load cpe/23.12
    module load craype-accel-amd-gfx90a
    module load gcc-mixed/12.2.0
    module load PrgEnv-amd
    module load amd/5.7.1
    module load rocm/5.7.1
    module load cray-mpich/8.1.28
    module load craype/2.7.21
    module load cmake
    module unload darshan-runtime
    module load libfabric/1.20.1
    module list

    ### NO USER SERVICEABLE PARTS BELOW THIS LINE
    export BUILDROOT=${BENCH_TOPDIR}/build
    export INSTALLROOT=${BENCH_TOPDIR}/install

    #Fixup LDpath because of packaging issues
    export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

    # I need these flags because the Cray Wrappers are broken
    # And I want to ensure I use the right versions of all the libraries
    export MPICH_ROOT=${CRAY_MPICH_ROOTDIR}
    export GTL_ROOT=${CRAY_MPICH_ROOTDIR}/gtl/lib
    export MPICH_DIR=${CRAY_MPICH_ROOTDIR}/ofi/amd/5.0

    ## These must be set before running
    export TARGET_GPU=gfx90a

    # Set CFLAGS because I am not using a wrapper
    export MPI_CFLAGS="${CRAY_XPMEM_INCLUDE_OPTS} -I${MPICH_DIR}/include "
    export MPI_LDFLAGS=" ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem -Wl,-rpath=${MPICH_DIR}/lib \
    -L${MPICH_DIR}/lib -lmpi -Wl,-rpath=${GTL_ROOT} -L${GTL_ROOT} -lmpi_gtl_hsa \
    -L${ROCM_PATH}/llvm/lib -Wl,-rpath=${ROCM_PATH}/llvm/lib"

    # Set pore paths and LDPATHs
    export PATH=${ROCM_PATH}/llvm/bin:$PATH
    export LD_LIBRARY_PATH=${ROCM_PATH}/llvm/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${INSTALLROOT}/quda/lib:${LD_LIBRARY_PATH}
    export MPICH_GPU_SUPPORT_ENABLED=1
    ```
4. Build QUDA
    - Changes in `OLCF-6_MILC_benchmark/build/quda/CMakeLists.txt`
        - L320-322 modify to -O2
            ```CMake
            set(CMAKE_CXX_FLAGS_DEVEL
            "-g -O2"
            CACHE STRING "Flags used by the C++ compiler during regular development builds.")
            ```
        - L342-344 modify to -O2
            ```CMake
            set(CMAKE_C_FLAGS_DEVEL
            "-g -O2"
            CACHE STRING "Flags used by the C compiler during regular development builds.")
            ```
    - Compile QUDA
        ```bash
        source ${BENCH_TOPDIR}/build/env.sh
        cd ${BENCH_TOPDIR}/build

        if [ -d ./build_quda ];
        then
        rm -rf ./build_quda
        fi

        mkdir ./build_quda
        cd ./build_quda

        export QUDA_GPU_ARCH=${TARGET_GPU}

        cmake ${BUILDROOT}/quda \
        -G "Unix Makefiles" \
        -DQUDA_TARGET_TYPE="HIP" \
        -DQUDA_GPU_ARCH=${TARGET_GPU} \
        -DROCM_PATH=${ROCM_PATH} \
        -DQUDA_DIRAC_DEFAULT_OFF=ON \
        -DQUDA_DIRAC_STAGGERED=ON \
        -DQUDA_FORCE_GAUGE=ON \
        -DQUDA_FORCE_HISQ=ON \
        -DQUDA_INTERFACE_MILC=ON \
        -DQUDA_MPI=ON \
        -DQUDA_DOWNLOAD_USQCD=ON \
        -DQUDA_DOWNLOAD_EIGEN=ON \
        -DCMAKE_INSTALL_PREFIX=${INSTALLROOT}/quda \
        -DCMAKE_BUILD_TYPE="DEVEL" \
        -DCMAKE_CXX_COMPILER="amdclang++" \
        -DCMAKE_HIP_COMPILER="amdclang++" \
        -DCMAKE_C_COMPILER="amdclang" \
        -DBUILD_SHARED_LIBS=ON \
        -DQUDA_BUILD_SHAREDLIB=ON \
        -DQUDA_BUILD_ALL_TESTS=OFF \
        -DQUDA_CTEST_DISABLE_BENCHMARKS=ON \
        -DCMAKE_C_STANDARD=99 \
        -DCMAKE_CXX_FLAGS="${MPI_CFLAGS} -I${ROCM_PATH}/include \
                    --offload-arch=gfx90a" \
        -DCMAKE_C_FLAGS="${MPI_CFLAGS} -I${ROCM_PATH}/include \
                    --offload-arch=gfx90a" \
        -DCMAKE_SHARED_LINKER_FLAGS="${MPI_LDFLAGS} -L${ROCM_PATH}/lib \
                    -L${ROCM_PATH}/llvm/lib --offload-arch=gfx90a \
                    -lpthread" \
        -DCMAKE_EXE_LINKER_FLAGS="${MPI_LDFLAGS} -L${ROCM_PATH}/lib \
                    -L${ROCM_PATH}/llvm/lib --offload-arch=gfx90a -lpthread"
        cmake --build . -j 32 -v
        cmake --install .
        ```
5. Build MILC
    - Changes in `OLCF-6_MILC_benchmark/build/milc_qcd/Makefile`
        - add following around L10
            ```Makefile
            CTIME = -DCGTIME -DFFTIME -DFLTIME -DGFTIME -DIOTIME
            PRECISION=1
            WANT_MIXED_PRECISION_GPU=0
            MY_CC = amdclang
            MY_CXX = amdclang++
            MPP = true
            OMP = true
            PRECISION = 1
            WANT_MIXED_PRECISION_GPU = 0
            CGEOM = -DFIX_NODE_GEOM -DFIX_IONODE_GEOM
            CTIME = -DCGTIME -DFFTIME -DFLTIME -DGFTIME -DIOTIME
            KSCGMULTI = -DKS_MULTICG=HYBRID -DMULTISOURCE
            CUDA_HOME =
            QUDA_HOME = ${INSTALLROOT}/quda
            WANTQUDA = true
            WANT_FN_CG_GPU = true
            WANT_GF_GPU = true
            WANT_GA_GPU = true
            WANT_FL_GPU = true
            WANT_FF_GPU = true
            OCFLAGS = -std=c99 ${MPI_CFLAGS}
            OCFLAGS += -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas -Wno-unused-function
            OCFLAGS += -fopenmp
            OCXXFLAGS = -std=c++17 ${MPI_CFLAGS}
            LDFLAGS += -fopenmp -L${ROCM_PATH}/lib -lhipfft -lhipblas -lrocblas -lamdhip64 ${MPI_LDFLAGS}
            COMPILER = clang
            OFFLOAD = rocm
            ```
        - Replace OPT (L143) with
            ```Makefile
            OPT              ?= -g -O2
            ```
    - Changes in `OLCF-6_MILC_benchmark/build/milc_qcd/libraries/Make_vanilla`
        - Add following else clause at L39
            ```Makefile
            else ifeq ($(strip ${COMPILER}),clang)
                CC = amdclang
            ```
    - Changes in `OLCF-6_MILC_benchmark/build/milc_qcd/ks_imp_rhmc/control.c`
        - L50-52: add calipers
            ```C
            MPI_Pcontrol(2); // make sure profile data is reset
            MPI_Pcontrol(1); // enable profiling
            hpctoolkit_sampling_start();
            ```
        - L153-155: add calipers
            ```C
            hpctoolkit_sampling_stop();
            MPI_Pcontrol(3); // generate verbose report
            MPI_Pcontrol(0); // disable profiling
            ```
    - Compile MILC
        ```bash
        source ${BENCH_TOPDIR}/build/env.sh

        cd ${BENCH_TOPDIR}/build/milc_qcd/ks_imp_rhmc

        cp ../Makefile .
        make clean

        make su3_rhmd_hisq
        ```