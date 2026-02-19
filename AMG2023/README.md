# AMG2023 README
For more detailed installation parameters, please refer to the [installation document](https://github.com/LLNL/AMG2023/blob/main/amg-doc.pdf).

## Perlmutter Compilation

### Steps to Compile

1. Clone the repository:
    ```sh
    cd AMG2023
    git clone -b v2.32.0 https://github.com/hypre-space/hypre.git
    ```

2. Navigate to the source directory:
    ```sh
    cd hypre/src
    ```

3. Load necessary modules:
    ```sh
    module load PrgEnv-nvidia
    module load gpu
    module load cudatoolkit
    module load craype-accel-nvidia80
    module load nccl
    export CRAY_ACCEL_TARGET=nvidia80
    export MPICH_GPU_SUPPORT_ENABLED=1
    ```

4. Configure the build:
    ```sh
    ./configure --with-cuda --enable-device-memory-pool --enable-mixedint --prefix=/pscratch/sd/c/cunyang/AMG2023 --with-gpu-arch=80
    ```

5. Load additional modules:
    ```sh
    module load cmake/3.30.2
    module load PrgEnv-cray
    module load cudatoolkit
    module load craype-accel-nvidia80
    ```

6. Create and navigate to the build directory:
    ```sh
    cd ../../AMG2023
    mkdir build
    cd build
    ```

7. Run CMake to configure the build system:
    ```sh
    cmake -DHYPRE_PREFIX=/pscratch/sd/c/cunyang/AMG2023 ..
    ```

## Frontier Compilation

### Steps to Compile

1. Load modules
    ```sh
    module reset

    module load cray-mpich/8.1.30
    module load craype-accel-amd-gfx90a
    module load rocm/6.1.3
    export MPICH_GPU_SUPPORT_ENABLED=1

    # load compatible cmake version
    module load Core/24.07
    module load cmake/3.27.9
    ```
2. Configure hypre (v2.32.0)
    - Clone hypre v2.32.0 and navigate to src: 
        ```sh
        git clone -b v2.32.0 https://github.com/hypre-space/hypre.git
        cd hypre/src
        ```
    - Configure hypre (in hypre/src)
        ```sh
        ./configure --with-hip --enable-device-memory-pool --enable-mixedint --with-gpu-arch=gfx90a \
            --with-MPI-lib-dirs="${MPICH_DIR}/lib" --with-MPI-libs="mpi" \
            --with-MPI-include="${MPICH_DIR}/include" \
            CFLAGS="-g -I${ROCM_PATH}/include/ -I${ROCM_PATH}/llvm/include/ \
            -I${ROCM_PATH}/include/rocsparse/" \
            LDFLAGS="-L${ROCM_PATH}/lib/ -L${ROCM_PATH}/llvm/lib/ -lrocsparse"
        ```
    - Compile hypre (in hypre/src)
        ```sh
        # build with make
        make
        ```
3. Configure AMG2023
    - Clone repo: 
        ```sh
        git clone https://github.com/pssg-int/AMG2023`
        cd AMG2023
        ```
    - Add mpiP to LD_LIBRARY_PATH
        ```sh
        export LD_LIBRARY_PATH=/ccs/home/keshprad/mpiP:$LD_LIBRARY_PATH
        ```
    - Configure cmake
        ```sh
        mkdir build && cd build

        cmake .. -DHYPRE_PREFIX=/ccs/home/keshprad/hypre/src/hypre/ \
            -DCMAKE_C_FLAGS="-I${ROCM_PATH}/include/ -I${ROCM_PATH}/llvm/include/ -I${ROCM_PATH}/include/rocsparse/" \
            -DCMAKE_EXE_LINKER_FLAGS="-L${ROCM_PATH}/lib/ -L${ROCM_PATH}/llvm/lib/ -lrocsparse -lrocrand"
        ```
    - Compile AMG2023 (in AMG2023/build)
        ```sh
        make install
        ```
