# This script assumes it is being run by another sbatch script, 
# so does not include portions for SBATCH vars (e.g. account, time, etc.)

# run like: bash /ccs/home/keshprad/gpu-benchmarks/benchmark/frontier/gemm.sh <num_nodes> <output_dir>

#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <rocm_version> <number_of_nodes> <output_dir>"
    exit 1
fi
# `5.7.1` or `6.1.3`
ROCM_VERSION=$1
# `16` or `64`
NUM_NODES=$2
# output directory
OUTPUT_DIR=$3

# setup cray-mpich version
if [[ "$ROCM_VERSION" == "6.1.3" ]]; then
    MPICH_VERSION=8.1.30
else
    MPICH_VERSION=8.1.28
fi

OUTPUT_FILE=$OUTPUT_DIR/output-gemm.log

{
    # reset modules
    echo resetting modules:
    module reset
    # load modules
    echo loading modules:
    module load PrgEnv-cray craype-accel-amd-gfx90a cpe/23.05 amd/${ROCM_VERSION}
    module load cray-mpich/${MPICH_VERSION}
    module load rocm/${ROCM_VERSION}
    module load libfabric/1.20.1
    module list

    GPU_BENCHMARKS_ROOT=/lustre/orion/csc547/scratch/keshprad/gpu-benchmarks
    EXEC=$GPU_BENCHMARKS_ROOT/matmul/frontier/gemm_rocm-${ROCM_VERSION}.x
    NUM_TASKS=$(($NUM_NODES * 8))

    export MPICH_GPU_SUPPORT_ENABLED=1
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
    MIN_MSG_SIZE=$((1024))
    MAX_MSG_SIZE=$((1024 * 32))
    REPEATS=100

    echo start gemm: $(date)
    CMD="srun -N $NUM_NODES -n $NUM_TASKS \
            --gpus-per-node 8 \
            --gpus-per-task 1 \
            --ntasks-per-node 8 \
            --output $OUTPUT_FILE \
            $EXEC $MIN_MSG_SIZE $MAX_MSG_SIZE $REPEATS"
    echo running:
    echo $CMD
    $CMD
    echo end gemm: $(date)
} &>> $OUTPUT_FILE
