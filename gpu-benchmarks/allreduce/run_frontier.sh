# This script assumes it is being run by another sbatch script, 
# so does not include portions for SBATCH vars (e.g. account, time, etc.)

# run like: bash /ccs/home/keshprad/gpu-benchmarks/benchmark/frontier/allreduce.sh <comm_type> <num_nodes> <output_dir>

#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <communication_type> <rocm_version> <number_of_nodes> <output_dir>"
    exit 1
fi
# `mpi` or `rccl`
COMM_TYPE=$1
# `5.7.1` or `6.1.3`
ROCM_VERSION=$2
# `16` or `64`
NUM_NODES=$3
# output directory
OUTPUT_DIR=$4

# setup cray-mpich version
if [[ "$ROCM_VERSION" == "6.1.3" ]]; then
    MPICH_VERSION=8.1.30
else
    MPICH_VERSION=8.1.28
fi

OUTPUT_FILE=$OUTPUT_DIR/output-allreduce.log

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
    EXEC=$GPU_BENCHMARKS_ROOT/allreduce_$COMM_TYPE\_rocm-${ROCM_VERSION}.x
    NUM_TASKS=$(($NUM_NODES * 8))
    ITERATIONS=100
    if [[ "$COMM_TYPE" == "mpi" ]]; then
        MIN_MSG_SIZE=$((1  * 1024))                 # 1 KB
        MAX_MSG_SIZE=$((1  * 1024 * 1024))          # 1 MB
    elif [[ "$COMM_TYPE" == "rccl" ]]; then
        MIN_MSG_SIZE=$((16 * 1024 * 1024))          # 16 MB
        MAX_MSG_SIZE=$((2  * 1024 * 1024 * 1024))   # 2 GB
    fi

    export MPICH_GPU_SUPPORT_ENABLED=1
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

    echo start allreduce: $(date)
    CMD="srun -N $NUM_NODES -n $NUM_TASKS \
            --output $OUTPUT_FILE \
            $EXEC $NUM_TASKS $MIN_MSG_SIZE $MAX_MSG_SIZE $ITERATIONS"
    echo running:
    echo $CMD
    $CMD
    echo end allreduce: $(date)
} &>> $OUTPUT_FILE
