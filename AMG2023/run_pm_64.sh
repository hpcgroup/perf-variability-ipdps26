#!/bin/bash
#SBATCH -N 64
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J amg_64
#SBATCH -t 00:30:00
#SBATCH -A m2404
#SBATCH --exclusive
module purge
module use /global/common/software/m3169/perlmutter/modulefiles
module load PrgEnv-nvidia
module load cudatoolkit
module load nccl/2.21.5
module load craype-accel-nvidia80
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.21.5/lib:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1

BASE_DIR="/pscratch/sd/c/cunyang/result/AMG2023/64nodes"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
#OUTPUT_DIR="$BASE_DIR/$TIMESTAMP"
OUTPUT_DIR="$BASE_DIR/$TIMESTAMP-job$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR"

EXEC=/pscratch/sd/c/cunyang/micro/all/gpu-benchmarks/mpi/all-reduce/allreduce.x
NUM_GPUS=256
MIN_MSG_SIZE=$((1024)) # 1048576 = 1024 * 1024                                                                                                    
MAX_MSG_SIZE=$((1024 * 1024))
ITERATIONS=100
export MPICH_OFI_CXI_COUNTER_REPORT=5
echo "Running with srun..."

srun -n 256 -c 32 --gpus-per-node=4 --gpus-per-task=1 --ntasks-per-node=4  --gpu-bind=none  $EXEC $NUM_GPUS $MIN_MSG_SIZE $MAX_MSG_SIZE $ITERATIONS &> "$OUTPUT_DIR/allreduce.out"
srun -n 256 -c 32 --gpus-per-node=4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none   /global/homes/c/cunyang/gpu-benchmarks/matmul/perlmutter/gemm.x &> "$OUTPUT_DIR/gemm.out"

export HYPRE_INSTALL_DIR=/pscratch/sd/c/cunyang/AMG2023
export MPIP="-o -f $OUTPUT_DIR"

srun -n 256 -c 32 --gpus-per-node=4 --ntasks-per-node=4  --gpus-per-task=1   --gpu-bind=none  ./amg -P 4 8 8 -n 128 64 64 -problem 1 -iter 500 &> "$OUTPUT_DIR/amg.out"

