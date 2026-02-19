#!/bin/bash
#SBATCH -N 16
#SBATCH -C gpu
#SBATCH -t 01:00:00
#SBATCH -A m2404
#SBATCH --job-name=L96192-gen
#SBATCH -o L96192-gen-gpu.o%j
#SBATCH -q regular
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --exclusive
module purge

module load PrgEnv-nvidia
export CRAY_ACCEL_TARGET=nvidia80

module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80
module unload xalt
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib:$LD_LIBRARY_PATH
N10_MILC="../../.."
MILC_QCD_DIR=${N10_MILC}/build/PM-quda/milc_qcd

#bind=${N10_MILC}/bin/bind4-perlmutter.sh
exe=${MILC_QCD_DIR}/ks_imp_rhmc/su3_rhmd_hisq
input=params.40.16

export QUDA_ENABLE_GDR=1
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_NEMESIS_ASYNC_PROGRESS=1

export QUDA_ENABLE_P2P=1
export QUDA_MILC_HISQ_RECONSTRUCT=13
export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

export OMP_NUM_THREADS=16
export SLURM_CPU_BIND="cores"
export OMP_PROC_BIND="spread, spread, spread"


BASE_DIR="/pscratch/sd/c/cunyang/result/MILC/16nodes"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="$BASE_DIR/$TIMESTAMP-job$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR"

EXEC=/pscratch/sd/c/cunyang/micro/all/gpu-benchmarks/mpi/all-reduce/allreduce.x
NUM_GPUS=64
MIN_MSG_SIZE=$((1024)) # 1048576 = 1024 * 1024
MAX_MSG_SIZE=$((1024 * 1024))
ITERATIONS=100

echo "Running with srun..."

srun -n 64 -c 32 --gpus-per-node=4 --gpus-per-task=1 --ntasks-per-node=4  --gpu-bind=none  $EXEC $NUM_GPUS $MIN_MSG_SIZE $MAX_MSG_SIZE $ITERATIONS &> "$OUTPUT_DIR/allreduce.out"
srun -n 64 -c 32 --gpus-per-node=4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none   /global/homes/c/cunyang/gpu-benchmarks/matmul/perlmutter/gemm.x &> "$OUTPUT_DIR/gemm.out"

export MPIP="-o -f $OUTPUT_DIR"

# Tuning results are stored in qudatune_dir.
qudatune_dir="$PWD/qudatune_40.16.nobind"
export QUDA_RESOURCE_PATH=${qudatune_dir}
if [ ! -d ${qudatune_dir} ]; then
    mkdir ${qudatune_dir}
fi

srun -n $SLURM_NTASKS \
     -c $SLURM_CPUS_PER_TASK \
      $exe $input  "$OUTPUT_DIR/milc.out"

