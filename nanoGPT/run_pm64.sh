#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH -A m2404
#SBATCH --qos=regular
#SBATCH -t 00:30:00
#SBATCH --nodes=64
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive

export HF_HOME="/pscratch/sd/c/cunyang/saved_data"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

module load python
module load cudatoolkit
module load nccl
module load cudnn
source ${SCRATCH}/torch2.5/bin/activate

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
## master addr and port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1
export MPICH_GPU_SUPPORT_ENABLED=0
export FI_CXI_RDZV_EAGER_SIZE=0
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
export OUTPUT_DIR="/pscratch/sd/c/cunyang/result/nanoGPT/64nodes/$TIMESTAMP-job$SLURM_JOB_ID"
mkdir -p ${OUTPUT_DIR}

MIN_MSG_SIZE=$((1048576 * 16)) # 1048576 = 1024 * 1024
MAX_MSG_SIZE=$((1048576 * 2048))

export MPICH_OFI_CXI_COUNTER_REPORT=5
SCRIPT="/pscratch/sd/c/cunyang/micro/all/gpu-benchmarks/nccl/all-reduce/allreduce.x $GPUS $MIN_MSG_SIZE $MAX_MSG_SIZE 100"
run_cmd="srun -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 $SCRIPT >& $OUTPUT_DIR/allreduce.out"
echo $run_cmd
eval $run_cmd

srun -n 256 -c 32 --gpus-per-node=4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none   /global/homes/c/cunyang/gpu-benchmarks/matmul/perlmutter/gemm.x &> "$OUTPUT_DIR/gemm.out"

SCRIPT="train_pm.py config/train_gpt_neox_20B.py"

# run_cmd="srun gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh python -u $SCRIPT &> $OUTPUT_DIR/nanoGPT_withoutprof.out"

# echo $run_cmd
# eval $run_cmd

export WITH_PROFILER=1
run_cmd="srun -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh python -u $SCRIPT &> $OUTPUT_DIR/nanoGPT.out"

echo $run_cmd
eval $run_cmd
