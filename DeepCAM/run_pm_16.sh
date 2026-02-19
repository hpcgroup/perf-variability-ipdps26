#!/bin/bash
#SBATCH --job-name=deepcam-16
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=16
#SBATCH --time=01:00:00
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --output=%x-%j.out
#SBATCH -A m2404
#SBATCH --gpu-bind=none
#SBATCH --exclusive

module purge
module load PrgEnv-nvidia cray-mpich cudatoolkit craype-accel-nvidia80 python nccl cudnn cray-hdf5
source /pscratch/sd/c/cunyang/torch2.5-py3.10/bin/activate
export PYTHONNOUSERSITE=1

output_dir=output_dir
NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
export WORLD_SIZE=$GPUS
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_BLOCKING_WAIT=0
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=1
export FI_CXI_RDZV_THRESHOLD=0
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
export OUTPUT_DIR="/pscratch/sd/c/cunyang/result/deepCAM/16nodes/$TIMESTAMP-job$SLURM_JOB_ID"
mkdir -p ${OUTPUT_DIR}

MIN_MSG_SIZE=$((1048576 * 16)) # 1048576 = 1024 * 1024
MAX_MSG_SIZE=$((1048576 * 2048))

SCRIPT="/pscratch/sd/c/cunyang/micro/all/gpu-benchmarks/nccl/all-reduce/allreduce.x $GPUS $MIN_MSG_SIZE $MAX_MSG_SIZE 100"
run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 $SCRIPT >& $OUTPUT_DIR/allreduce.out"
echo $run_cmd
eval $run_cmd

srun -n 64 -c 32 --gpus-per-node=4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none   /global/homes/c/cunyang/gpu-benchmarks/matmul/perlmutter/gemm.x &> "$OUTPUT_DIR/gemm.out"

export WITH_PROFILER=1
srun --network=no_vni  python -u ./src/deepCam/train.py \
--wireup_method nccl-slurm \
--run_tag $(date +'%y%m%d%H%M%S%N') \
--experiment_id 1 \
--data_dir_prefix /pscratch/sd/c/cunyang/deepcam/data/All-Hist/numpy \
--output_dir output_dir \
--model_prefix segmentation \
--optimizer LAMB \
--start_lr 0.004 \
--lr_schedule type=multistep,milestones=800,decay_rate=0.1 \
--lr_warmup_steps 100 \
--lr_warmup_factor 1. \
--weight_decay 0.01 \
--logging_frequency 0 \
--save_frequency 0 \
--min_epochs 0 \
--max_epochs 2 \
--data_num_threads 8 \
--seed $(date +%s) \
--batchnorm_group_size 2 \
--shuffle_mode global \
--data_format dali-numpy \
--data_oversampling_factor 1 \
--precision_mode amp \
--enable_nhwc \
--local_batch_size 2 \
--local_batch_size_validation 1 \
--disable_tuning \
--disable_comm_overlap \
--enable_graph --enable_groupbn &> "$OUTPUT_DIR/deepcam.out"
