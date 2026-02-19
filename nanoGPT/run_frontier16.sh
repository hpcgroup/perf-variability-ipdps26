#!/bin/bash
#SBATCH -N 16
#SBATCH -n 128
#SBATCH -q normal
#SBATCH -J nanogpt
#SBATCH --gpu-bind none
#SBATCH -t 01:00:00
#SBATCH -A csc547
#SBATCH --output /lustre/orion/csc547/scratch/keshprad/perfvar/nanoGPT_logs/16nodes/%x-%j/job-output.log
#SBATCH --error /lustre/orion/csc547/scratch/keshprad/perfvar/nanoGPT_logs/16nodes/%x-%j/job-error.log
#SBATCH --exclusive
# Run like: sbatch run_frontier16.sh

echo "start run: $(date)"
# HPE Cassini performance counters: collect network data
export MPICH_OFI_CXI_COUNTER_REPORT=5
export JOB_OUTPUT_PATH=/lustre/orion/csc547/scratch/keshprad/perfvar/nanoGPT_logs/16nodes/$SLURM_JOB_NAME-$SLURM_JOB_ID
OUTPUT_FILE=$JOB_OUTPUT_PATH/output-nanoGPT.log
ERROR_FILE=$JOB_OUTPUT_PATH/error-nanoGPT.log

export SCRATCH="/lustre/orion/csc547/scratch/keshprad"
export WRKSPC="${SCRATCH}/nanoGPT"
export HF_HOME="${SCRATCH}/.cache/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
cd $WRKSPC

# load modules
ROCM_VERSION=6.1.3
echo resetting modules:
module reset
echo loading modules:
module load PrgEnv-gnu/8.5.0
module load rocm/${ROCM_VERSION}
module load craype-accel-amd-gfx90a
module load cray-python/3.9.13.1
module load cray-mpich/8.1.30
module load cray-hdf5-parallel/1.12.2.9
module load libfabric/1.20.1
module list
# activate env
source ${WRKSPC}/axonn_nanogpt/bin/activate

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 8 ))
## master addr and port
# setting variables for torch.distributed
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS
export OMP_NUM_THREADS=7

## some RCCL env variables
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_ATS=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0
export CUDA_DEVICE_MAX_CONNECTIONS=1
# AWS-OFI-RCCL
export LD_LIBRARY_PATH="${WRKSPC}/repos/aws-ofi-rccl/lib:$LD_LIBRARY_PATH"
# other
export MPICH_GPU_SUPPORT_ENABLED=1
export GPU_MAX_HW_QUEUES=1
export OFI_NCCL_USE_IPV6_TCP=1

SCRIPT="train_frontier.py config/train_gpt_neox_5B.py"

# run with profiler
export WITH_PERFORMANCE_COUNTERS=0
export WITH_PROFILER=1
OUTPUT_FILE="$JOB_OUTPUT_PATH/output-nanoGPT.log"
# log start date
echo "start nanoGPT: $(date)" &>> $OUTPUT_FILE
run_cmd="srun -N $NNODES -n $GPUS --cpu-bind=cores --gpus-per-node=8 --ntasks-per-node=8 scripts/get_rank.sh python -u $SCRIPT"
echo $run_cmd &>> $OUTPUT_FILE
eval $run_cmd &>> $OUTPUT_FILE
# log end date
echo "end nanoGPT: $(date)" &>> $OUTPUT_FILE

# run with cassini performance counters
export WITH_PERFORMANCE_COUNTERS=1
export WITH_PROFILER=0
OUTPUT_FILE="$JOB_OUTPUT_PATH/output-nanoGPT-with_performance_counters.log"
# log start date
echo "start nanoGPT: $(date)" &>> $OUTPUT_FILE
run_cmd="srun -N $NNODES -n $GPUS --cpu-bind=cores --gpus-per-node=8 --ntasks-per-node=8 scripts/get_rank.sh python -u $SCRIPT"
echo $run_cmd &>> $OUTPUT_FILE
eval $run_cmd &>> $OUTPUT_FILE
# log end date
echo "end nanoGPT: $(date)" &>> $OUTPUT_FILE

# Run gpu benchmarks
COMM_TYPE=rccl
PERF_VARIABILITY_ROOT=/ccs/home/keshprad/perf-variability
echo running allreduce benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allreduce/run_frontier.sh $COMM_TYPE $ROCM_VERSION $SLURM_JOB_NUM_NODES $JOB_OUTPUT_PATH
# echo running allgather benchmark
# bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allgather/run_frontier.sh $COMM_TYPE $ROCM_VERSION $SLURM_JOB_NUM_NODES $JOB_OUTPUT_PATH
echo running gemm benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/gemm/run_frontier.sh $ROCM_VERSION $SLURM_JOB_NUM_NODES $JOB_OUTPUT_PATH

echo "end run: $(date)"