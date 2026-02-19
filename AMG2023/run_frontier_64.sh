#!/bin/bash
#SBATCH -N 64
#SBATCH -n 512
#SBATCH -q normal
#SBATCH -J amg
#SBATCH --gpu-bind none
#SBATCH -t 00:45:00
#SBATCH -A csc547
#SBATCH --output /lustre/orion/csc547/scratch/keshprad/perfvar/AMG2023_logs/64nodes/%x-%j/job-output.log
#SBATCH --error /lustre/orion/csc547/scratch/keshprad/perfvar/AMG2023_logs/64nodes/%x-%j/job-error.log
#SBATCH --exclusive
# Run like: sbatch run_frontier_64.sh

echo start: $(date)
# HPE Cassini performance counters: collect network data
export MPICH_OFI_CXI_COUNTER_REPORT=5
OUTPUT_DIR=/lustre/orion/csc547/scratch/keshprad/perfvar/AMG2023_logs/64nodes/$SLURM_JOB_NAME-$SLURM_JOB_ID
JOB_OUTPUT_FILE=$OUTPUT_DIR/job-output.log

# Run gpu benchmarks
COMM_TYPE=mpi
ROCM_VERSION=6.1.3
PERF_VARIABILITY_ROOT=/ccs/home/keshprad/perf-variability
echo running allreduce benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allreduce/run_frontier.sh $COMM_TYPE $ROCM_VERSION $SLURM_JOB_NUM_NODES $OUTPUT_DIR
# echo running allgather benchmark
# bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allgather/run_frontier.sh $COMM_TYPE $ROCM_VERSION $SLURM_JOB_NUM_NODES $OUTPUT_DIR
echo running gemm benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/gemm/run_frontier.sh $ROCM_VERSION $SLURM_JOB_NUM_NODES $OUTPUT_DIR

APP_ROOT=/ccs/home/keshprad/AMG2023
cd $APP_ROOT

# reset modules
echo resetting modules:
module reset
# load modules
echo loading modules:
module load ums ums023 hpctoolkit
module load cray-mpich/8.1.30
module load craype-accel-amd-gfx90a
module load rocm/${ROCM_VERSION}
module load libfabric/1.20.1
module list &>> $JOB_OUTPUT_FILE

export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=gfx90a
export HYPRE_INSTALL_DIR=/ccs/home/keshprad/hypre/src/hypre/
# mpiP, libdwarf
export MPIP_DLL_PATH=/ccs/home/keshprad/mpiP_rocm-${CRAY_ROCM_VERSION}/libmpiP.so
export LD_LIBRARY_PATH=/ccs/home/keshprad/libdwarf-20210528/lib:$LD_LIBRARY_PATH
# hpctoolkit
export LD_LIBRARY_PATH=${HPCTOOLKIT_ROOT}/lib/hpctoolkit:$LD_LIBRARY_PATH
export HPCTOOLKIT_MEASUREMENT_DIR=$OUTPUT_DIR/hpctoolkit
export HPCTOOLKIT_DB_DIR="$OUTPUT_DIR/hpctoolkit-db"

OUTPUT_FILE="$OUTPUT_DIR/output-AMG2023.log"
MPIP_OUTPUT_DIR="$OUTPUT_DIR/mpiP"
mkdir -p $MPIP_OUTPUT_DIR
export MPIP="-o -f $MPIP_OUTPUT_DIR"
{
    # log start date
    echo start AMG2023: $(date)
    # define command
    cmd="srun --export=ALL,LD_PRELOAD=$MPIP_DLL_PATH \
        ./build/amg -P 8 8 8 -n 128 64 64 -problem 1 -iter 500"
    echo solving:
    echo $cmd
    $cmd
    # log end date
    echo end AMG2023: $(date)
} &>> $OUTPUT_FILE

# OUTPUT_FILE="$OUTPUT_DIR/output-AMG2023_hpctoolkit.log"
# MPIP_OUTPUT_DIR="$OUTPUT_DIR/mpiP_hpctoolkit"
# mkdir -p $MPIP_OUTPUT_DIR
# export MPIP="-o -f $MPIP_OUTPUT_DIR"
# {
#     # log start date
#     echo start AMG2023: $(date)
#     # define command
#     cmd="srun --export=ALL,LD_PRELOAD=$MPIP_DLL_PATH \
#         hpcrun -ds -o $HPCTOOLKIT_MEASUREMENT_DIR -e CPUTIME -e gpu=amd \
#         ./build/amg -P 8 8 8 -n 128 64 64 -problem 1 -iter 500"
#     echo solving:
#     echo $cmd
#     $cmd
#     # log end date
#     echo end AMG2023: $(date)
# } &>> $OUTPUT_FILE

echo end: $(date)

# # hpctoolkit generate database from measurements and program structure information
# hpcstruct $HPCTOOLKIT_MEASUREMENT_DIR
# hpcprof -o $HPCTOOLKIT_DB_DIR $HPCTOOLKIT_MEASUREMENT_DIR
