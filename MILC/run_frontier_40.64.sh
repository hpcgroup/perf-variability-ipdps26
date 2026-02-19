#!/bin/bash
#SBATCH --job-name=milc_40.64
#SBATCH --account=csc547
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=64
#SBATCH --ntasks=512
#SBATCH --gpu-bind=none
#SBATCH --exclusive
#SBATCH -t 00:45:00
#SBATCH --output /lustre/orion/csc547/scratch/keshprad/perfvar/MILC_logs/64nodes/%x-%j/job-output.log
#SBATCH --error /lustre/orion/csc547/scratch/keshprad/perfvar/MILC_logs/64nodes/%x-%j/job-error.log
# RUN LIKE: sbatch run_frontier_40.64.sh

echo "start run: $(date)"
# HPE Cassini performance counters: collect network data
export MPICH_OFI_CXI_COUNTER_REPORT=5
ORION_SCRATCH=/lustre/orion/csc547/scratch/keshprad
OUTPUT_DIR=$ORION_SCRATCH/perfvar/MILC_logs/64nodes/$SLURM_JOB_NAME-$SLURM_JOB_ID
MILC_OUTPUT_FILE=$OUTPUT_DIR/output-MILC.log

# Run gpu benchmarks
COMM_TYPE=mpi
ROCM_VERSION=5.7.1
PERF_VARIABILITY_ROOT=/ccs/home/keshprad/perf-variability
echo running allreduce benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allreduce/run_frontier.sh $COMM_TYPE $ROCM_VERSION $SLURM_JOB_NUM_NODES $OUTPUT_DIR
# echo running allgather benchmark
# bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allgather/run_frontier.sh $COMM_TYPE $ROCM_VERSION $SLURM_JOB_NUM_NODES $OUTPUT_DIR
echo running gemm benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/gemm/run_frontier.sh $ROCM_VERSION $SLURM_JOB_NUM_NODES $OUTPUT_DIR

# define paths variables
BENCH_TOPDIR=/ccs/home/keshprad/MILC/OLCF-6_MILC_benchmark
MILC_QCD_DIR=${BENCH_TOPDIR}/build/milc_qcd
exe=${MILC_QCD_DIR}/ks_imp_rhmc/su3_rhmd_hisq
input=$PERF_VARIABILITY_ROOT/MILC/params_frontier.40.64
# Load modules, setup environment
source ${BENCH_TOPDIR}/build/env.sh
module load ums ums023 hpctoolkit
module list &>> "$OUTPUT_DIR/job-output.log"

# Define environment variables
# mpich
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_NEMESIS_ASYNC_PROGRESS=1
# quda
export QUDA_ENABLE_GDR=1
export QUDA_ENABLE_P2P=1
export QUDA_MILC_HISQ_RECONSTRUCT=13
export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9
# omp
export OMP_NUM_THREADS=7
export OMP_PROC_BIND="spread, spread, spread"
export SLURM_CPU_BIND="cores"
# hpctoolkit
export LD_LIBRARY_PATH=${HPCTOOLKIT_ROOT}/lib/hpctoolkit:$LD_LIBRARY_PATH
export HPCTOOLKIT_MEASUREMENT_DIR=$OUTPUT_DIR/hpctoolkit
export HPCTOOLKIT_DB_DIR="$OUTPUT_DIR/hpctoolkit-db"

# qudatune
# Tuning results are stored in qudatune_dir.
export QUDA_RESOURCE_PATH="$ORION_SCRATCH/perfvar/MILC/qudatune_40.64"
if [ ! -d ${QUDA_RESOURCE_PATH} ]; then
    mkdir -p ${QUDA_RESOURCE_PATH}
fi

# mpiP
export LD_LIBRARY_PATH=/ccs/home/keshprad/libdwarf-20210528/lib:$LD_LIBRARY_PATH
export MPIP="-o -f $OUTPUT_DIR"
export MPIP_DLL_PATH=/ccs/home/keshprad/mpiP_rocm-${CRAY_ROCM_VERSION}/libmpiP.so

# log date
cd $PERF_VARIABILITY_ROOT/MILC/
command="srun --export=ALL,LD_PRELOAD=$MPIP_DLL_PATH \
    -n $SLURM_NTASKS -c 7 \
    $exe $input $MILC_OUTPUT_FILE"
echo running milc
echo $command &>> $MILC_OUTPUT_FILE
eval $command &>> $MILC_OUTPUT_FILE

echo end run: $(date)