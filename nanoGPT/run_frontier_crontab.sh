#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_nodes>"
    exit 1
fi
# `16` or `64`
NUM_NODES=$1

PERF_VARIABILITY_ROOT=/ccs/home/keshprad/perf-variability

# load lmod
source /usr/share/lmod/lmod/init/bash
# load default LMOD_SYSTEM_DEFAULT_MODULES and MODULEPATH
export LMOD_SYSTEM_DEFAULT_MODULES=craype-x86-trento:craype-network-ofi:perftools-base:xpmem:cray-pmi:PrgEnv-cray:DefApps
export MODULEPATH=/sw/frontier/spack-envs/modules/cce/17.0.0/cray-mpich-8.1.28/cce-17.0.0:/sw/frontier/spack-envs/modules/cce/17.0.0/cce-17.0.0:/sw/frontier/spack-envs/modules/Core/24.07:/opt/cray/pe/lmod/modulefiles/mpi/crayclang/17.0/ofi/1.0/cray-mpich/8.0:/opt/cray/pe/lmod/modulefiles/comnet/crayclang/17.0/ofi/1.0:/opt/cray/pe/lmod/modulefiles/compiler/crayclang/17.0:/opt/cray/pe/lmod/modulefiles/mix_compilers:/opt/cray/pe/lmod/modulefiles/perftools/23.12.0:/opt/cray/pe/lmod/modulefiles/net/ofi/1.0:/opt/cray/pe/lmod/modulefiles/cpu/x86-trento/1.0:/opt/cray/pe/modulefiles/Linux:/opt/cray/pe/modulefiles/Core:/opt/cray/pe/lmod/lmod/modulefiles/Core:/opt/cray/pe/lmod/modulefiles/core:/opt/cray/pe/lmod/modulefiles/craype-targets/default:/sw/frontier/modulefiles:/opt/cray/modulefiles

# run sbatch script
script=$PERF_VARIABILITY_ROOT/nanoGPT/run_frontier$NUM_NODES\.sh
sbatch $script