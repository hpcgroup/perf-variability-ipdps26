#!/bin/bash
#
#AMG
cd /pscratch/sd/c/cunyang/AMG2023/build
unset ${!SLURM_@};
sbatch submit_64.sh

#DeepCAM
cd /pscratch/sd/c/cunyang/deepcam/optimized_pm/HPE+LBNL/benchmarks/deepcam/implementations/deepcam-pytorch
unset ${!SLURM_@};
sbatch run64.sh

#nanoGPT
cd /pscratch/sd/c/cunyang/nanoGPT
unset ${!SLURM_@};
sbatch run64.sh


# #MILC
cd /pscratch/sd/c/cunyang/milc/lattice-qcd-workflow/benchmarks/medium/generation
unset ${!SLURM_@};
sbatch submit_64.sh