#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --mincpus=10
#SBATCH --mail-user=loay.rashid@students.iiit.ac.in
#SBATCH --mem-per-cpu=2048
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=attr_net_results.txt

module load cuda/10.1
module load cudnn/7-cuda-10.0

python ~/attr_net.py -i 75000 -a 40 -e 25
~                                                                               
~                   
