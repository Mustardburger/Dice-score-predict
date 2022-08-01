#!/bin/bash
######## --send email ########
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=phuc@wustl.edu

######## Job Name: Test_Job ########
#SBATCH -J Generate_segs
#SBATCH -o logs_seg/generate_segs.o%j
#SBATCH -e logs_seg/generate_segs.e%j
######## Number of nodes: 1 ########
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres gpu:1,vmem:16gb:1
#SBATCH --mem 32G 
#SBATCH -t 17:00:00
 
cd /mnt/beegfs/home/phuc/my-code/dsc-predict

######## Load module environment required for the job ########
module load cuda/10.2
source activate pytorch

######## Run the job ########
python generate_data.py --aug_rate 0.2