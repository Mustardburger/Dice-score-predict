#!/bin/bash
######## --send email ########
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=phuc@wustl.edu

######## Job Name: Test_Job ########
#SBATCH -J Train_dsc_model
#SBATCH -o logs/Train_dsc_model.o%j
#SBATCH -e logs/Train_dsc_model.e%j
######## Number of nodes: 1 ########
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres gpu:1,vmem:32gb:1
#SBATCH --mem 16G 
#SBATCH -t 20:00:00
#SBATCH --reservation=Aris_group
 
cd /mnt/beegfs/home/phuc/my-code/dsc-predict

######## Load module environment required for the job ########
module load cuda/10.2
source activate pytorch

######## Run the job ########
python train.py --data_root /mnt/beegfs/scratch/phuc/seg-quality-control/experiments/ --data_id model_0 --data_train_ratio 0.4 --data_val_ratio 0.3 --num_in_channels 4 --num_epoch 150