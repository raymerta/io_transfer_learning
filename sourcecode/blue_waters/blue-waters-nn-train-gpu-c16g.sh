#!/usr/local_rwth/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:pascal:1
#SBATCH --partition=c16g
#SBATCH --job-name=BW_NN_train
#SBATCH --account=thes1067

#SBATCH --output=/home/eo080593/Projects/2021-dmytro-povaliaiev/batch_job_logs/blue_waters/Blue_Waters_%J.log


echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

srun -n1 /home/eo080593/Software/miniconda3/envs/pytorch-1.12/bin/python SmoothL1Loss_Bigger_Batch_Size_Full_Dataset.py