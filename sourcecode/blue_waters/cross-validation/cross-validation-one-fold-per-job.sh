#!/usr/local_rwth/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:pascal:1
#SBATCH --partition=c16g
#SBATCH --job-name=BW_Full_CV
#SBATCH --account=thes1067

#SBATCH --output=/home/eo080593/Projects/2021-dmytro-povaliaiev/batch_job_logs/blue_waters/CV_Full_Dataset_One_Fold_Pytorch_1.12_%J.log



echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

module load cuda/11.6 cudnn/8.3.2

echo "Running fold $1"
srun -n1 /home/eo080593/Software/miniconda3/envs/pytorch-1.12/bin/python Cross_Validation_Run_One_Fold.py $1