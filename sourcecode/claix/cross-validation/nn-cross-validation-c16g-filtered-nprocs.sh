#!/usr/local_rwth/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:pascal:1
#SBATCH --partition=c16g
#SBATCH --job-name=Claix_Filtered_1/2_CV
#SBATCH --account=thes1067

#SBATCH --output=/home/eo080593/Projects/2021-dmytro-povaliaiev/batch_job_logs/claix/Claix_CV_Filtered_NProcs_%J.log


echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

srun -n1 /home/eo080593/Software/miniconda3/envs/optuna-pytorch-1.12/bin/python Claix_Cross_Validate_Fine-tuning_of_Pre-trained_Model_Filtered_NProcs.py