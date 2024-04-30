#!/usr/local_rwth/bin/zsh

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --account=thes1067
#SBATCH --job-name=merge_csv
#SBATCH --output=/home/eo080593/Projects/2021-dmytro-povaliaiev/batch_job_logs/Blue_Waters_POSIX_Merge_CSVs%J.log

# Redirecting stderr to avoid flood of "incompatible version" messages from Darshan binary lib does not work

SECONDS=0

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"


python_exe=/home/eo080593/Software/miniconda3/bin/python

dataset_path=/work/thes1067/data/blue_waters_dataset

echo "Merging CSVs of the dataset $dataset_path"

# Start the script
srun -n1 /usr/bin/time -v $python_exe Blue_Waters_CSV_Merging_POSIX.py $dataset_path

# Print out the total run time
if (( $SECONDS > 3600 )) ; then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)" 
elif (( $SECONDS > 60 )) ; then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)"
else
    echo "Completed in $SECONDS seconds"
fi