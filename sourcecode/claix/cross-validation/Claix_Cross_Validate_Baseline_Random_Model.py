# coding: utf-8
import random
import pandas as pd
from pathlib import Path
import numpy as np

import torch.nn as nn
import torch

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


KFOLD_SEEDS = [728841181, 879843057, 1155483495, 1159944860, 1309364699, 1379701443, 1392436736, 1474235857, 1801054430, 1812549005]

DATASET_LEN = 1288
DATASET_NAME = f"claix_posix_npb_4_16_64_nprocs_Ciao_C_{DATASET_LEN}"
DATASET_DIR = r"/home/thes1067/data/claix_dataset/data/claix"

device="cpu"
print(f"Using {device}")

def main():
    config = dict(
        shuffle=True,
        stratified_split=False,
        random_seed=1234,
        num_folds = 5,
        test_size=0.2
    )
    
    # Fix the seeds
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    torch.manual_seed(config["random_seed"])
    torch.cuda.manual_seed_all(config["random_seed"])

    dataset_path = Path(DATASET_DIR, DATASET_NAME).with_suffix(".csv")
    print(f"Cross-validating using dataset {DATASET_NAME}")

    # ## Load the data
    df_claix_posix = pd.read_csv(dataset_path)
    print(f"Length: {len(df_claix_posix)}")

    # ### Drop the non-invariant columns
    df_claix_posix = df_claix_posix.drop(['uid', 'jobid', 'hints', 'start_time', 'end_time', 'lib_ver'],
                                                                axis=1)


    # ### Drop columns to match the Blue Waters dataset on which the model was trained
    df_claix_posix = df_claix_posix.drop(['POSIX_FDSYNCS',
                                            'POSIX_RENAMED_FROM',
                                            'POSIX_F_VARIANCE_RANK_TIME',
                                            'POSIX_F_VARIANCE_RANK_BYTES'],
                                            axis=1)			

    # ### Separate bandwidth from input features

    df_bandwidth = df_claix_posix.pop('bandwidth')

    for kfold_seed in KFOLD_SEEDS:
        print(f"Current K-fold seed: {kfold_seed}")
        run_kfold_with_seed(df_claix_posix, df_bandwidth, kfold_seed, config)



def run_kfold_with_seed(df_claix_posix, df_bandwidth, kfold_seed, config): 
    config["kfold_seed"] = kfold_seed

    kfold = KFold(n_splits=config["num_folds"], shuffle=config["shuffle"], random_state=config["kfold_seed"])

    for fold, (train_ids, test_ids) in enumerate(kfold.split(df_claix_posix, y=df_bandwidth)):
        print(f'CLAIX FOLD {fold+1}')
        print('--------------------------------')

        run_fold(fold, config, df_claix_posix, df_bandwidth, train_ids, test_ids)



def run_fold(fold, config, df_claix_posix, df_bandwidth, train_ids, test_ids):
    config["fold"] = fold + 1

    _, _, y_train, y_test = train_test_split(df_claix_posix,
                                                        df_bandwidth,
                                                        test_size=0.2,
                                                        random_state=config["kfold_seed"])

    # Calculate the IQR bounds
    bandwidth_q1 = y_train.quantile(0.25)
    bandwidth_q3 = y_train.quantile(0.75)
    bandwidth_iqr = bandwidth_q3 - bandwidth_q1

    tensor_y_test = torch.Tensor(y_test.values)

    # Generate a Tensor with random values in the [Q1, Q3) range
    random_predictions = torch.rand(len(y_test)) * (bandwidth_q3 - bandwidth_q1) + bandwidth_q1
    print(random_predictions)

    loss_fn = nn.SmoothL1Loss(reduction="sum").to(device)
    loss = loss_fn(tensor_y_test, random_predictions) / len(y_test)
    print(loss)


if __name__ == '__main__':
    main()