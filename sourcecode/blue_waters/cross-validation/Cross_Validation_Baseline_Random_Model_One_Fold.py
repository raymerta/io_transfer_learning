# coding: utf-8

import random
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch.nn as nn
import torch

def main(fold, seed):
    config = dict(
        shuffle=True,
        stratified_split=False,
        random_seed=1234,
        num_folds = 5,
        test_size=0.2,
        kfold_seed=seed,
        fold=fold
    )

    config = ConfigStruct(**config)

    print(f"Currend K-fold seed: {seed}")

    device = "cpu"
    print(f"Using {device}")
    
    # Fix the seeds
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)

    # ## Load the data
    DATASET_DIR = rf"/home/thes1067/data/blue_waters_dataset/cross_validation/{config.num_folds}-fold/{seed}"

    df_fold_train = pd.read_csv(Path(DATASET_DIR, f"fold_{fold}_train.csv"), index_col=0)
    df_fold_test = pd.read_csv(Path(DATASET_DIR, f"fold_{fold}_test.csv"), index_col=0)

    print(f'FOLD {fold}')
    print('--------------------------------')

    # ### Separate bandwidth from input features
    y_train = df_fold_train.pop('bandwidth')
    y_test = df_fold_test.pop('bandwidth')
    len_y_test = len(y_test)

    tensor_y_test = torch.Tensor(y_test.values)

    bandwidth_q1 = y_train.quantile(0.25)
    bandwidth_q3 = y_train.quantile(0.75)
    bandwidth_iqr = bandwidth_q3 - bandwidth_q1

    # Generate a Tensor with random values in the [Q1, Q3) range
    random_predictions = torch.rand(len(y_test)) * (bandwidth_q3 - bandwidth_q1) + bandwidth_q1
    print(random_predictions)

    loss_fn = nn.SmoothL1Loss(reduction="sum").to(device)
    loss = loss_fn(tensor_y_test, random_predictions)  / len_y_test
    print(loss)


if __name__ == '__main__':
    fold = sys.argv[1]
    seed = sys.argv[2]
    main(fold, seed)


class ConfigStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)