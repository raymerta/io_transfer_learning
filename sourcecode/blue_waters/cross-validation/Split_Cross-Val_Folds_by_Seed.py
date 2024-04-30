import sys
import numpy as np
import pandas as pd
from pathlib import Path
import random

from sklearn.model_selection import KFold

DATASET_DIR = r"/home/thes1067/data/blue_waters_dataset"
DATASET_NAME = "blue_waters_posix_all_no_outliers"
DATASET_PATH = Path(DATASET_DIR, DATASET_NAME).with_suffix(".csv")

config = dict(
        shuffle=True,
        num_folds = 5,
        test_size=0.2,
        random_seed=1234
    )

def main(kfold_seed):
    print(f"Using split seed {kfold_seed} for k-fold")
    df_blue_waters_posix = pd.read_csv(DATASET_PATH)

    random_seed = 1234
    random.seed(random_seed)
    np.random.seed(random_seed)

    CV_DATASETS_DIR = Path(DATASET_DIR, "cross_validation", "5-fold", str(kfold_seed))
    if not CV_DATASETS_DIR.exists():
        CV_DATASETS_DIR.mkdir()

    kfold = KFold(n_splits=config["num_folds"], shuffle=config["shuffle"], random_state=kfold_seed)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(df_blue_waters_posix, y=df_blue_waters_posix["bandwidth"])):
        print(f'FOLD {fold+1}')
        print('--------------------------------')

        # Get the rows marked for training
        df_fold_train = df_blue_waters_posix.iloc[train_ids]

        df_fold_train.to_csv(Path(CV_DATASETS_DIR, f"fold_{fold+1}_train.csv"))
        print("Training Dataset: done")

        # Get the rows marked for test
        df_fold_test = df_blue_waters_posix.iloc[test_ids]

        df_fold_test.to_csv(Path(CV_DATASETS_DIR, f"fold_{fold+1}_test.csv"))
        print("Test Dataset: done")


if __name__ == '__main__':
    kfold_seed = int(sys.argv[1])
    main(kfold_seed)