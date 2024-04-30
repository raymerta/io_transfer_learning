# coding: utf-8
import random
import pandas as pd
from pathlib import Path
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


KFOLD_SEEDS = [728841181, 879843057, 1155483495, 1159944860, 1309364699, 1379701443, 1392436736, 1474235857, 1801054430, 1812549005]

DATASET_LEN = 1288
DATASET_NAME = f"claix_posix_npb_4_16_64_nprocs_Ciao_C_{DATASET_LEN}"
DATASET_DIR = r"/home/thes1067/data/claix_dataset/data/claix"

BW_MODEL_DIR = Path(r"/home/thes1067/models/blue_waters/cross_val/regular/5-fold")

CLAIX_MODEL_DIR = rf"/home/thes1067/models/claix/cross_val/5-fold"

def main():
    config = dict(
        epochs=1200,
        learning_rate=0.001,
        weight_decay=1e-5,
        dropout=0.05,
        shuffle=True,
        nprocs_filter=False,
        random_seed=1234,
        num_folds = 5,
        test_size=0.2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    
    # Fix the seeds
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    torch.manual_seed(config["random_seed"])
    torch.cuda.manual_seed_all(config["random_seed"])

    # Set PyTorch to use deterministic algorithms if possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        run_kfold_with_seed(df_claix_posix, df_bandwidth, kfold_seed, config, device)



def run_kfold_with_seed(df_claix_posix, df_bandwidth, kfold_seed, config, device):
    config["kfold_seed"] = kfold_seed

    for bw_model_fold in range (0,5):
        print(f'\n\n\nBLUE WATERS MODEL FOLD {bw_model_fold}')
        print('--------------------------------')

        kfold = KFold(n_splits=config["num_folds"], shuffle=config["shuffle"], random_state=config["kfold_seed"])

        for fold, (train_ids, test_ids) in enumerate(kfold.split(df_claix_posix, y=df_bandwidth)):
            print(f'CLAIX FOLD {fold}')
            print('--------------------------------')

            run_fold(fold, bw_model_fold, config, df_claix_posix, df_bandwidth, train_ids, test_ids, device)



def run_fold(fold, bw_model_fold, config, df_claix_posix, df_bandwidth, train_ids, test_ids, device):
    config["fold"] = fold
    config["bw_model_fold"] = bw_model_fold

    # Check if a saved model for this fold combo already exists -> don't rerun if yes
    claix_model_name = build_claix_model_name(config, config['fold'], config['bw_model_fold'])
    claix_model_seed_dir = Path(CLAIX_MODEL_DIR, str(config['kfold_seed']))

    if not claix_model_seed_dir.exists():
        claix_model_seed_dir.mkdir()

    claix_model_path = Path(claix_model_seed_dir, claix_model_name)
    if claix_model_path.exists():
        print(f"Found existing model for BW fold {config['bw_model_fold']}, CLAIX fold {config['fold']}. Skipping...")
        return

    BW_model_path = list(Path(BW_MODEL_DIR, str(config['kfold_seed'])).glob(f"*fold_{config['fold']}*.tar"))[0]

    print(f"Model path: {BW_model_path}")
    print(f"Model file exists: {BW_model_path.is_file()}")

    if not BW_model_path.is_file():
        quit()

    config = ConfigStruct(**config)

    # ### Load the pre-trained model
    model = nn.Sequential(
        nn.Linear(97, 2048),
        nn.Dropout(p=config.dropout),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.Dropout(p=config.dropout),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.Dropout(p=config.dropout),
        nn.ReLU(),
        nn.Linear(128, 1),
    ).to(device)

    print("Loading pretrained model...")

    checkpoint = torch.load(BW_model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model_epoch = checkpoint['epoch']

    print(f"Current epoch: {model_epoch}")

    output_layer = list(model.children())[-1]
    output_layer.reset_parameters()
    print("Output layer weights reset")

    training_dataloader, test_dataloader = setup_dataflow(df_claix_posix, df_bandwidth, train_ids, test_ids, config.shuffle, device)

    print(f"Length of the test set: {len(test_ids)}")
    print(f"Length of the training set: {len(train_ids)}")
    print(f"Number of training batches: {len(training_dataloader)}")
    print(f"Number of test batches: {len(test_dataloader)}")

    # # Do the transfer learning
    loss_fn = nn.SmoothL1Loss(reduction="sum").to(device)

    optimizer = optim.Adamax(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    model_epoch = 0
    model.train()

    def train(epoch):
        for batch, (X, y) in enumerate(training_dataloader):
            y_pred = model(X)
            
            # Divide the summed loss by the number of elements in the current batch to get the average loss
            loss = loss_fn(y, y_pred) / len(X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss = loss.item()

        model.train()

    def test():
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item() 

        # Divide the summed test loss by the number of elements in the whole test dataset to get the average loss
        test_loss /= len(test_dataloader.dataset)


        return test_loss

    for epoch in range(model_epoch, config.epochs):
        train(epoch)
        test_loss = test()
        print(f"Epoch {epoch+1} loss: Avg loss: {test_loss:>8f} \n")

        scheduler.step(test_loss)

        model_epoch = epoch

    torch.save({
        'epoch': model_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, claix_model_path)



def setup_dataflow(df_claix_posix, df_bandwidth, train_ids, test_ids, shuffle, device):
    # ### Setup training dataloader
    # Get the rows marked for training
    X_train = df_claix_posix.iloc[train_ids]
    y_train = df_bandwidth.iloc[train_ids]

    # Scale the input features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    tensor_X_train = torch.Tensor(X_train_scaled).to(device)
    tensor_y_train = torch.Tensor(y_train.values).view(-1, 1).to(
    device)  # Transform to a 2D array to avoid shape mismatch (gives errors)

    training_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    # The dataset is so small we can process it whole in one batch
    training_dataloader = DataLoader(training_dataset, batch_size=len(training_dataset), shuffle=shuffle)

    # ### Setup test dataloader 
    # Get the rows marked for test
    X_test = df_claix_posix.iloc[test_ids]
    y_test = df_bandwidth.iloc[test_ids]

    # Use scaling parameters from the training dataset to scale the test one
    X_test_scaled = scaler.transform(X_test)

    tensor_X_test = torch.Tensor(X_test_scaled).to(device)
    tensor_y_test = torch.Tensor(y_test.values).view(-1, 1).to(
    device)  # Transform to a 2D array to avoid shape mismatch (gives errors)

    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
    # The dataset is so small we can process it whole in one batch
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

    return training_dataloader, test_dataloader


def build_claix_model_name(config, fold, bw_fold):
    model_name = f"fold_{bw_fold}-{fold}_{config['loss']}"

    model_name+=f"_{config['learning_rate']}_lr"

    if(hasattr(config, "nprocs")) :
        model_name+= f"_{config['nprocs']}_nprocs"

    model_name+=f"_{config['dropout']}_dropout_pytorch_v{config['pytorch_version']}"

    return model_name

if __name__ == '__main__':
    main()

class ConfigStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)