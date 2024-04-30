# coding: utf-8

import random
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch

KFOLD_SEED = 1474235857 # Changed manually for each seed

class ConfigStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def main(fold):
    config = dict(
        epochs=600,
        batch_size=2048,
        learning_rate=0.008,
        weight_decay=1e-5,
        dropout=0.05,
        shuffle=True,
        num_folds = 5,
        test_size=0.2,
        kfold_seed=KFOLD_SEED,
        random_seed=1234,
        fold=fold
    )

    config = ConfigStruct(**config)
    print(f"Currend K-fold seed: {KFOLD_SEED}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    
    # Fix the seeds
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)

    # Set PyTorch to use deterministic algorithms if possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ## Load the data
    DATASET_DIR = rf"/home/thes1067/data/blue_waters_dataset/cross_validation/{config.num_folds}-fold/{KFOLD_SEED}"

    df_fold_train = pd.read_csv(Path(DATASET_DIR, f"fold_{fold}_train.csv"), index_col=0)
    df_fold_test = pd.read_csv(Path(DATASET_DIR, f"fold_{fold}_test.csv"), index_col=0)

    # ### Separate bandwidth from input features

    y_train = df_fold_train.pop('bandwidth')
    y_test = df_fold_test.pop('bandwidth')
    print(f'FOLD {fold}')
    print('--------------------------------')
        
    MODEL_NAME = build_model_name(config, fold)
    MODEL_FILENAME = MODEL_NAME + ".tar"
    MODEL_DIR = Path(rf"/home/thes1067/models/blue_waters/cross_val/regular/{config.num_folds}-fold/{KFOLD_SEED}")
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir()
    MODEL_PATH = Path(MODEL_DIR, MODEL_FILENAME)

    training_dataloader, test_dataloader = setup_dataflow(df_fold_train, y_train, df_fold_test, y_test, config.batch_size, config.shuffle, device)

    # ### Train a fully-connected Multi-Layered Perceptron
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

    # # By default Pytorch returns avg loss per minibatch elements. But since the last batch
    # # (both in training and test) does not have enough instances, sum all the loss across the batches
    # # and then divide it by total number of elements in the the test set.
    loss_fn = nn.SmoothL1Loss(reduction="sum").to(device)

    optimizer = optim.Adamax(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    model_epoch = 0

    model.train()
    
    def train():
        size = len(training_dataloader)
        for batch, (X, y) in enumerate(training_dataloader):
            y_pred = model(X)
            
            # Divide the summed loss by the number of elements in the current batch to get the average loss
            loss = loss_fn(y, y_pred) / len(X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss = loss.item()
                print(f"loss: {loss:>7f} [{batch:>5d}/{size:>5d}]")

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

        print(f"Avg loss: {test_loss:>8f} \n")

        return test_loss

    for epoch in range(model_epoch, config.epochs):  # for epoch in range(model_epoch, model_epoch + config.epochs):

        print(f"Epoch {epoch + 1}\n-------------------------------")
        train()
        test_loss = test()

        scheduler.step(test_loss)

        model_epoch = epoch

    torch.save({
            'epoch': model_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, MODEL_PATH)


def setup_dataflow(X_train, y_train, X_test, y_test, batch_size, shuffle, device):

    # ### Setup training dataloader
    # Scale the input features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    tensor_X_train = torch.Tensor(X_train_scaled).to(device)
    tensor_y_train = torch.Tensor(y_train.values).view(-1, 1).to(
    device)  # Transform to a 2D array to avoid shape mismatch (gives errors)

    training_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle)

    # ### Setup test dataloader 
    # Use scaling parameters from the training dataset to scale the test one
    X_test_scaled = scaler.transform(X_test)

    tensor_X_test = torch.Tensor(X_test_scaled).to(device)
    tensor_y_test = torch.Tensor(y_test.values).view(-1, 1).to(
    device)  # Transform to a 2D array to avoid shape mismatch (gives errors)

    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return training_dataloader, test_dataloader



def build_model_name(config, fold):
    model_name = f"CV_fold_{fold}_{config.loss}"

    model_name+=f"_{config.optimizer}"

    model_name+=f"_{config.test_size}_testSize"

    model_name += f"_{config.scaling}"
    model_name+=f"_{config.batch_size}_batch"
    model_name+=f"_{config.learning_rate}_lr"
        
    if(hasattr(config, "nprocs")):
        model_name+= f"_{config.nprocs}_nprocs"
    
    model_name+=f"_{config.dropout}_dropout_pytorch_v{config.pytorch_version}"

    return model_name


if __name__ == '__main__':
    fold = sys.argv[1]
    main(fold)

